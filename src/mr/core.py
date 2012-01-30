#!/usr/bin/env python

"""
p2p MapReduce Engine for Python

Terminology:
    - Peer:
        A running p2p.Peer() process
    - Directory:
        A directory on the shared filesystems where Peers register
        themselves
    - Cluster:
        The set of Peers registered in the same Directory
    - Client:
        A client submitting tasks to the Cluster, and receiving the results.
    - Coordinator:
        The Peer to which a Client connected and submitted a task, that
        coordinates the execution of the Task
    - Worker:
        A process spawned by the Peer to execute a given MR Task
    - Master:
        The Peer process, when discussed in context of communication with
        its worker

General Design:
    - Peers start up independently, and are given the Directory to register
      in. They register by creating a file with their IP and port on which
      they're listening. They prepend some 'busyness statistic' to the
      filename.
    - A Client looks up the least busy Peer in the Directory, and initiates
      a connection. If the connection is refused, the client retries with a
      different Peer until one is found.
    - Once connected, the Client submits a task to the Peer. The submission
      consists of the module filename where the task code resides, the list of
      kernels from that module that are to be executed, extra local
      variables, and the list of items to map.

    - Upon receiving the submission, the Peer to which the Client is
      connected becomes the Coordinator for the Task. It prepends a "startup
      kernel" to the list of Task kernels, which takes as a single item the
      entire list that was passed by the Client, and emits each item. This
      becomes the 0th step of the task. It appends the "return kernel",
      whose purpose is to collect the final return values of the Task. This
      becomes the n+1st step of the task.
      It assigns a 64-bit task ID to the task, the first 32bits being its
      own IP, and the other 32bits being an ever-increasing integer. It
      stores the list of kernels+all other task data into a 'task struct'.
    - The Coordinator then messages itself to initiate the Task. It next
      messages the Gatherer to start step 0 on the task, and step n+1.
      The n+1st step kernel will launch a HTTP server on which the Client
      can listen to and retrieve the results. The address of this server is
      returned to the Client.
      The Coordinator next connects to the returned Gatherer and sends the
      item to map, followed by the END_OF_STEP sentinel.
    - When a Peer is contacted (by the Coordinator) to initiate a Task, it
      launches a Worker+Gatherer+Scatterer for the Task and returns the
      Gatherer address back to the requesting Peer.
    - When a Gatherer is contacted (by the Coordinator) to initiate a step,
      it just sets a flag that it's active for that particular step.

    - When a Gatherer starts up it a) fork()s a Scatterer, b) performs
      user-defined per-Task initialization (calls a function) and c) fork()s
      a Worker
    - When a Gatherer receives a (step, key, value) pair, it stores them into
      its memory-mapped buffer in the form of a linked list. For each key, it
      keeps an in-memory list of the starting offset in the buffer, and the
      number of items in the buffer.
    - When a Gatherer receives an END_OF_STEP sentinel, it marks all keys
      belonging to that step as 'finalized' -- expecting no more data. Once
      all keys belonging to the step have been processed by the Worker, the
      Gatherer responds to the Worker's query for more with an END_OF_STEP
      sentinel for the given step, and removes the step from the list of
      active steps.
    - The Worker queries the Gatherer for the next steps/keys to reduce,
      letting the Gatherer know which steps/keys it is reducing at the
      moment. The Gatherer responds with a list of new items to reduce for
      those steps/keys, and/or new steps/keys to operate on. The worker
      reduces all of these in an interlieved fashon (i.e., round-robins
      calling next() for each of the active reducers). If a returned step
      has never-before been seen, the worker performs any (user-defined) 
      per-step initialization. When the worker receives an END_OF_STEP 
      sentinel for a given step, it queues it to the Scatterer and performs
      any user-defined per-step cleanup.
    - When the Gatherer is asked by the worker for the next items to operate
      on, it can use the opportunity to compactify the memory-mapped buffer
      if this is deemed to be helpful.
    - When a Scatterer encounters the END_OF_STEP sentinel, it messages the
      Coordinator with an EndOfStep message. The Coordinator records that
      this Worker is processing one less step, as well as that there are one
      less workers processing this particular step.
    - Once the number of active Workers for step k drops to zero, the 
      Coordinator emits an END_OF_STEP sentinel to all Gatherers
      active for step k+1. If this was the last step of a Task, the
      Coordinator messages all active Gatherers to shut down, followed by
      deletion the Task-related structures.

    - The Scatterer listens on the Output FIFO. Data items in the Output
      FIFO are key/value pairs. For each new key, the Scatterer contacts the
      Coordinator to obtain the address of the Gatherer to whom to send the
      data. Otherwise it just uses a previously established connection to
      forward the data.
    - The Coordinator, when queried by a Scatterer where to send a datum for
      (task tid, step sid, key k), either returns the already known remote
      Gatherer address for a key (or a hash thereof) it has already seen, or
      selects the next least busy Peer from the Directory. The Coordinator
      attempts to initiate the Task tid on the remote Peer; if unsuccessful,
      retries with the next Peer from the Directory. If successful, the
      remote Peer will respond with the address of the Gatherer on which the
      task tid, step sid, is receiving data.
    - If the Coordinator is queried for the destination to send the last
      step, it returns the location of its own Gatherer. That Gatherer has
      opened an HTTP port on which the Client can listen for the results.
    - The Gatherers periodically report on their progress by messaging the
      Coordinator.

Output FIFO design:
	- Two memory maps. While the Worker writes in one, the Scatterer
	  reads from the other

        - When the Scatterer finishes forwarding its map, it sets the
          scattererWorking Event to False. It then waits on the Event

        - When the Worker is about to write a value, it checks if
	  scattererWorking == False. If False, it replaces the current map
	  with the (now empty) new map. It sets scattererWorking == True to
	  signal the Scatterer to continue

Requirements:
    - Peer event loop has to be reentrant (i.e., a Peer must be able to
      message itself.)

Schematic of a Peer node and the Worker:

         v   ^
         |   |
     -------------
    | Coordinator |
     ------------- 
          | ^
          v |
 ->-\ -----------     --     -------------     --     ----------- /->-
  ->-| Gatherer  |->-|IF|->-|   Worker    |->-|OF|->-| Scatterer |->-
 ->-/ -----------     --     -------------     --     ----------- \->-


"""

import SocketServer
import socket, os, time, sys
import numpy as np
import subprocess as sp
import getopt
import cStringIO
import imp
import xmlrpclib
import asyncore
import BaseHTTPServer
import SimpleXMLRPCServer
import logging
import threading, Queue
import datetime
import glob
import cgi, urllib, urlparse
import cPickle
import random
import tempfile
import mmap
import struct
import platform
import gc
import binascii
import select
import contextlib
import collections
from base64 import b64encode, b64decode
from collections import defaultdict, deque
from heapq import heappop, heappush, heapify
import traceback
import weakref

class ThreadedXMLRPCServer(SocketServer.ThreadingMixIn, SimpleXMLRPCServer.SimpleXMLRPCServer):
	def __init__(self, *args, **kwargs):
		#print >>sys.stderr, kwargs
		try:
			self.daemon_threads = kwargs['daemon']
			del kwargs['daemon']
		except KeyError:
			pass
		SimpleXMLRPCServer.SimpleXMLRPCServer.__init__(self, *args, **kwargs)

# Buffer size -- can't be too big on 32bit platforms (otherwise all mmaps
# won't fit in memory)
BUFSIZE = 100 * 2**20 if platform.architecture()[0] == '32bit' else 200 * 2**30

logger = logging.getLogger('mr')

def RLock(name):
	return threading.RLock()

def Event(name):
	return threading.Event()

def Lock(name):
	return threading.Lock()

if False:
	if True:
		class Lock(object):
			def __init__(self, name):
				self.__name = name
				self.__t = 0.
				self.__n = 0
				self.lock = threading.Lock()

			def acquire(self):
				self.__t0 = time.time()
				self.lock.acquire()
				self.__n += 1

			def release(self, *dummy):
				self.lock.release()
				self.__t += time.time() - self.__t0

			__enter__ = acquire
			__exit__ = release

			def __del__(self):
				logger.info("Lock timing: %s, %.4f, n=%d" % (self.__name, self.__t, self.__n))

		class RLock(threading._RLock):
			def __init__(self, name):
				self.__name = name
				self.__t = 0.
				self.__n = 0
				threading._RLock.__init__(self)

			def acquire(self):
				if not self._is_owned():
					self.__t0 = time.time()
				threading._RLock.acquire(self)
				self.__n += 1

			__enter__ = acquire

			def release(self):
				threading._RLock.release(self)
				if not self._is_owned():
					self.__t += time.time() - self.__t0

			def __del__(self):
				logger.info("RLock timing: %s, %.4f, n=%d" % (self.__name, self.__t, self.__n))

		class Event(threading._Event):
			def __init__(self, name, verbose=None):
				self.__name = name
				self.__t = 0
				self.__n = 0
				threading._Event.__init__(self, verbose)
	
			def __str__(self):
				return "Event %s" % self.__name

			def wait(self, timeout=None):
				t0 = time.time()
				threading._Event.wait(self, timeout)
				self.__t += time.time() - t0
				self.__n += 1

			def __del__(self):
				logger.info("Event wait timing: %s, %.4f, n=%d" % (self.__name, self.__t, self.__n))

	else:
		# Verbose wrappers for debugging
		class Event(threading._Event):
			def __init__(self, name, verbose=None):
				self.__name = name
				threading._Event.__init__(self, verbose)
	
			def __str__(self):
				return "Event %s" % self.__name
	
			def wait(self, timeout=None):
				ct = threading.current_thread()
	
				#logger.debug("[%s] Waiting on %s" % (self, ct))
				while not threading._Event.wait(self, 60):
					logger.debug("[%s] EVENT EXPIRED ON %s" % (self, ct))

		class RLock(threading._RLock):
			def acquire(self):
				#new = not self._is_owned()
				ct = threading.current_thread()
	
				#if new:
				#	logger.debug("[%s] Acquiring for %s" % (self, ct))
	
				succ = False
				while not succ:
					for _ in xrange(10):
						if threading._RLock.acquire(self, blocking=False):
							succ = True
							break
						time.sleep(.1)
					#	logger.debug("[%s] Acquiring for %s (attempt %d)." % (self, ct, retry))
					else:
						logger.debug("[%s] DEADLOCKED IN THREAD %s." % (self, ct))
			
				#if new:
				#	logger.debug("[%s] Acquired in %s." % (self, ct))
	
			__enter__ = acquire
	
			def release(self):
				ret = threading._RLock.release(self)
				#ct = threading.current_thread()
				#if not self._is_owned():
				#	logger.debug("[%s] Releasing on %s" % (self, ct))
				return ret

if os.getenv("PROFILE", 0):
	import cProfile
	outfn = os.getenv("PROFILE_LOG", "profile.log") + '.' + str(os.getpid())
	#cProfile.runctx("server.serve_forever()", globals(), locals(), outfn)

	class Thread(threading.Thread):
		def run(self):
			profiler = cProfile.Profile()
			try:
				return profiler.runcall(threading.Thread.run, self)
			finally:
				profiler.dump_stats('%s.%s.profile' % (outfn, self.name,))
else:
	from threading import Thread

class AsyncoreLogErrorMixIn:
	"""
	Mix-in for asyncore.dispatcher to raise an exception on error
	
	Otherwise it's a horror to debug.
	"""
	def handle_error(self):
		_, t, v, tbinfo = asyncore.compact_traceback()

		# sometimes a user repr method will crash.
		try:
			self_repr = repr(self)
		except:
			self_repr = '<__repr__(self) failed for object at %0x>' % id(self)

		logger.error(
			'uncaptured python exception, closing channel %s (%s:%s %s)' % (
				self_repr,
				t,
				v,
				tbinfo
				)
			)
		self.handle_close()
		raise

def dirdict(self):
	return dict(( (name, getattr(self, name)) for name in dir(self)))

def unpack_callable(func):
	""" Unpack a (function, function_args) tuple
	"""
	func, func_args = (func, ()) if callable(func) or func is None else (func[0], func[1:])
	return func, func_args

def mr_init():
	"""
	This should get called upon import of the Client app,
	and hopefully store the pristine state of its startup
	environment.
	"""
	global fn, cwd, argv, env

	fn = sys.argv[0]
	argv = list(sys.argv)
	cwd = os.getcwd()
	env = dict(os.environ)

class TaskSpec(object):
	fn      = None		# The __main__ filename of the user's task
	cwd	= None		# The current working directory of the user's task
	argv    = None		# Command-line arguments of the user's task
	env     = None	 	# User's task environment

	nitems  = None		# Number of items
	nkernels= None		# Number of kernels
	nlocals = None		# Number of locals

	def __init__(self, fn=None, argv=None, cwd=None, env=None, nitems=0, nkernels=0, nlocals=0):
		self.fn    = fn
		self.cwd   = cwd
		self.argv  = argv
		self.env   = env
		
		self.nitems = nitems
		self.nkernels = nkernels
		self.nlocals = nlocals

	def __str__(self):
		s =  "fn=%s, cwd=%s, argv=%s, len(env)=%s, " % (self.fn, self.cwd, self.argv, len(self.env))
		s += "nitems=%d, nkernels=%d, nlocals=%d" % (self.nitems, self.nkernels, self.nlocals)
		return s

	def serialize(self):
		# Custom serialization format, because we don't want to use cPickle (unsafe)
		# and we can't guarantee there won't be any binary characters in argv or env

		out = cStringIO.StringIO()

		out.write(b64encode(self.fn) + '\n')
		out.write(b64encode(self.cwd) + '\n')

		out.write(str(len(self.argv)) + '\n')
		for v in self.argv: out.write(b64encode(v) + '\n')

		out.write(str(len(self.env)) + '\n')
		for k, v in self.env.iteritems():
			out.write(b64encode(k) + '\n')
			out.write(b64encode(v) + '\n')

		out.write(b64encode("%d %d %d" % (self.nitems, self.nkernels, self.nlocals)) + '\n')

		return out.getvalue()

	@staticmethod
	def unserialize(data):
		task = TaskSpec()

		it = iter(data.split('\n'))

		task.fn  = b64decode(next(it))
		task.cwd = b64decode(next(it))

		n = int(next(it))
		task.argv = [ b64decode(next(it)) for _ in xrange(n) ]

		n = int(next(it))
		task.env = dict( (b64decode(next(it)), b64decode(next(it))) for _ in xrange(n) )

		task.nitems, task.nkernels, task.nlocals = map(int, b64decode(next(it)).split(' '))

		return task

class ConnectionError(Exception):
	pass

class Pool:
	def __init__(self, directory):
		self.directory = directory

	def print_status(self, status, nitems):
		keys_out_prev = nitems
		for stage, keys_in, values_out, keys_out, ended in status:
			try:
				pct = "%6.2f%%" % (100. * keys_in / keys_out_prev)
			except TypeError:
				pct = "       "

			state = "COMPLETED" if ended else "IN PROGRESS"

			print >>sys.stderr, "Stage %2d: %7d keys to %7d values, %s (%s)" % (stage, keys_in, values_out, pct, state)

			keys_out_prev = keys_out

	def print_status1(self, status, nitems):
		keys_out_prev = nitems
		for stage, keys_in, values_out, keys_out, ended in status:
			try:
				pct = "%6.2f%%" % (100. * keys_in / keys_out_prev)
			except TypeError:
				pct = "       "
			keys_out_prev = keys_out

			if ended and stage+1 != len(status):
				continue

			state = "COMPLETED" if ended else "IN PROGRESS"

			sys.stderr.write("\r"+" "*75+"\r")
			sys.stderr.write("Stage %1d/%1d: %7d keys to %7d values, %s (%s)" % (stage+1, len(status), keys_in, values_out, pct, state))
			break

	def _status_stream_thread(self, fp, queue):
		try:
			while True:
				msg, args = cPickle.load(fp)
				queue.put((msg, args))
				#print >>sys.stderr, datetime.datetime.now().ctime(), "[PROGRESS]", msg, args
		except EOFError:
			fp.close()

	def _result_stream_thread(self, rurl, queue):
		rfp = urllib.urlopen(rurl)
		try:
			while True:
				queue.put(("RITEM", cPickle.load(rfp)))
		except EOFError:
			rfp.close()

	def map_reduce_chain(self, items, kernels, locals=[], progress_callback=None):
		# Prepare request
		spec = TaskSpec(fn, argv, cwd, env, len(items), len(kernels), len(locals))
		req = {
			'spec': spec.serialize(),
			'data': b64encode(cPickle.dumps([kernels, locals], -1) + cPickle.dumps(items, -1)),
		      }
		req = urllib.urlencode(req)

		# Choose a random peer
		peers = glob.glob(self.directory + '/*.peer')
		if not len(peers):
			raise ConnectionError('No active peers found in %s' % self.directory)
		purl = os.getenv("PYMR_PEER", file(random.choice(peers)).readline().strip())
		url = purl + "/execute"

		# Submit the task
		fp = urllib.urlopen(url, req)

		# Launch status thread
		queue = Queue.Queue()
		th = threading.Thread(target=self._status_stream_thread, args=(fp, queue))
		th.start()

		# Listen for progress messages: a stream of pickled
		# (msg, args) tuples
		for (msg, args) in iter(queue.get, ("DONE", None)):
			if msg == "STATUS":
				##self.print_status1(args, len(items))
				##status_args = args
				self.print_status1(args, len(items))
			elif msg == "RESULT":
				##sys.stderr.write("\r"+" "*75+"\r")
				#self.print_status(status_args, len(items))
				rurl = args
				rth = threading.Thread(target=self._result_stream_thread, args=(rurl, queue))
				rth.start()
			elif msg == "RITEM":
				yield args
#			else:
#				print >>sys.stderr, datetime.datetime.now().ctime(), "[PROGRESS]", msg, args

		sys.stderr.write('\n')
#		print >>sys.stderr, "EXITING map_reduce_chain"

def _make_buffer_mmap(size, return_file=False):
	# See if we have a user-specified temp directory
	try:
		dir = os.environ["TMP"]
		logger.debug("Using '%s' as the temporary file directory" % dir)
	except KeyError:
		dir = None

	# Create the temporary memory mapped buffer. It will go away as soon as the mmap is closed, or the process exits.
	fp = tempfile.TemporaryFile(dir=dir)
	os.ftruncate(fp.fileno(), size)		# Resize to self.bufsize
	mm = mmap.mmap(fp.fileno(), 0)		# create a memory map

	if not return_file:
		fp.close()				# Close immediately to trigger the unlink()-ing of the unrelying file.
		return mm
	else:
		return mm, fp

class KeyChainSentinelClass(object):
	pass

KeyChainSentinel = KeyChainSentinelClass()

class AckDoneClass(object):
	def __eq__(self, a):
		return isinstance(a, AckDoneClass)

AckDoneSentinel = AckDoneClass()

class AsyncoreLoopChannel(AsyncoreLogErrorMixIn, asyncore.file_dispatcher):
	"""
	A class that can be used to invoke functions from
	within an asyncore loop on the asyncore thread.
	"""
	r_fd = None		# Pipe used to wake up the Scatterer in the asyncore thread (read endpoint)
	w_fd = None		# Pipe used to wake up the Scatterer in the asyncore thread (write endpoint)
	callbacks = None# The callbacks to call when the asyncore loop is entered

	lock = None		# A lock protecting all member variables

	def __init__(self, map):
		self.lock = RLock("AsyncoreLoopChannel")
		self.callbacks = []

		# Note: r_fd is os.dup()-ed by file_dispatcher
		self.r_fd, self.w_fd = os.pipe()
		asyncore.file_dispatcher.__init__(self, self.r_fd, map)

	def close(self):
		# Close both ends of the communication pipe
		#logger.debug("Closing %s %s" % (self.w_fd, self.r_fd))
		os.close(self.w_fd)
		os.close(self.r_fd)

		asyncore.file_dispatcher.close(self)

	def handle_read(self):
		# Called from within the asyncore thread when something is
		# written to w_fd
		#logger.info("In asyncore thread")
		with self.lock:
			# Clear the signal
			nread = 8192
			while self.recv(nread) == nread:
				pass

			# Execute the callbacks
			for callback in self.callbacks:
				callback()
			del self.callbacks[:]

	def schedule(self, callback):
		# Called from outside the asyncore thread to
		# schedule a callback be called from the thread,
		# and to wake the thread up
		with self.lock:
			self.callbacks.append(callback)
			os.write(self.w_fd, '1')

class AsyncoreThread(Thread):
	__callback_channel = None
	map = None

	def __init__(self, asyncore_args=(), asyncore_kwargs={}, *args, **kwargs):
		self.map = {}
		self.__callback_channel = AsyncoreLoopChannel(self.map)

		asyncore_kwargs = asyncore_kwargs.copy()
		asyncore_kwargs['map'] = self.map
		Thread.__init__(self, target=asyncore.loop, args=asyncore_args, kwargs=asyncore_kwargs, *args, **kwargs)
		logger.error(self.name)

	def schedule(self, callback):
		# Schedule a callback to be executed from within the asyncore loop
		self.__callback_channel.schedule(callback)
		#logger.info("Scheduled")

	def close_all(self, ignore_all=False):
		# Close all channels, clear the channel map, and
		# exit the asyncore loop (and therefore the thread).
		#logger.info(self.map)
		self.schedule( lambda: asyncore.close_all(self.map, ignore_all) )

def serialize_message(fp, stage, key, value):
	# Serialize the key/value. This defines the packet format
	# on the wire.
	pkt_beg = fp.tell()
	fp.seek(8, 1)					# 1. [ int64] Payload length (placeholder, filled in 5.)

	payload_beg = fp.tell()
	fp.write(struct.pack('<I', stage))		# 2. [uint32] Destination stage

	cPickle.dump(key, fp, -1)				# 3. [pickle] Key
	cPickle.dump(value, fp, -1)				# 4. [pickle] Value

	end = fp.tell()
	fp.seek(pkt_beg)
	fp.write(struct.pack('<Q', end - payload_beg))	# 5. Fill in the length
	fp.seek(end)

class Worker(object):
	"""
	The class encapsulating a Worker process, and acting
	as an XMLRPC method provider.
	"""

	class Scatterer(object):
		class ScattererChannel:
			"""
			A buffered connection, one per connected gatherer
			"""
			margin = 4 * 2**10	# A margin to ensure there's always enough buffer space to queue an AckDoneSentinel
			bufsize = 512 * 2**10	# 512k buffer

			buf = None		# Output buffer
			at = None		# Read pointer position
			size = None		# Number of bytes in the buffer

			incomplete = False	# Flag set to True when queue() fails to queue the entire packet
			pending_eofs = []	# Stages to which to sent the EOF as soon as possible

			sock = None		# Connected socket

			rdbuf = None					# Input buffer (for ack. messages)
			want_read = 0					# >0 if this socket wants to be recv-d()

			scatterer = None	# Parent Scatterer

			def __init__(self, scatterer, host, port):
				self.buf = memoryview(bytearray(self.bufsize + self.margin))
				self.at = 0
				self.size = 0
				self.scatterer = scatterer

				self.rdbuf = cStringIO.StringIO()
				self.want_read = 0

				self.pending_eofs = []

				self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				self.sock.connect((host, port))
				self.sock.setblocking(0)

				self.scatterer.epoll.register(self.sock, 0)

			def close(self):
				self.scatterer.epoll.unregister(self.sock)
				self.sock.close()
				self.buf = None

			def queue(self, mm, offs, maxlen, extra=0):
				# Store the packet of the remaining length 'maxlen'
				# into the buffer. If we fail to do so, set the
				# incomplete flag, expecting that the caller will
				# retry sometime in the future.

				beg = self.size
				length = min(self.bufsize + extra - beg, maxlen)

				self.buf[beg:beg+length] = mm[offs:offs+length]
				self.size += length

				# Record if we're in "incomplete packet" state
				self.incomplete = length != maxlen

				self._flush_eofs()

				return length

			def queue_eof(self, stage):
				# Record the end-of-stage marker, taking into account
				# that there may be a partially queued packet in the
				# buffer.
				self.pending_eofs.append(stage)

				if not self._flush_eofs():
					logger.warning("Delaying EOF transmission")

			def _flush_eofs(self):
				# Queue pending EOFs, if possible
				if len(self.pending_eofs) == 0 or self.incomplete:
					return False

				# Do this before looping, as self.queue() calls us.
				eofs = self.pending_eofs
				self.pending_eofs = []

				for stage in eofs:
					fp = cStringIO.StringIO()
					serialize_message(fp, stage, AckDoneSentinel, stage)
					pkt = fp.getvalue()
					queued = self.queue(pkt, 0, len(pkt), extra=self.margin)
					assert queued == len(pkt)
					logger.debug("EOF queued for stage=%s" % stage)

				return True

			def send(self):
				self._flush_eofs()

				sent = self.sock.send(self.buf[self.at:self.size])
				self.at += sent

				if self.at == self.size:
					self.at = self.size = 0

			def recv_ack(self):
				data = self.sock.recv(4096)
				self.rdbuf.write(data)

				# Process as much as possible
				stages = []
				if self.rdbuf.tell() >= 4:
					s = self.rdbuf.getvalue()
					rem = s[len(s)-(len(s)%4):]
					logger.debug("len(sata)=%s, len(s)=%s, len(rem)=%s, tell=%s" % (len(data), len(s), len(rem), self.rdbuf.tell()))
					self.rdbuf.truncate(0)
					self.rdbuf.write(rem)

					for at in xrange(0, len(s), 4):
						stage, = struct.unpack('<I', s[at:at+4])
						logger.debug("GOT EOF ACK, stage=%s" % stage)
						stages.append(stage)

				return stages

			def add_want_read(self, inc):
				self.want_read += inc

			def epoll_flags(self):
				eflags = 0
				if self.at < self.size:		eflags |= select.EPOLLOUT
				if self.want_read:		eflags |= select.EPOLLIN
				return eflags

		### Scatterer ###################
		parent = None		# Parent Worker instance
		coordinator = None	# XMLRPC proxy to the Coordinator

		fd_destinations = None	# Map of (fd) -> ScatterChannel
		destinations = None	# Map of (host, port) -> ScatterChannel
		key_destinations = None	# Map of (stage, keyhash) -> ScatterChannel
		stage_destinations=None # Map of (stage) -> set(ScatterChannel)
		known_destinations=None # Map of (stage) -> dict(keyhash: wurl)

		pending_channels = None # A list of channels that need to be added to the asyncore map
		pending_done = None		# A list of stages that have finished (and for which we can release Channels)

		buffers = None
		all_buffers = None

		def __init__(self, parent, curl):
			self.parent = parent
			self.coordinator = xmlrpclib.ServerProxy(curl)

			self.ctl = deque()					# Message passing into the thread
			self.ctl_evt = Event("Scatterer")	# Notification mechanism for this thread
			self.epoll = select.epoll()			# Waiting on sockets

			self.buffers = defaultdict(set)		# Output buffers that the worker writes to (keyed by destination stage)
			self.all_buffers = set()			# Set of all output buffers

			self.known_destinations = defaultdict(dict)				# (stage) -> {(keyhash) -> (wurl)}
			self.stage_destinations = defaultdict(set)				# (stage) -> set(ScatterChannel) map
			self.destinations = weakref.WeakValueDictionary()			# (host, port) -> ScatterChannel map
			#self.fd_destinations = weakref.WeakValueDictionary()		# fd -> ScatterChannel map
			self.fd_destinations = {}									# fd -> ScatterChannel map
			self.key_destinations = weakref.WeakValueDictionary()		# (stage, keyhash) -> ScatterChannel map
			self.local_destinations = weakref.WeakValueDictionary()		# A set of all (stage, keyhash) pairs whose destinations are local

		def _close(self):
			del self.parent
			del self.coordinator
			self.epoll.close()
			del self.epoll

		def _html_(self):
			with self.lock:
				info = """
				<h2>Scatterer</h2>
				<table border=1>
					<tr><th>Destinations</th><td>{ndest}</td></tr>
				</table>
				""".format(ndest=len(self.destinations))

				brows = [
					'<tr><th>{host}:{port}</th>  <td>{cinfo}</td></tr>'
					.format(host=host, port=port, cinfo=chan._html_()) for (host, port), chan in self.destinations.iteritems()
					]
				info += """
				<h3>Scatterer destinations</h3>
				<table border=1>
					<tr><th>Gatherer addr</th> <th>Channel Info</th></tr>
					{brows}
				</table>
				""".format(brows='\n'.join(brows))

			return info

		def _all_acknowledged(self, stage):
			# Called once all connected endpoints acknowledge
			# the receipt of our data. Clean up and
			# notify the Coordinator that this Worker is
			# completely done with stage=stage-1

			logger.debug("All Gatherers ack. receiving data for stage %s" % (stage,))

			# House keeping
			#logger.debug("stage_destinations[%s]=%s" % (stage, self.stage_destinations[stage]))
			#logger.debug("buffers[%s]=%s" % (stage, self.buffers[stage]))
			assert len(self.stage_destinations[stage]) == 0
			assert len(self.buffers[stage]) == 0
			del self.stage_destinations[stage]
			del self.buffers[stage]
			# Close all connections that are serving no-one
			fd_dest = {}
			for scs in self.stage_destinations.itervalues():
				for sc in scs:
					fd_dest[sc.sock.fileno()] = sc
			logger.debug("Replacing fd_destinations: %s -> %s" % (self.fd_destinations, fd_dest))
			self.fd_destinations = fd_dest

			logger.debug("Calling coordinator.stage_ended")
			self.coordinator.stage_ended(self.parent.url, stage-1)
			logger.debug("RETURNED Calling coordinator.stage_ended")

		def queue_from(self, buffer, single_packet=False):
			# Called from run() to transfer data to output buffers.
			# Auto-creates the output buffers/connection when needed, after querying
			# the Coordinator for destinations.
			if len(buffer.lastq):
				buffer.end = buffer.lastq.popleft()

			#logger.debug("buffer.hash = %s, single_packet = %s" % (buffer.hash, single_packet))
			stored = 0
			while buffer.at != buffer.end:
				if buffer.hash is None:
					assert not single_packet
					# Get a new packet
					buffer.hash, = struct.unpack('<I', buffer.mm[buffer.at:buffer.at+4])
					buffer.at += 4 # Move to the beginning of the packet
					buffer.length, = struct.unpack('<Q', buffer.mm[buffer.at:buffer.at+8])
					buffer.length += 8 # The extra is for pkt. len
					buffer.full_length = buffer.length # Full length of the packet

				# Look for the end-of-buffer marker
				if buffer.hash == 0xFFFFFFFF:
					assert not single_packet
					raise EOFError

				# Find the destination channel
				stage = buffer.stage+1
				keyhash = buffer.hash
				if (stage, keyhash) in self.key_destinations:
					# Destination known
					sc = self.key_destinations[(stage, keyhash)]
				else:
					assert not single_packet
					# Unknown (stage, key). Find where to send them
					logger.debug("Getting destination from coordinator")
					if keyhash not in self.known_destinations[stage]:
						self.known_destinations[stage].update(self.coordinator.get_destinations(stage, keyhash))

					wurl = self.known_destinations[stage][keyhash]
					(host, port) = xmlrpclib.ServerProxy(wurl).get_gatherer_addr()

					if (host, port) not in self.destinations:
						# Open a new channel
						sc = self.ScattererChannel(self, host, port)
						self.destinations[(host, port)] = sc
					else:
						sc = self.destinations[(host, port)]

					# Remember for later in aux. lookup tables
					self.stage_destinations[stage].add(sc)
					self.key_destinations[(stage, keyhash)] = sc
					self.fd_destinations[sc.sock.fileno()] = sc
					if self.parent.get_gatherer_addr() == (host, port):
						logger.info("The destination is local.")
						self.local_destinations[(stage, keyhash)] = sc

				# Queue the data to the right buffer
				stored = sc.queue(buffer.mm, buffer.at, buffer.length)
				buffer.at += stored
				buffer.length -= stored

				if buffer.length == 0:
					# We've read the entire packet
					buffer.hash = None
					if single_packet:
						#logger.debug("Single packet")
						break
				else:
					# Exit if we've filled up the buffer
					break

			# Return True if we managed to queue an entire packet, or
			# if we haven't even begun.
			return buffer.hash is None or buffer.length == buffer.full_length

		def register_data_source(self, ob):
			"""
			Register the output buffer as a data source for the
			scatterer.
			"""
			self.ctl.append(lambda ob=ob: (
									self.buffers[ob.stage+1].add(ob),
									self.all_buffers.add(ob)
						))

			ob.evt = self.ctl_evt
			self.ctl_evt.set()

			return ob

		def shutdown(self):
			# Signal the Scatterer to end the run() loop
			def dummy():
				raise StopIteration()
			self.ctl.append(dummy)
			self.ctl_evt.set()

		def run(self):
			"""
			Main loop.
			"""
			try:
				incompletely_buffered = set()
				while True:
					# Check for control messages
					while len(self.ctl):
						callback = self.ctl.popleft()
						callback()
						logger.debug("Running callback, all_buffers=%s" % (self.all_buffers,))
	
					# Get new data (if any) and queue it to the output buffers
					# First try to complete the queuing of any incompletely
					# written packets from the previous iteration
					ib2 = set()
					for buffer in incompletely_buffered:
						if not self.queue_from(buffer, single_packet=True):
							ib2.add(buffer)
					incompletely_buffered = ib2

					for buffer in self.all_buffers:
						try:
							if not self.queue_from(buffer):
								incompletely_buffered.add(buffer)
						except EOFError:
							# We've exhausted this buffer. Remove it
							# from the set
							stage = buffer.stage + 1
							self.buffers[stage].remove(buffer)
							self.ctl.append(lambda buffer=buffer: self.all_buffers.remove(buffer))

							# If all workers processing this stage have ended,
							# enqueue the EOF signal for remote Gatherers
							if len(self.buffers[stage]) == 0:
								for sc in self.stage_destinations[stage]:
									sc.queue_eof(stage)
									sc.add_want_read(1)

								# Handle the case where a stage emitted no data.
								if len(self.stage_destinations[stage]) == 0:
									self._all_acknowledged(stage)

					# Only arm the sockets for which we have data
					n_active = 0
					for fd, sc in self.fd_destinations.iteritems():
						eflags = sc.epoll_flags()
						if eflags == 0:
							continue
						self.epoll.modify(fd, eflags)
						n_active += 1
	
					# If no data is available, sleep until there is some
					if n_active == 0:
						# Wait for new data
						self.ctl_evt.wait()
						self.ctl_evt.clear()
						continue
					
					# Write while there's something to write,
					# or until we make ~10 runs around the loop at which
					# point recheck the buffers (this is for fairness, to prevent
					# the outputing of one big buffer to tie up the others for
					# a long time)
					for _ in xrange(10):
						# Wait for readiness
						events = self.epoll.poll(0.05)
	
						for fd, ev in events:
							sc = self.fd_destinations[fd]

							# Check for disconnects
							if ev & select.EPOLLHUP:
								sc.close()
								self.epoll.modify(fd, 0)
								del self.fd_destinations[fd]
								assert False
								continue
	
							# Write stuff out
							if ev & select.EPOLLOUT:
								sc.send()

							# Read back ACK messages
							if ev & select.EPOLLIN:
								for stage in sc.recv_ack():
									self.stage_destinations[stage].remove(sc)
									sc.add_want_read(-1)
									if(len(self.stage_destinations[stage]) == 0):
										self._all_acknowledged(stage)
	
							# See if we emptied this buffer
							eflags = sc.epoll_flags()
							self.epoll.modify(fd, eflags)
							if not eflags:
								n_active -= 1
	
						if n_active == 0:
							break
			except StopIteration:
				logger.debug("Got the exit signal.")
				pass

	class Gatherer(AsyncoreLogErrorMixIn, asyncore.dispatcher_with_send):
		"""
		Collect incoming (key, value) tuplets from other Workers
		and funnel them into a single stream to be fed to our
		Worker.

		Does this on a per-stage basis (one buffer per stage)
		"""
		class GathererChannel(AsyncoreLogErrorMixIn, asyncore.dispatcher_with_send):
			"""
			A connection to a Scatterer on another Worker.
			
			Receives the next (stage, key, value) tuple from the
			other Worker, and once received stores it into the
			apropriate buffer.
			"""
			parent = None	# Parent Gatherer
			buf = None	# cStringIO instance where we buffer the incoming packet, until it's complete

			buffer_cache = None	# WeakValueDictionary to Buffers (to avoid locking the parent when fetching one)

			def __init__(self, parent, sock, map):
				self.parent = parent
				self.buf = cStringIO.StringIO()
				self.buffer_cache = weakref.WeakValueDictionary()

				asyncore.dispatcher_with_send.__init__(self, sock, map=map)

			def handle_read(self):
				# Receive as much as possible into self.buf, and attempt to
				# process it if the message is complete.
				data = self.recv(1*1024*1024)
#				logger.debug("Receiving data (len=%s)" % (len(data),))
#				data = self.recv(4096)
				if len(data):
					self.buf.write(data)
					self.process_buffer()

			def handle_close(self):
				# The remote Scatterer has closed the connection.
				#logger.debug("Remote scatterer disconnected.")

				# TODO: Handle broken connections
				assert self.buf.getvalue() == '', "Connection broke?"
				
				self.close()

			def process_buffer(self):
				# See if we've received enough data to process one or 
				# more whole messages, and retire them to the parent's
				# buffer if we have.
				buf = self.buf
				size = buf.tell()	# Assumes buf's fileptr is at the end!

				buf.seek(0)	# Rewind to start for processing
				while size - buf.tell() >= 8:
					# 1.) read and process the packet size
					pkt_len, = struct.unpack('<Q', buf.read(8))		# 1. payload length (int64)

					# 2.) Give up if the entire packet hasn't been received yet
					if pkt_len > size - buf.tell():
						buf.seek(-8, 1)
						break

					# 3.) Load and store the packet if we received all of it
					# Hand off the complete packet to the parent Gatherer
					# for storage into the correct buffer OR handle
					# the message right here if the key equals AckDoneSentinel
					end = buf.tell() + pkt_len
					stage, = struct.unpack('<I', buf.read(4))	# 2. stage (uint32)
					key = cPickle.load(buf)				# 3. key (pickled)
					pkl_value = buf.read(end - buf.tell())		# 4. value (pickled)

					# Handle control messages right here
					if key == AckDoneSentinel:
						# Respond by echoing back the stage, in struct-packed format
						self.send(struct.pack('<I', stage))
						logger.debug("ACK EOF stage=%s" % (stage,))
					else:
						# Commit to the apropriate stage Buffer
						try:
							buffer = self.buffer_cache[stage]
						except KeyError:
							buffer = self.buffer_cache[stage] = self.parent.get_or_create_buffer(stage)
						buffer.append(key, pkl_value)
						#logger.debug("stage=%s" % stage)

				# Keep only the unread bits
				s = buf.read()
				buf.truncate(0)
				buf.write(s)

		class Buffer(object):
			lock = None		# Lock protecting mm, chains, next_key
			mm = None		# Buffer memory map
			chains = None		# dict: key -> (first, last).
						#       first is the offset to the first packet, last is the
						#       offset to the 'offset-to-next' field of the last packet
			next_key = None		# see iteritems() for discussion of this variable

			new_key_evt = None	# Event that is set once new keys become available
			new_value_evts = None	# dict: key -> (Event(), size) which becomes set once new
						# values are available for the given key
			all_recvd = False	# True if the stage _before_ the one this Buffer is buffering
						# has ended (i.e., no more key/values are to be received).
			keys_received = 0 # The number of keys we've received

			vtresh = 0; #2**20	# The size of data in bytes that we'll wait to get buffered before
							# we trigger a new_value_evts[key] event and let the _worker know
							# more is available (performance optimization)

			def __init__(self, size, stage):
				self.lock = Lock("Buffer(stage=%s)" % stage)
				self.new_key_evt = Event("new_key_evt")

				self.chains = {}
				self.new_value_evts = {}

				self.mm = _make_buffer_mmap(size)

				# Ensure the first in a buffer chain is always the chain of keys
				self.append(KeyChainSentinel, cPickle.dumps((KeyChainSentinel, 0), -1))
				self.next_key = self.chains[KeyChainSentinel][1]

			def _html_(self):
				with self.lock:
					info = "mm.tell()={mmtell}, next_key={next_key}, all_recvd={all_recvd}" \
						.format(mmtell=self.mm.tell(), **dirdict(self))
				return info

			def append(self, key, pkl_value):
				with self.lock:
					return self._append(key, pkl_value)

			def _append(self, key, pkl_value):
				# Append the pickled value to the chain of values
				# for key 'key'

				# Find/create the chain for the key, append pickled value
				mm = self.mm

				try:
					# Find the value chain of this key
					last = self.chains[key][1]

					# Store the offset to the next link in the chain
					mm[last:last+8] = struct.pack('<Q', mm.tell() - (last+8))
					new_key = False
				except KeyError:
					# This is a new value chain.
					new_key = True
					self.chains[key] = (mm.tell(), 0)

				# Add a link to the chain
				#logger.debug("key=%s, offset=%s, value=%s" % (key, mm.tell(), cPickle.loads(pkl_value)))
				mm.write(struct.pack('<Q', len(pkl_value)))	# 1) Length of the data (uint64)
				mm.write(pkl_value)				# 2) The data (pickled)
				mm.write('\xff'*8)				# 3) Offset to next item in chain (or 0xFFFFFFFFFFFFFFFF, if end-of-chain)

				# Update the 'last' field in the self.chains dict
				self.chains[key] = self.chains[key][0], mm.tell() - 8

				# If this is a new chain, link to it from the primary chain
				# .. unless this is the initialization of the primary chain
				if new_key and key is not KeyChainSentinel:
					self._append(KeyChainSentinel, cPickle.dumps((key, self.chains[key][0]), -1))
					self.keys_received += 1

					# Notify listener (Buffer.itervalues instances) that new keys are available
					self.new_key_evt.set()

				if key in self.new_value_evts:
					# Notify the active iterator that new values are available, if we
					# collected a reasonable quantity of those. Some buffering here should
					# help improve performance
					evt, size = self.new_value_evts[key]
					if size >= self.vtresh:
						#logger.info("Got key=%s val=%s" % (key, cPickle.loads(pkl_value)))
						evt.set()
						size = 0
					else:
						# Update collected size
						size += len(pkl_value)
					self.new_value_evts[key] = evt, size

			def iterate_chain_piece(self, key, first, last):
				"""
				A generator traversing a chain or values from [first, last)
				without locking.

				first is the offset in self.mm where the [len] field of the
				first packet to be read is.
				last is the offset to the [offset-to-next] field of the last
				packet to be read.

				The caller must ensure there's valid data between (first, last)
				and that the data does not change while this generator exists.
				"""
				at = first
				while True:
					len, = struct.unpack('<Q', self.mm[at:at+8])
					at += 8
					v = cPickle.loads(self.mm[at:at+len])
					at += len

					yield v

					# Reached the end of chain?
					if at == last:
						break

					# Load the position of the next packet
					offs, = struct.unpack('<Q', self.mm[at:at+8])
					at += offs + 8

			def itervalues(self, key, at, last, all_recvd = False):
				"""
				A blocking generator yielding values associated with 'key' until
				self.all_recvd is set.

				The caller guarantees that initially there's at least one key available,
				at offset 'at'. When no more values are available, the generator
				blocks until new values become available (and is signaled via kevt).
				"""
				val_evt = None
				try:
					while True:
						# yield the available values
						for v in self.iterate_chain_piece(key, at, last):
							#logger.debug("Consuming %s" % (v,))
							yield v

						# If we already know this stage has ended, no need to
						# enter the lock to check (this is a common case, as
						# usually there are more keys than reducers and we'll
						# usually only have to wait on the first key)
						if all_recvd:
							break

						while True:
							with self.lock:
								# See if more data showed up in the meantime.
								_, new_last = self.chains[key]
								offs, = struct.unpack('<Q', self.mm[last:last+8])
								all_recvd = self.all_recvd
								if val_evt is not None:
									val_evt.clear()

							# No data currently available ?
							if offs == 0xFFFFFFFFFFFFFFFF:
								# All done?
								if all_recvd:
									raise StopIteration()

								# Wait for more
								if val_evt is None:
									val_evt = Event("new_value_evt[%s]" % key)
									with self.lock:
										# Register so that append() and all_received() trip us
										self.new_value_evts[key] = val_evt, 0
								#logger.info("Here")
								val_evt.wait()
								#logger.info("Woken up key=%s" % key)
							else:
								# More data is available
								at = last + 8 + offs
								last = new_last
								break
				except StopIteration:
					pass
				finally:
					if val_evt is not None:
						with self.lock:
							del self.new_value_evts[key]

			def iteritems(self):
				# Find keys not yet being processed, and yield
				# generators that will yield values for those keys
				while True:
					value_gen = None

					with self.lock:
						# self.next_key is a offset to the 'offset-to-next' field
						# of the last processed item (==key) in the KeyChainSentinel key.
						# The reason for this variable is that there may be multiple instances
						# of iteritems() running in multiple _worker threads (i.e., if we're
						# processing more than one key at a time). In that case, we want each
						# instance to get unique keys back.
						# TODO: A cleaner implementation would pack this into a separate
						#       object, that would yield iteritems() iterators.
						nextoffs, = struct.unpack('<Q', self.mm[self.next_key:self.next_key+8])

						if nextoffs != 0xFFFFFFFFFFFFFFFF:
							# Load the key
							pos = self.next_key + nextoffs + 8 + 8	# Skip the [len] field
							at0 = self.mm.tell()
							try:
								self.mm.seek(pos)
								(key, first) = cPickle.load(self.mm)
								self.next_key = self.mm.tell()		# Advance the next_key
							finally:
								self.mm.seek(at0)

							# Create the generator of values from this key chain
							_, last = self.chains[key]
							value_gen = self.itervalues(key, first, last, self.all_recvd)
						else:
							# Check if this stage was marked as done, exit if so
							if self.all_recvd:
								return

					# Note: we're doing this outside of the locked section
					if value_gen is not None:
						yield key, value_gen
					else:
						# No keys to work on. Sleep waiting for a new one to be added.
						self.new_key_evt.wait()
						self.new_key_evt.clear()

			def all_received(self):
				# Expect to receive no more data in this buffer (because
				# the stage feeding it has ended)
				with self.lock:
					assert not self.all_recvd	# Should not receive this more than once
					self.all_recvd = True

					# Wake up all threads waiting for new data to let
					# them know this is it
					self.new_key_evt.set()
					for evt, _ in self.new_value_evts.itervalues():
						evt.set()

					return self.keys_received

		## Gatherer #######################################################3
		parent = None		# Worker instance
		asyncore_map = None	# asyncore map that Gatherer participates in

		port = None		# The port we're listening on for incoming Scatterer connections
		bufsize = BUFSIZE	# Buffer size (per stage)

		buffers = {}		# A dictionary of Buffer instances, keyed by stage
		lock = None		# Lock guarding the buffers variable

		def __init__(self, parent, curl, map):
			self.parent = parent
			self.asyncore_map = map

			self.lock = Lock("Gatherer")
			self.buffers = {}

			# Set up the async server: open a socket to listen on
			# and store the port in self.port
			asyncore.dispatcher_with_send.__init__(self, map=map)
			self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
			for port in xrange(parent.port, 2**16):
				try:
					self.bind(("", port))
					self.port = port
					break
				except socket.error:
					pass

			self.listen(100)

		def _html_(self):
			with self.lock:
				info = """
				<h2>Gatherer</h2>
				<table border=1>
					<tr><th>Port</th><td>{port}</td></tr>
					<tr><th>BUFSIZE</th><td>{bufsize}</td></tr>
				</table>
				""".format(bufsize=self.bufsize, **self.__dict__)

				brows = [
					'<tr><th>{stage}</th>  <td>{binfo}</td></tr>'
					.format(stage=stage, binfo=buffer._html_()) for stage, buffer in self.buffers.iteritems()
					]
				info += """
				<h3>Gatherer buffers</h3>
				<table border=1>
					<tr><th>Stage</th> <th>Buffer Info</th></tr>
					{brows}
				</table>
				""".format(brows='\n'.join(brows))

			return info

		def get_or_create_buffer(self, stage):
			with self.lock:
				return self._get_or_create_buffer(stage)

		def _get_or_create_buffer(self, stage):
			"""
			Return or create a Buffer instance for stage
			
			NOT LOCKED
			"""
			# Find/create the mmap for the requested stage
			if stage in self.buffers:
				buf = self.buffers[stage]
			else:
				#logger.debug("Creating buffer for stage=%d" % stage)
				buf = self.buffers[stage] = self.Buffer(self.bufsize, stage)

			return buf

		def iteritems(self, stage):
			"""
			Return an iterator returning (key, valueiter) pairs
			for each key in stage 'stage'. See the documentation of
			Buffer.iteritems() for more info.
			"""
			with self.lock:
				return self._get_or_create_buffer(stage).iteritems()

		def append(self, stage, key, pkl_value):
			"""
			Store a (key, pkl_value) pair into the buffer for
			the selected stage.
			
			Note: GathererChannels bypass this and call append()
			      directly on Buffers (optimization)
			"""
			with self.lock:
				buffer = self._get_or_create_buffer(stage)

			return buffer.append(key, pkl_value)

		def stage_ended(self, stage):
			# Notification from the coordinator that a particular
			# stage has ended. That means that all buffers one stage
			# later should expect no more data.
			with self.lock:
				buffer = self.buffers[stage+1]

			return buffer.all_received()

		def worker_done_with_stage(self, stage):
			# Called by the local worker thread to signal it's done
			# processing the data in this stage, and that it can
			# be safely discarded.
			logger.info("Discarding buffer for stage=%s" % (stage,))
			with self.lock:
				del self.buffers[stage]

		def handle_accept(self):
			# Accept a new connection to this Gatherer (presumably
			# by another Scatterer)
			pair = self.accept()
			if pair is not None:
				sock, _ = pair
				self.GathererChannel(self, sock, self.asyncore_map)

	class StageRunner(threading.Thread):
		"""
		An object representing a stage being run.
		"""
		class StageThread(threading.Thread):
			"""
			A single thread running a stage, with its output
			buffer. Note that there may be more than one thread
			running the same stage (if the user requests so)
			"""

			class ResultHTTPRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
				def do_GET(self):
					# Just return the results to the user
					self.send_response(200)
					self.send_header("Content-type", "binary/octet-stream")
					self.end_headers()
		
					for v in self.server.valiter:
						cPickle.dump(v, self.wfile, -1)
		
				def log_message(self, format, *args):
					# Need to override this, otherwise log messages will
					# wind up on stderr
					logger.info("ResultHTTPRequestHandler: " + format % args)

			## StageThread ###############################
			bufsize = BUFSIZE 	# buffer size - 1TB (ext3 max. file size is 2TB)
	
			stage = None		# The stage this thread is processing
	
			mm = None			# Memory map buffer used for buffering
			lastq = None		# dequeue containing the last watermark
	
			evt = None			# Event to set to signal when there's more data
			parent = None		# Parent StageRunner instance
			thread_idx = None	# Thread index (0..nthreads for the stage)
	
			# Variables for use by the Scatterer thread
			at = 0		# Read position within the buffer
			end = 0		# Read stop position within the buffer
			hash = None	# The last hash that was read
			length = 0	# The number of bytes remaining to the end of the packet that was last read
	
			values_generated = 0 # The number of generated values
			keys_processed = 0 # The number of processed keys
	
			def __init__(self, parent, thread_idx):
				self.parent = parent
				self.thread_idx = thread_idx

				# Cache frequently used values from the parent Runner
				self.stage = parent.stage
				self.hash_key = parent.hash_key
				self.coordinator = parent.parent.coordinator
				self.kernels = parent.parent.kernels
				self.hostname = parent.parent.hostname

				# Output buffer	
				self.mm = _make_buffer_mmap(self.bufsize)	
				self.lastq = deque(maxlen=1)

				# Support for bypassing TCP if the destination is local
				self.scatterer = parent.parent.scatterer
				self.gatherer = parent.parent.gatherer

				# Initialize the thread
				threading.Thread.__init__(self, name='Stage-%1d-%1d' % (self.stage, thread_idx))
				self.daemon = True

			def queue_eof(self):
				# Queue the EOF marker
				# Runs from the worker thread
				#logger.info("QUEUE EOF stage=%s " % (self.stage,))
				self.hash_key = lambda key: 0xFFFFFFFF
				self.queue((None, None))
	
			def queue(self, kv):
				# Queue the (key, value) into the buffer
				# Runs from the worker thread
				key, value = kv
				keyhash = self.hash_key(key)

				if (self.stage+1, keyhash) in self.scatterer.local_destinations:
					# bypass TCP/IP if we're the destination
					pkl_value = cPickle.dumps(value, -1)
					try:
						buffer = self._gatherer_buffer_cache
					except AttributeError:
						buffer = self._gatherer_buffer_cache = self.gatherer.get_or_create_buffer(self.stage+1)
					buffer.append(key, pkl_value)
				else:
					# Prepend key hash
					self.mm.write(struct.pack('<I', keyhash))	# 0. [uint32] Key hash
		
					# store the message
					serialize_message(self.mm, self.stage+1, key, value)

					# Notify the scatterer
					self.lastq.append(self.mm.tell())
					if not self.evt.is_set():
						self.evt.set()
	
			def _serve_results(self, kv):
				"""
				Open a HTTP port and serve the results back to the client.
				"""
				if False:
					# Force this method to be compiled as a generator
					yield None
		
				_, v = kv
				httpd, port = start_server(BaseHTTPServer.HTTPServer, self.ResultHTTPRequestHandler, 1024, self.hostname)

				with self.coordinator() as coord:
					coord.notify_client_of_result("http://%s:%s" % (self.hostname, port))

				httpd.valiter = v	# The iterator returning the values
				httpd.handle_request()
				logger.info("Results served")

			def run(self):
				# Executes the kernel for a given stage in
				# a separate thread (called from start())
				stage = self.stage
	
				logger.info("Thread %d for stage %s active on %s" % (self.thread_idx, stage, self.parent.parent.url))
				# stage = -1 and stage = len(kernels) are special (feeder and collector kernels)
				# stage = 0 and stage = len(kernels)-1 have wrappers to make them look like reducers
				if stage == -1:
					# Feeder kernel
					def K_start(kv):
						_, v = kv
						items = list(v)
						for k, item in enumerate(items[0]):
							yield k, item
					K_fun, K_args = K_start, ()
				elif stage == len(self.kernels):
					# Server kernel (serves the results back to the user)
					K_fun, K_args = self._serve_results, ()
				else:
					K_fun, K_args = unpack_callable(self.kernels[stage])
	
					if stage == 0:
						# stage = 0 kernel has a thin wrapper removing
						# the keys before passing the values to the mapper
						def K(kv, K_fun, K_args):
							_, v = kv
							for val in v:
								for res in K_fun(val, *K_args):
									yield res
						K_fun, K_args = K, (K_fun, K_args)
	
					if stage == len(self.kernels)-1:
						# last stage has a wrapper keying the outputs
						# to the same value (so they get redirected to
						# the collector)
						def K(kv, K_fun, K_args):
							for res in K_fun(kv, *K_args):
								yield (0, res)
						K_fun, K_args = K, (K_fun, K_args)
	
				# Do the actual work
				for key, valgen in self.gatherer.iteritems(stage):
					for kv in K_fun((key, valgen), *K_args):
						self.values_generated += 1
						self.queue(kv)
					self.keys_processed += 1
					#logger.debug("Processed stage=%s nkeys=%s nvals=%s" % (self.stage, self.keys_processed, self.values_generated))

				# Let the parent know a thread has ended.
				self.parent.unregister_thread(self.thread_idx)

				self.queue_eof()

		## StageRunner ##############################################################
		stage = None
		parent = None		# Parent Worker instance
		nthreads = None		# The total number of threads to be executed for this stage

		threads = None		# dict:thread_idx->StageThread of threads still running
		lock = None			# Lock protecting self.threads

		keys_processed = None	# The number of keys processed by _completed_ threads
		values_generated = None	# The number of values generated by _completed_ threads
		
		monitor_evt = None	# Event used to signal the monitor thread that something happened

		def __init__(self, parent, stage, maxpeers, nthreads):
			self.parent = parent
			self.stage = stage
			self.maxpeers = maxpeers
			self.nthreads = nthreads
			
			self.threads = dict()
			self.lock = Lock("StageRunner-%s" % stage)
			
			self.monitor_evt = Event("StageMonitor-%s" % stage)
			
			self.keys_processed = 0
			self.values_generated = 0

			# Initialize the thread object
			threading.Thread.__init__(self, name='Monitor-%1d' % (self.stage))
			self.daemon = True

		def unregister_thread(self, thread_idx):
			# Remove a thread from the running threads dictionary
			logger.debug("Unregistering thread for stage=%s" % (self.stage,))
			with self.lock:
				th = self.threads[thread_idx]
				del self.threads[thread_idx]

				# Remember the number of keys/values processed, for later
				self.keys_processed, self.values_generated = th.keys_processed, th.values_generated

			# Let the coordinator know a thread has ended
			with self.parent.coordinator() as coord:
				coord.stage_thread_ended(self.parent.url, self.stage)

			# Let the progress thread know something interesting
			# happened
			self.monitor_evt.set()

		def hash_key(self, key):
			# Return a hash for the key. The Coordinator will
			# be queried for the destination based on this hash.
			# The hash may be any value except 0xFFFFFFFF
			keyhash = binascii.crc32(str(hash(key))) % self.maxpeers
			return keyhash

		def _finalize_stage_run(self):
			# Ran from the monitor thread, once all stage threads exit.
			# Let the parent know we're done.
			logger.debug("Here")
			self.parent._unregister_stage_runner(self.stage)

			# Let the gatherer know we're done consuming
			self.parent.gatherer.worker_done_with_stage(self.stage)

		def run(self):
			# Periodically report our progress to the Coordinator,
			# as long as there's something to report.
			while True:
				logger.debug("Sending progress")
				self.report_progress()

				logger.debug("Entering lock")
				with self.lock:
					logger.debug("Testing number of threads")
					if len(self.threads) == 0:
						# All threads have finished
						self._finalize_stage_run()
						break

				# TODO: The delay should be made random
				self.monitor_evt.wait(1)
				logger.debug("Monitor woken up")
				self.monitor_evt.clear()
			logger.debug("Exiting monitor")

		def start(self):
			# Starts up the threads for the stage. Called
			# from run_stage()
			for i in xrange(self.nthreads):
				# Create a new thread
				th = self.StageThread(self, i)

				with self.lock:
					self.threads[i] = th

				# Register our thread's output as the data source for the scatterer
				self.parent.scatterer.register_data_source(th)

				# Run the thread
				th.start()

			# Run the monitor thread
			threading.Thread.start(self)

		def report_progress(self, force=False):
			# Collect progress info from threads processing this
			# stage, and report to the Coordinator
			keys_processed = self.keys_processed
			values_generated = self.values_generated

			with self.lock:
				for th in self.threads.itervalues():
					keys_processed += th.keys_processed
					values_generated += th.values_generated

			# Report back to the coordinator
			with self.parent.coordinator() as coord:
				coord.progress_report(self.parent.url, self.stage, keys_processed, values_generated)

	### Worker ################################
	server  = None		# XMLRPC server instance
	hostname = None		# The hostname of this machine
	port    = None		# The port on which we're listening
	url     = None		# The Worker's XMLRPC server URL

	kernels = None		# The list of kernels to execute
	locals  = None		# Local variables to be made available to the kernels (TODO: not implemented yet)

	curl	= None		# Coordinator XMLRPC server URL	
	gatherer = None		# Gatherer instance
	scatterer = None	# Scatterer instance
	asyncore_thread = None	# AsyncoreThread instance running asyncore.loop with the gatherer and scatterer

	stage_runners = None	# dict:stage->StageRunner for active stages
	
	t_started = None	# Time when we launched

	def _html_(self):
		with self.lock:
			"""
			Return a HTML info page about this Worker
			"""
			uptime = datetime.datetime.now() - self.t_started
	
			info = """
			<h1>Worker {hostname}:{port}</h1>
			
			<h2>Info</h2>
			<table border=1>
				<tr><th>XMLRPC URL</th><td>{url}</td></tr>
				<tr><th>Uptime</th><td>{uptime}</td></tr>
				<tr><th>Hostname</th><td>{hostname}</td></tr>
				<tr><th>Port</th><td>{port}</td></tr>
				<tr><th>Coordinator</th><td><a href='{curl}'>{curl}</a></td></tr>
			</table>
			""".format(uptime=uptime, **self.__dict__)

			srows = [ 
				'<tr><th>{stage}</th>  <td>{nthreads}</td></tr>'
				.format(stage=stage, nthreads=nthreads) for stage, nthreads in self.running_stage_threads.iteritems()
				]
			info += """
			<h2>Running stages</h2>
			<table border=1>
				<tr><th>Stage</th> <th>nthreads</th></tr>
				{srows}
			</table>
			""".format(srows='\n'.join(srows))

		info += self.gatherer._html_()
		info += self.scatterer._html_()

		return info

	_coordinator_pool = None
	@contextlib.contextmanager
	def coordinator(self):
		"""
		Manage a pool of xmlrpclib.ServerProxy instances
		connecting to the coordinator.
		
		The point of this function is to be thread safe.
		"""
		try:
			try:
				coord = self._coordinator_pool.popleft()
			except AttributeError:
				# Auto-create the container
				self._coordinator_pool = collections.deque(maxlen=5)
				raise IndexError
		except IndexError:
			# Auto-create the object
			coord = xmlrpclib.ServerProxy(self.curl)

		yield coord

		# Return to the pool
		self._coordinator_pool.append(coord)

	def stat(self):
		# Return basic statistics about this server
		with self.lock:
			return {
				'url': self.url,
				't_started': self.t_started,
				't_now': datetime.datetime.now(),
				'running_stages': self.running_stage_threads.items()
			}

	def __init__(self, server, hostname):
		self.server = server

		self.hostname, self.port = hostname, server.server_address[1]
		self.url = 'http://%s:%s' % (self.hostname, self.port)

		self.stage_runners = dict()
		self.lock      = RLock("Worker")

		# Time when we launched
		self.t_started = datetime.datetime.now()

	def get_gatherer_addr(self):
		# Return the port on which our Gatherer listens
		return (self.hostname, self.gatherer.port)

	def initialize(self, curl, data):
		# Initialize the connection back to the Coordinator
		# WARNING: This routine MUST NOT call back the Coordinator
		#          or else it will deadlock
		self.curl = curl

		# Initialize Gatherer and Scatterer threads, with the asyncore
		# loop running in a separate thread
		self.asyncore_thread = AsyncoreThread(name='Gather', asyncore_kwargs={'timeout': 3600})
		self.gatherer  = self.Gatherer(self, curl, map=self.asyncore_thread.map)
		self.asyncore_thread.daemon = True
		self.asyncore_thread.start()

		self.scatterer = self.Scatterer(self, curl)
		self.scatterer_thread = Thread(name='Scatter', target=self.scatterer.run)
		self.scatterer_thread.daemon = True
		self.scatterer_thread.start()

		# Initialize the task:
		fp = cStringIO.StringIO(b64decode(data))
		[ self.kernels, self.locals ] = cPickle.load(fp)

		# Place the (pickled) items on the gatherer's queue
		if True:
			self.gatherer.append(-1, 0, fp.read())
		else:
			items = cPickle.load(fp)
			for item in items:
				self.gatherer.append(-1, 0, cPickle.dumps(item, -1))
		self.stage_ended(-2)

	def run_stage(self, stage, maxpeers):#, nthreads=1, npeers=100):
		# Start running a stage, in a separate thread
		# WARNING: This routine must not call back into the
		#          Coordinator (will deadlock)
		logger.debug("Starting stage %s on %s (maxpeers=%s)" % (stage, self.url, maxpeers))
#		if stage != -1:
#			return
		with self.lock:
			logger.debug("Entered lock")
			assert -1 <= stage <= len(self.kernels)
			assert stage not in self.stage_runners
			sr = self.stage_runners[stage] = self.StageRunner(self, stage, maxpeers, 1)

		sr.start()

	def _unregister_stage_runner(self, stage):
		# Called by StageRunner._monitor when all stage threads end
		with self.lock:
			logger.debug("Unregistering runner for stage=%s" % stage)
			assert stage in self.stage_runners
			del self.stage_runners[stage]

	def stage_ended(self, stage):
		# Notification from the coordinator that a particular
		# stage has ended. Pass it on to the Gatherer.
		##assert stage not in self.stage_runners, stage
		return self.gatherer.stage_ended(stage)

	def shutdown(self):
		# Command from the Coordinator to shut down
		logger.info("Shutting down worker %s" % (self.url,))

		# Wait for remaining stage runners, as they may have not
		# exited yet
		with self.lock:
			stage_runners = dict(self.stage_runners)
		for runner in stage_runners.itervalues():
			logger.debug("Joining runner %s" % runner.name)
			runner.join()

		with self.lock:
			# Assert all jobs have finished
			assert len(self.stage_runners) == 0, str(self.stage_runners)

			# Shut down the scatterer thread
			self.scatterer.shutdown()
			self.scatterer_thread.join(10)
			if self.scatterer_thread.is_alive():
				logger.error("Scatterer thread still alive.")
			else:
				logger.info("Shut down the scatterer thread")

			# Close the Gatherer and the Scatterer
			#logger.info("Caling asyncore_thread.close_all")
			self.asyncore_thread.close_all()
			#logger.info("Waiting for asyncore thread to join")
			self.asyncore_thread.join(10)
			if self.asyncore_thread.is_alive():
				logger.error("Asyncore thread still alive. asyncore_thread.map=%s" % (self.asyncore_thread.map))
			else:
				logger.info("Shut down the asyncore thread")

		self.server.shutdown()

		# Help the garbage collector a bit
		self.scatterer._close()
		del self.scatterer
		del self.gatherer

class Peer:
	class _Coordinator(object):
		class WorkerProxy(xmlrpclib.ServerProxy):
			url                   = None	# URL of the Worker
			purl		 	     = None	# URL of the Worker's Peer
			running_stage_threads = None	# defaultdict(int): the number of threads running each stage on the Worker
			nkeys                 = None	# defaultdict(int): the number of keys assigned to this worker, per stage
			process               = None	# subprocess.Popen class for local workers, None for remote workers
			processing_status = None # dict:stage->count -- The number of keys already processed

			def __eq__(self, other):
				return self.url == other.url

			def __hash__(self):
				return hash(self.url)

			def __lt__(self, other):
				return self.url < other.url

			#def __getattr__(self, name):
			#	logger.debug("GETATTR: %s" % name)
			#	return xmlrpclib.ServerProxy.__getattr__(self, name)

			def run_stage(self, stage, *args, **kwargs):
				"""
				Thin layer over the run_stage RPC that records
				the run in running_stage_threads
				"""
				self.running_stage_threads[stage] += 1
				self.processing_status[stage] = np.zeros(2, dtype=int)
				ret = self.__getattr__('run_stage')(stage, *args, **kwargs)

				for worker in self.parent.workers.itervalues():
					if worker is self: continue
					if stage in worker.running_stage_threads:
						break
				else:
					self.parent._progress("STAGE_STARTED", (stage,))
				self.parent._progress("STAGE_STARTED_ON_WORKER", (self.url, stage))

				return ret

			def __init__(self, parent, url, purl, process=None):
				xmlrpclib.ServerProxy.__init__(self, url)

				self.parent = parent
				self.url = url
				self.purl = purl
				self.process = process
			
				self.running_stage_threads = defaultdict(int)
				self.nkeys = defaultdict(int)
				self.processing_status = {}

		## _Coordinator ####################
		id		= None	# Unique task ID
		server  = None  # XMLRPCServer instance for this Coordinator (note: this is _not_ the Peer that launched it)
		hostname= None  # Our host name
		port    = None  # TCP port
		url     = None  # XMLRPC URL of our server
		purl	= None	# Parent peer URL

		spec    = None	# TaskSpec instance describing the task
		data    = None  # Serialized task data, for the workers

		workers = None	# dict: wurl -> WorkerProxy for workers working on the task
		worker_heap = None # list of (nkeys, worker) acting as a heap, to quickly find a least busy worker
		queue   = None  # Queue for the Coordinator to message the launching Peer with task progress
		pserver = None  # XMLRPC Proxy to the parent Peer that launched us

		destinations = None	# destinations[stage][key] gives the WorkerProxy that receives (stage, key) data
		maxpeers = None # dict:stage -> maxpeers

		all_peers = None		# The set of all peers
		free_peers = None		# The set of currently unused peers
		free_peers_last_refresh = 0	# The last time when free_peers was refreshed from the Directory

		lock    = None	# Lock protecting instance variables
		t_started = None # Time (datetime) when we were started
		
		ended_stages = None # dict: stages->nkeys_produced for stages that have ended

		def progress_report(self, wurl, stage, keys_processed, values_generated):
			with self.lock:
				worker = self.workers[wurl]
				worker.processing_status[stage][:] = (keys_processed, values_generated)

				self._report_status_to_client()

		def _report_status_to_client(self, force=False):
			# Message the client about progress
			# TODO: Split this out into a separate thread.
			with self.lock:
				try:
					tprog = self.tprog
				except AttributeError:
					tprog = self.tprog = 0

			if force:
				tprog = 0

			now = time.time()
			if now > tprog + 1:
				stage_status = defaultdict(lambda: np.zeros(2, dtype=int))
				with self.lock:
					# Sum up how we're progressing
					for worker in self.workers.itervalues():
						for stage, status in worker.processing_status.iteritems():
							stage_status[stage] += status
					# Add the information about which stages have ended
					# and construct the report
					report = []
					for stage in xrange(0, self.spec.nkernels):
						try:
							keys_out = self.ended_stages[stage]
							ended = True
						except KeyError:
							keys_out, ended = None, False
						keys_in, values_out = stage_status[stage] if stage in stage_status else (0, 0)
						report.append((stage, keys_in, values_out, keys_out, ended))
				self._progress("STATUS", report)
				self.tprog = time.time()

		def _html_(self):
			with self.lock:
				"""
				Return a HTML info page about this Coordinator
				"""
				uptime = datetime.datetime.now() - self.t_started
		
				info = """
				<h1>Coordinator {hostname}:{port}</h1>
				
				<h2>Info</h2>
				<table border=1>
					<tr><th>XMLRPC URL</th><td>{url}</td></tr>
					<tr><th>Uptime</th><td>{uptime}</td></tr>
					<tr><th>Hostname</th><td>{hostname}</td></tr>
					<tr><th>Port</th><td>{port}</td></tr>
					<tr><th>Task ID</th><td>{id}</td></tr>
					<tr><th>Parent Peer</th><td><a href='{purl}'>{purl}</a></td></tr>
				</table>
				""".format(uptime=uptime, **self.__dict__)

				wrows = [ 
					'<tr><th><a href="{wurl}">{wurl}</a></th>  <td>{t_started}</td> <td>{t_now}</td> <td>{running_stages}</td>  </tr>'
					.format(wurl=wurl, **worker.stat()) for wurl, worker in self.workers.iteritems()
					]
				info += """
				<h1>Workers</h1>
				<table border=1>
					<tr><th>Worker</th> <th>Started On</th> <th>Local Time</th> <th>Running stages</th></tr>
					{crows}
				</table>
				""".format(crows='\n'.join(wrows))
		
			return info

		def __init__(self, server, hostname, parent_url, id, spec, data):
			self.lock    = RLock("Coordinator")

			self.pserver = xmlrpclib.ServerProxy(parent_url)
			self.purl    = parent_url
			self.server  = server

			self.hostname, self.port = hostname, server.server_address[1]
			self.url = 'http://%s:%s' % (self.hostname, self.port)

			self.id      = id
			self.spec    = TaskSpec.unserialize(spec)
			self.data    = data
			self.queue   = Queue.Queue()
			self.workers = {}
			self.worker_heap = []
			self.destinations = defaultdict(dict)
			self.maxpeers = dict()
			
			self.ended_stages = {}

			# Time when we launched
			self.t_started = datetime.datetime.now()

		def __str__(self):
			s = self.__class__.__name__ + ":\n"
			for var in self.__dict__:
				vala = str(getattr(self, var)).split('\n')
				val = vala[0] + '...' if len(vala) != 1 else vala[0]
				if len(val) > 120:
					val = val[:117] + '...'
				s += "    %s: %s\n" % (var, val)
			return s

		def _progress(self, cmd, msg):
			# Send a message to the calling thread
			# with the progress info
			#
			# The queue is thread-safe
			self.queue.put((cmd, msg))

		def _start_remote_worker(self, purl):
			"""
			Start a worker for this task on a peer purl.

			Returns an instance of Worker
			"""
			logger.debug("Conneecting to remote peer %s" % (purl,))
			assert purl in self.free_peers

			if purl == self.purl:
				peer = self.pserver
			else:
				peer = xmlrpclib.ServerProxy(purl)

			# Launch the worker
			logger.debug("Launch worker for task %s" % (self.id,))
			wurl = peer.start_worker(self.id, self.spec.serialize())
			worker = self.WorkerProxy(self, wurl, purl)

			# Store the worker into our list of workers
			with self.lock:
				assert wurl not in self.workers
				self.workers[wurl] = worker
				self.free_peers.remove(purl)
				heappush(self.worker_heap, (0, worker))

			# Initialize the task on the Worker
			#logger.debug("Calling worker.initialize on %s" % (wurl,))
			worker.initialize(self.url, self.data)

			self._progress("WORKER_START", (purl, wurl))

			return worker

		def stat(self):
			# Return basic statistics about this server
			with self.lock:
				return {
					'task_id': str(self.id),
					't_started': self.t_started,
					't_now': datetime.datetime.now(),
					'n_workers': len(self.workers)
				}

		def stage_thread_ended(self, wurl, stage):
			# Called by a Worker when one of its threads
			# executing the stage finishes.
			with self.lock:
				worker = self.workers[wurl]
				worker.running_stage_threads[stage] -= 1

				self._progress("THREAD_ENDED_ON_WORKER", (wurl, stage, worker.running_stage_threads[stage]))

		def stage_ended(self, wurl, stage):
			# Called by a Worker when all of its threads
			# executing the stage have finished, and when it has
			# gotten the acknowledgments from upstream Workers that
			# they've received all the stage+1 data it sent them.
			
			# We launch the real work in a separate thread, and immediately
			# return to the worker
			Thread(target=self._stage_ended_thread, args=(wurl, stage)).start()
			
		def _stage_ended_thread(self, wurl, stage):
			# Called by a Worker when all of its threads
			# executing the stage have finished, and when it has
			# gotten the acknowledgments from upstream Workers that
			# they've received all the stage+1 data it sent them.
			task_ended = False
			with self.lock:
				worker = self.workers[wurl]
				assert worker.running_stage_threads[stage] == 0, worker.running_stage_threads
				# Decrease this worker's load and rebuild worker_heap
				self.worker_heap = [ (load - w.nkeys[stage] if w is worker else load, w) for load, w in self.worker_heap ]
				heapify(self.worker_heap)
				#
				del worker.running_stage_threads[stage]
				del worker.nkeys[stage]

				logger.debug("Worker %s notified us that stage %s has ended" % (wurl, stage))
				self._progress("STAGE_ENDED_ON_WORKER", (wurl, stage))

				# Check if this stage is done on all workers
				logger.debug("%d Workers active" % len(self.workers))
				for worker in self.workers.itervalues():
					if stage in worker.running_stage_threads:
						logger.debug("Stage %d active on worker %s" % (stage, worker.url))
						break
					else:
						logger.debug("%s -> %s" % (worker.url, worker.running_stage_threads))
				else:
					# This stage was not found in any of the worker's
					# running_stage_threads. Means this stage for the
					# entire task has ended.

					# Let Workers processing stage+1 know that the
					# previous stage has finished. Also collect the stats
					# on how many keys for that stage each worker has seen
					nkeys = 0
					for worker in self.workers.itervalues():
						if stage+1 in worker.running_stage_threads:
							nkeys += worker.stage_ended(stage)
	
					# Note this stage has finished
					self.ended_stages[stage] = nkeys

					# Remove this stage from the destinations map
					try:
						del self.destinations[stage]
						del self.maxpeers[stage]
					except KeyError:
						# Note: having no self.destinations[stage] is legal; means
						# there were no results generated by the previous stage
						pass

					# Notify the client
					self._progress("STAGE_ENDED", (stage,))
					self._report_status_to_client(force=True)

					# Check if the whole task has ended
					# Note: this is not a mistake -- there is one more stage than 
					# the number of kernels - the last stage funneled the data back
					# to the user.
					task_ended = stage == self.spec.nkernels

			if task_ended:
				logger.debug("Shutting down task...")
				self.shutdown()
				logger.debug("Shut down done...")
				self._progress("DONE", None)
				logger.debug("Progress message sent...")

		def notify_client_of_result(self, rurl):
			# Called by the last kernel, to notify the client
			# where to pick up the data
			self._progress("RESULT", rurl)

		def shutdown(self):
			# Called to shut down everything.
			with self.lock:
				workers = self.workers.values()
				assert len(self.destinations) == 0

			ths = []
			for worker in workers:
				assert len(worker.running_stage_threads) == 0
				logger.debug("Shutting down worker %s for task %s" % (worker.url, self.url))
				th = Thread(target=lambda worker=worker: worker.shutdown())
				th.start()
				ths.append(th)

			for th in ths:
				th.join()

			logger.debug("Shutting down server")
			self.server.shutdown()

		def _refresh_peers(self, force=False):
			"""
			Refresh the lists of all and unused peers
			"""
			if not force and time.time() < self.free_peers_last_refresh + 60:
				return

			self.all_peers = set(self.pserver.list_peers())
			self.free_peers = list(self.all_peers - set(w.purl for w in self.workers.itervalues()))
			self.free_peers_last_refresh = time.time()

			logger.debug("Refreshed the list of peers (%d all, %d unused)" % (len(self.all_peers), len(self.free_peers)))

		def _maxpeers(self, stage):
			"""
			Return the maximum number of Peers that will execute a stage
			
			This is taken to be equal to the number of peers that are
			running at the time of first call for a given stage. If stage
			is equal to spec.nkernels, 1 is returned (as all results
			have to be funneled to a single Worker, for the delivery to
			the user)
			"""
			with self.lock:
				if stage not in self.maxpeers:
					if stage == self.spec.nkernels:
						return 1
					else:
						self._refresh_peers()
						self.maxpeers[stage] = len(self.all_peers)

				return self.maxpeers[stage]			

		def start(self):
			# Called by the Peer that launched us to start the 
			# first Worker and stage.
			self._progress("START", None)

			# Pre-launch all workers
			self._refresh_peers(force=True)
			ths = []
			for purl in self.all_peers:
				th = Thread(target=self._start_remote_worker, args=(purl,))
				th.start()
				ths.append(th)
			for th in ths:
				th.join()

			# Start the first stage on one of the workers
			with self.lock:
				nkeys, worker = heappop(self.worker_heap)
				worker.run_stage(-1, self._maxpeers(0))
				worker.nkeys[-1] += 1
				heappush(self.worker_heap, (nkeys+1, worker))

		def get_destinations(self, stage, key):
			"""
			Get all known key->wurl pairs for the stage 'stage', ensuring
			that 'key' is among them (by creating a new worker, if needed)
			"""
			logger.info("Get destination for stage=%s key=%s" % (stage, key))
			with self.lock:
				if key not in self.destinations[stage]:
					# Refresh the list of unused peers every 60 sec or so...
					self._refresh_peers()

					logging.debug("Here: free_peers=%s", self.free_peers)
					if len(self.free_peers):
						# Prefer a Peer we're not yet running on
						purl = random.choice(self.free_peers)
						worker = self._start_remote_worker(purl)
						nkeys = 0
					else:
						# Choose an existing worker with least amount
						# of keys
						#worker = random.choice(self.workers.values())
						nkeys, worker = heappop(self.worker_heap)

					# Start the stage if not already running
					if stage not in worker.running_stage_threads:
						worker.run_stage(stage, self._maxpeers(stage))

					# Remember the destination for this key
					self.destinations[stage][key] = worker

					worker.nkeys[stage] += 1
					nkeys += 1
					heappush(self.worker_heap, (nkeys, worker))

					logger.info("Returning %s (load: nkeys=%s)" % (worker.url, nkeys))

				return [ (key, worker.url) for key, worker in self.destinations[stage].iteritems() ]

	## Peer #####################
	def __init__(self, server, port):
		self.server   = server
		self.port     = port
		self.hostname = socket.gethostname()
		self.url      = "http://%s:%s" % (self.hostname, self.port)
		self.peer_id  = np.uint64(hash(self.hostname + str(time.time())) & 0xFFFFFFFF)

		# Register our availability
		self.directory       = os.getenv("PYMR", 'peers')
		self.directory_entry = self.directory + '/' + self.hostname + ':' + str(port) + '.peer'

		# Initialize coordinated tasks array
		self.coordinators = {}
		self.coordinator_ctr = 0

		# The dictionary of workers (indexed by task_id)
		self.workers = {}

		# Global Peer Lock
		self.lock = RLock("Peer")

		# Time when we launched
		self.t_started = datetime.datetime.now()

	def _html_(self):
		with self.lock:
			"""
			Return a HTML info page about this Peer
			"""
			uptime = datetime.datetime.now() - self.t_started
	
			info = """
			<h1>Peer {hostname}:{port}</h1>
			
			<h2>Info</h2>
			<table border=1>
				<tr><th>XMLRPC URL</th><td>{url}</td></tr>
				<tr><th>Uptime</th><td>{uptime}</td></tr>
				<tr><th>Hostname</th><td>{hostname}</td></tr>
				<tr><th>Port</th><td>{port}</td></tr>
				<tr><th>Peer ID</th><td>{peer_id}</td></tr>
				<tr><th>Peer Directory Entry</th><td>{directory_entry}</td></tr>
			</table>
			""".format(uptime=uptime, **self.__dict__)

			crows = [ 
				'<tr><th><a href="{curl}">{curl}</a></th>  <td>{task_id}</td> <td>{t_started}</td> <td>{t_now}</td> <td>{n_workers}</td>  </tr>'
				.format(curl=coordinator.url, **coordinator.stat()) for coordinator in self.coordinators.itervalues()
				]
			info += """
			<h1>Coordinated tasks</h1>
			<table border=1>
				<tr><th>Coordinator</th> <th>Task ID</th> <th>Started On</th> <th>Local Time</th> <th>Workers</th></tr>
				{crows}
			</table>
			""".format(crows='\n'.join(crows))
	
		# Add info about the peers
		peers = self.list_peers()
		prows = [ 
			'<tr><th><a href="{purl}">{purl}</a></th>  <td>{peer_id}</td><td>{t_started}</td><td>{t_now}</td><td>{n_coordinators}</td><td>{n_workers}</td>  </tr>'
			.format(purl=purl, **xmlrpclib.ServerProxy(purl).stat()) for purl in peers
			]
		info += """
		<h2>Peers</h2>
		Active peers: {npeers}
		<table border=1>
			<tr><th>Peer</th> <th>Peer ID</th> <th>Started On</th> <th>Local Time</th> <th>Coordinators</th> <th>Workers</th></tr>
			{prows}
		</table>
		""".format(npeers=len(prows), prows='\n'.join(prows))

		return info

	def __del__(self):
		self._unregister()

	def _register(self):
		file(self.directory_entry, 'w').write(self.url + '\n')
		logger.debug("Registered in Directory as %s for %s" % (self.directory_entry, self.url))

	def _unregister(self):
		try:
			os.unlink(self.directory_entry)
		except OSError:
			pass
		logger.debug("Unregistered %s" % (self.directory_entry))

	def _execute(self, spec, data):
		"""
		Execute a task.
		"""
		with self.lock:
			# Create and launch a new _Coordinator XMLRPC server
			task_id = "%s.%s" % (self.peer_id, self.coordinator_ctr)
			self.coordinator_ctr += 1

			server, _ = start_threaded_xmlrpc_server(HTMLAndXMLRPCRequestHandler, 1023, self.hostname)
			coordinator = self.coordinators[task_id] = self._Coordinator(server, self.hostname, self.url, task_id, spec, data)
			server.register_instance(coordinator)
			server.register_introspection_functions()
			th = Thread(name='Coord-%03d' % (self.coordinator_ctr-1,), target=server.serve_forever, kwargs={'poll_interval': 0.1})
			th.daemon = True
			th.start()

		# Start the task in the spawned coordinator thread
		xmlrpclib.ServerProxy(coordinator.url).start()

		# Begin listening for notifications of events, yield them back to the user
		for msg in iter(coordinator.queue.get, ("DONE", None)):
			logger.debug("Progress msg to client: %s" % (msg,))
			yield msg

		yield "DONE", None

		# delete this Coordinator task
		with self.lock:
			del self.coordinators[task_id]

		th.join()
		logger.info("Done running task %s" % (task_id,))

		# Garbage collect to free up resources
		del th, server, coordinator
		logger.info("Garbage collecting __del__")
		gc.collect()

	def _cleanup(self):
		with self.lock:
			for worker in self.workers.itervalues():
				if worker.process is not None:
					logger.info("Terminating process %s" % worker.process.pid)
					worker.process.terminate()
					worker.process.communicate()

	############################
	# Peer: public XMLRPC API
	def stat(self):
		# Return basic statistics about this server
		with self.lock:
			return {
				'peer_id': str(self.peer_id),
				't_started': self.t_started,
				't_now': datetime.datetime.now(),
				'n_coordinators': len(self.coordinators),
				'n_workers': len(self.workers)
			}

	def list_peers(self):
		"""
		Return the set of all active Peers
		"""
		# Read the first line of each *.peer file in the Directory
		return list( file(fn).readline().strip() for fn in glob.iglob(self.directory + '/*.peer') )

	def start_worker(self, task_id, spec):
		"""
		Start a Worker for the given task.

		This only spawns the Worker executable, and starts up its
		XMLRPC server. The initialization of the MapReduce job is a
		separate step, invoked directly on the worker by the
		Coordinator.
		"""
		with self.lock:
			assert task_id not in self.workers

			spec = TaskSpec.unserialize(spec)

			worker_stub = os.path.abspath(sys.argv[0])

			# Ensure the worker's PYTHONPATH includes the location of pymr modules,
			# otherwise we won't be able to launch
			#
			pymr_mod_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
			if 'PYTHONPATH' in spec.env:
				spec.env['PYTHONPATH'] = spec.env['PYTHONPATH'] + ':' + pymr_mod_path
			else:
				spec.env['PYTHONPATH'] = pymr_mod_path

			# spawn the Worker -- we do this with subprocess, instead of fork()
			# to give the process a clean slate, free of any (namespace) clutter
			# that may have accumulated in this Peer
			worker_process = sp.Popen(
					[
					worker_stub,
					'--worker=%s' % self.hostname,
					spec.fn
					] + spec.argv,
				stdin=sp.PIPE, stdout=sp.PIPE, stderr=None,
				cwd=spec.cwd,
				env=spec.env)

			# Get the Worker's address and connect to its XMLRPC server
			wurl = worker_process.stdout.readline().strip()
			logger.debug("Got worker URL: %s" % (wurl,))

			# Establish connection, record the worker (keyed by task_id)
			#worker = self._Coordinator.WorkerProxy(wurl, self.url, worker_process)
			worker = xmlrpclib.ServerProxy(wurl)
			self.workers[task_id] = worker

			# Launch a thread to monitor when the process exits
			th = Thread(name="PMon-%s" % (worker_process.pid,), target=self._monitor_worker_process, args=(task_id, worker_process, wurl,))
			th.start()

			return wurl

	def _monitor_worker_process(self, task_id, process, wurl):
		# Called as a separate thread to monitor worker process progress
		# and remove it from self.workers map once it terminates
		retcode = process.wait()
		with self.lock:
			logger.info("Worker %s (pid=%s) exited with retcode=%s" % (wurl, process.pid, retcode))
			del self.workers[task_id]

	def shutdown(self):
		return self.server.shutdown()

	############################
	# Mock functions

	def pow(self, x, y):
		return pow(x, y)

	def add(self, x, y) :
		return x + y

	def div(self, x, y):
		return float(x) / float(y)

	def mul(self, x, y):
		return x * y

	def mad(self, x, y, z):
		s = xmlrpclib.ServerProxy('http://%s:%s' % (self.hostname, self.port))
		a = s.mul(x,y)
		return s.add(a, z)

class HTMLAndXMLRPCRequestHandler(SimpleXMLRPCServer.SimpleXMLRPCRequestHandler):
	def do_GET(self):
		fun = '_html_' + self.path[1:]

		f = getattr(self.server.instance, fun, None) 
		if f is None:
			self._err_404("Function %s not found" % (fun,))
		else:
			self.send_response(200)
			self.send_header("Content-type", "text/html")
			self.end_headers()

			self.wfile.write(f())

	def _err_404(self, response):
		# Report a 404 error
		self.send_response(404)
		self.send_header("Content-type", "text/plain")
		self.send_header("Content-length", str(len(response)))
		self.end_headers()
		self.wfile.write(response)

class PeerRequestHandler(HTMLAndXMLRPCRequestHandler):
	# Override do_POST() to permit stateful connections
	# for task submission and progress reporting
	def do_POST(self):
		if self.path != '/execute':
			return HTMLAndXMLRPCRequestHandler.do_POST(self)

		logger.debug("New client connection")
		req = self._parse_request()

		#if True:
		#	self.send_response(200)
		#	self.send_header("Content-type", "text/html")
		#	self.end_headers()
		#
		#	self.wfile.write("Here<br>")
		#	self.wfile.write(str(req))
		#	return

		for arg in ['spec', 'data']:
			if arg not in req:
				self._err_404("Argument '%s' missing." % arg)
				return
			if len(req[arg]) != 1:
				self._err_404("Incorrect value format for '%s'." % arg)
				return
			req[arg] = req[arg][0]

		self.send_response(200)
		self.send_header("Content-type", "binary/octet-stream")
		self.end_headers()

		# Forward progress reports
		for msg in self.server.instance._execute(req['spec'], req['data']):
			cPickle.dump(msg, self.wfile, -1)
			self.wfile.flush()

	def _parse_request(self):
		ctype, pdict = cgi.parse_header(self.headers['content-type'])
		if ctype == 'multipart/form-data':
			return cgi.parse_multipart(self.rfile, pdict)
		elif ctype == 'application/x-www-form-urlencoded':
			# Get arguments by reading body of request.
			# We read this in chunks to avoid straining
			# socket.read(); around the 10 or 15Mb mark, some platforms
			# begin to have problems (bug #792570).
			max_chunk_size = 10*1024*1024
			size_remaining = int(self.headers["content-length"])
			L = []
			while size_remaining:
				chunk_size = min(size_remaining, max_chunk_size)
				L.append(self.rfile.read(chunk_size))
				size_remaining -= len(L[-1])
			qs = ''.join(L)

			return urlparse.parse_qs(qs, keep_blank_values=1)
		else:
			return None

def start_server(ServerClass, HandlerClass, port=1023, addr='', **kwargs):
	# Find the next available port
	while port < 2**16:
		try:
			server = ServerClass((addr, port), HandlerClass, **kwargs)
			return server, port
		except socket.error:
			port += 1

def start_threaded_xmlrpc_server(HandlerClass, port=1023, addr='', **kwargs):
	return start_server(ThreadedXMLRPCServer, HandlerClass, allow_none=True, logRequests=False, **kwargs)
