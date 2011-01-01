#!/usr/bin/env python

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
from base64 import b64encode, b64decode
from collections import defaultdict

import core

# Buffer size -- can't be too big on 32bit platforms (otherwise all mmaps
# won't fit in memory)
BUFSIZE = 100 * 2**20 if platform.architecture()[0] == '32bit' else 2**40

logger = logging.getLogger('mrp2p')

if False:
	# Verbose wrappers for debugging
	class Event(threading._Event):
		def __init__(self, name, verbose=None):
			self.__name = name
			threading._Event.__init__(self, verbose)
	
		def __str__(self):
			return "Event %s" % self.__name
	
		def wait(self, timeout=None):
			ct = threading.current_thread()
	
			logger.debug("[%s] Waiting on %s" % (self, ct))
			while not threading._Event.wait(self, 3):
				logger.debug("[%s] EVENT EXPIRED ON %s" % (self, ct))
	
	class RLock(threading._RLock):
		def acquire(self):
			new = not self._is_owned()
			ct = threading.current_thread()
	
			if new:
				logger.debug("[%s] Acquiring for %s" % (self, ct))
	
			for retry in xrange(10):
				if threading._RLock.acquire(self, blocking=False):
					break
				time.sleep(1)
				logger.debug("[%s] Acquiring for %s (attempt %d)." % (self, ct, retry))
			else:
				logger.debug("[%s] DEADLOCKED IN THREAD %s." % (self, ct))
				raise Exception()
			
			if new:
				logger.debug("[%s] Acquired in %s." % (self, ct))
	
		__enter__ = acquire
	
		def release(self):
			ret = threading._RLock.release(self)
			ct = threading.current_thread()
			if not self._is_owned():
				logger.debug("[%s] Releasing on %s" % (self, ct))
			return ret
else:
	RLock = threading.RLock
	Event = threading.Event

def unpack_callable(func):
	""" Unpack a (function, function_args) tuple
	"""
	func, func_args = (func, ()) if callable(func) or func is None else (func[0], func[1:])
	return func, func_args

def mrp2p_init():
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
	nkernel = None		# Number of kernels
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

	def map_reduce_chain(self, items, kernels, locals=[], progress_callback=None):
		# Prepare request
		spec = TaskSpec(fn, argv, cwd, env, len(items), len(kernels), len(locals))
		req = {
			'spec': spec.serialize(),
			'data': b64encode(cPickle.dumps([kernels, locals], -1) + cPickle.dumps(items, -1)),
		      }
		req = urllib.urlencode(req)
		#print req;
		#s = urlparse.parse_qs(req)['spec'][0]
		#u = TaskSpec.unserialize(s)
		#print u
		#exit()

		# Choose a random peer
		peers = glob.glob(self.directory + '/*.peer')
		if not len(peers):
			raise ConnectionError('No active peers found in %s' % self.directory)
		purl = file(random.choice(peers)).readline().strip()
		url = purl + "/execute"

		# Submit the task
		fp = urllib.urlopen(url, req)

		# Listen for progress messages: a stream of pickled
		# (msg, args) tuples
		while True:
			try:
				msg, args = cPickle.load(fp)
				print "PROGRESS:", msg, args
			except EOFError:
				fp.close()
				break

			if msg == "RESULT":
				# Results are a stream of pickled Python objects that we unpickle and
				# pass back to the client
				rurl = args
				rfp = urllib.urlopen(rurl)
				try:
					while True:
						yield cPickle.load(rfp)
				except EOFError:
					rfp.close()

def _make_buffer_mmap(size, dir=None):
	# Create the temporary memory mapped buffer. It will go away as soon as the mmap is closed, or the process exits.
	fp = tempfile.TemporaryFile(dir=dir)
	os.ftruncate(fp.fileno(), size)		# Resize to self.bufsize
	mm = mmap.mmap(fp.fileno(), 0)		# create a memory map
	fp.close()				# Close immediately to trigger the unlink()-ing of the unrelying file.
	return mm

class KeyChainSentinelClass(object):
	pass

KeyChainSentinel = KeyChainSentinelClass()

class AckDoneClass(object):
	def __eq__(self, a):
		return isinstance(a, AckDoneClass)

AckDoneSentinel = AckDoneClass()

class Worker(object):
	"""
	The class encapsulating a Worker process, and acting
	as an XMLRPC method provider.
	"""
	class Scatterer(asyncore.file_dispatcher):
		"""
		Queue the items for sending into separate buffers, one
		per each _destination_.
		
		To add a new destination (from a different thread), write
		to Scatterer's fd to trigger entering the Scatterer's thread.
		"""
		class ScattererChannel(asyncore.dispatcher):
			bufsize = BUFSIZE 	# buffer size - 1TB (ext3 max. file size is 2TB)
			ctresh  = 50 * 2**20	# compactification treshold (50M)

			addr	= None	# (host, port) destination

			lock    = None	# The lock guarding mm, mm_* vars, and serving_stages
			mm	= None	# Buffer memory map

			mm_at	= 0	# read pointer in the buffer (data before this have been scattered)
			mm_end	= 0	# end read pointer (less than or equal to mm.tellg()). Used to avoid locking on every writable() call

			serving_stages = None	# The set of stages for which we've sent out keys.
			asyncore_map = None		# asyncore thread map

			def __hash__(self):
				return id(self)

			def __eq__(self, b):
				return id(self) == id(b)

			def finish_stage(self, stage):
				self.serving_stages.remove(stage)

			def __init__(self, host, port, map):
				# Constructor: this gets called from the Worker's thread,
				# but we don't initialize the socket or connect to the destination
				# here, but do it from the asyncore thread (see delayed_init())
				self.addr = (host, port)
				self.asyncore_map = map

				self.mm = _make_buffer_mmap(self.bufsize)

				self.lock = RLock()
				
				self.serving_stages = set()

			def delayed_init(self, map):
				# The __init__ equivalent called from asyncore's thread
				# Connects to the destination host:port
				asyncore.dispatcher.__init__(self, map=self.asyncore_map)

				self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
				self.connect(self.addr)

			def writable(self):
				# We're writable only if the buffer isn't empty
				if self.mm_at == self.mm_end:
					# Check if extra stuff has been added to
					# the buffer
					with self.lock:
						self.mm_end = self.mm.tell()

				return self.mm_at < self.mm_end

			def handle_write(self):
				# Write as much as we can to the destination.
				sent = self.send(self.mm[self.mm_at:self.mm_end])
				self.mm_at += sent

				# Compactify every time we send >= 50MB
				if self.mm_at > self.ctresh:
					with self.lock:
						count = self.mm.tell() - self.mm_at
						self.mm.move(0, self.mm_at, count)
						self.mm.seek(count)

						self.mm_at  = 0
						self.mm_end = count

						# Resize to flush the dirty pages
						self.resize(count)
						self.resize(self.bufsize)

			def queue(self, stage, k, v):
				# Runs from within a worker thread
				with self.lock:
					# Serialize the key/value
					pkt_beg_offs = self.mm.tell()
					self.mm.seek(8, 1)				# 1. [ int64] Payload length (placeholder, filled in 5.)

					payload_beg_offs = self.mm.tell()
					self.mm.write(struct.pack('<I', stage))		# 2. [uint32] Stage

					cPickle.dump(k, self.mm, -1)			# 3. [pickle] Key
					cPickle.dump(v, self.mm, -1)			# 4. [pickle] Value

					self.mm[pkt_beg_offs:pkt_beg_offs+8] = \
						struct.pack('<Q', self.mm.tell() - payload_beg_offs)	# 5. Fill in the length

					self.serving_stages.add(stage)			# Remember we've sent out a key for this stage

		parent = None		# Parent Worker instance
		asyncore_map = None	# asyncore map that this Scatterer participates in
		coordinator = None	# XMLRPC proxy to the Coordinator

		destinations = None	# Map of (host, port) -> ScatterChannel
		key_destinations = None	# Map of (stage, key) -> ScatterChannel
		stage_destinations=None # Map of (stage) -> set(ScatterChannel)

		pending_channels = None # A list of channels that need to be added to the asyncore map
		pending_done = None		# A list of stages that have finished (and for which we can release Channels)
		
		waiting_ack = None	# dict: stage->number of workers we're waiting to ack they've recvd all of our data

		r_fd = None		# Pipe used to wake up the Scatterer in the asyncore thread (read endpoint)
		w_fd = None		# Pipe used to wake up the Scatterer in the asyncore thread (write endpoint)

		lock = None		# A lock protecting all member variables

		def __init__(self, parent, curl, map):
			self.parent = parent
			self.asyncore_map = map
			self.pending_channels = []
			self.pending_done = []
			self.waiting_ack = {}
			self.coordinator = xmlrpclib.ServerProxy(curl)

			self.lock = RLock()

			self.destinations = {}			# (host,port) -> ScatterChannel map
			self.key_destinations = {}		# (stage,key) -> ScatterChannel map
			self.stage_destinations = defaultdict(set)	# (stage) -> set(ScatterChannel) map

			self.r_fd, self.w_fd = os.pipe()	# Set up this object to listed on the pipe from within the asyncore thread
			asyncore.file_dispatcher.__init__(self, self.r_fd, map)

		def __del__(self):
			# Close the communication pipe
			os.close(self.r_fd)
			os.close(self.w_fd)

		def close_channels(self, scs):
			# Must be called from asyncore thread
			#
			# Close the channel and remove it from all of our maps
			with self.lock:
				keys = [ k for k, v in self.destinations.iteritems() if v in scs ]
				for k in keys: del self.destinations[k]
				
				keys = [ k for k, v in self.key_destinations.iteritems() if v in scs ]
				for k in keys: del self.key_destinations[k]

				# Remove the scs from stage_destinations, and remove any
				# stage_destinations that are left empty because of it
				stages = [ k for k, v in self.stage_destinations.iteritems() if len(v & scs) ]
				for stage in stages:
					s = self.stage_destinations[stage]
					s -= scs
					if len(s) == 0:
						del self.stage_destinations[stage]

				# Close the Channels (this will close their sockets, remove them from
				# asyncore map)
				for sc in scs:
					sc.close()

		def handle_read(self):
			# Called from within the asyncore thread when something is
			# written to w_fd
			with self.lock:
				# add any pending channels to the asyncore loop
				for sc in self.pending_channels:
					sc.delayed_init(map=self.asyncore_map)
				del self.pending_channels[:]

				# Find and close any Channels in stages that have finished
				for stage in self.pending_done:
					to_close = set()
					for sc in self.stage_destinations[stage]:
						nleft = sc.finish_stage(stage)
						if nleft == 0:
							to_close.append(sc)
					self.close_channels(to_close)

				# Clear the signal
				nread = 8192
				while self.recv(nread) == nread:
					pass

		def notify(self):
			# Called from outside the asyncore thread
			#
			# Signal to wake up the asyncore thread (to leave
			# the select() call and invoke self.handle_read(),
			# as well as re-check the .writable() property of
			# all ScatterChannel instances)
			#
			# Note: it has to be locked to protect against a
			#       potential race condition in handle_read().
			with self.lock:
				os.write(self.w_fd, '1')

		def ack_done_producing(self, stage):
			# Called from worker thread when an ack for an
			# AckDone event is received (see end_stage)
			with self.lock:
				self.waiting_ack[stage] -= 1
				if self.waiting_ack[stage] == 0:
					# Queue up for cleanup
					self.pending_done.append(stage)
					self.notify()

					# Notify the Coordinator that this Worker is
					# completely done with stage=stage-1
					self.coordinator.stage_ended(self.parent.url, stage-1)

		def done_producing(self, stage):
			# Called from worker thread
			#
			# Send a control message to all connected Gatherers
			# to acknowledge they've received all the data sent
			# to them so far.
			#
			# The receiveing Gatherers will respond to the message
			# by invoking the Worker's gatherer_ack() XMLRPC routine.
			#
			# Each invocation of that routine reduces the "active
			# destination counter" by one. When the counter reaches zero,
			# the Worker signals the Coordinator that it has finished
			# processing the stage 'stage'.
			with self.lock:
				self.waiting_ack[stage] = len(self.stage_destinations[stage])

				for sc in self.stage_destinations[stage]:
					sc.queue(stage, AckDoneSentinel, (self.parent.url, stage))

				# Wake up the asyncore thread
				self.notify()

		def queue(self, stage, kv):
			# This gets called from the Worker's thread
			#
			# 1. Find the destination for the key
			# 2. Open a channel to that destination, if it doesn't yet exist (and write on self.w_fd)
			# 3. Queue up the value on that channel
			#
			# TODO: Rewrite this to avoid calling the Coordinator for every key
			#
			key, v = kv
			with self.lock:
				if (stage, key) in self.key_destinations:
					# Destination known
					sc = self.key_destinations[(stage, key)]
				else:
					# Unknown (stage, key). Find where to send them
					logger.debug("Getting destination from coordinator")
					wurl = self.coordinator.get_destination_for(stage, key)
					(host, port) = xmlrpclib.ServerProxy(wurl).get_gatherer_addr()

					if (host, port) not in self.destinations:
						# Open a new channel
						sc = self.ScattererChannel(host, port, self.asyncore_map)
						self.pending_channels.append(sc)
						self.destinations[(host, port)] = sc
					else:
						sc = self.destinations[(host, port)]

					# Remember for later
					self.key_destinations[(stage, key)] = sc
					
					# Store into a list of destinations for this stage
					self.stage_destinations[stage].add(sc)

				# Queue the data to the right channel
				sc.queue(stage, key, v)

				# Wake up the asyncore thread
				self.notify()

	class Gatherer(asyncore.dispatcher_with_send):
		"""
		Collect incoming (key, value) tuplets from other Workers
		and funnel them into a single stream to be fed to our
		Worker.

		Does this on a per-stage basis (one buffer per stage)
		"""
		class GathererChannel(asyncore.dispatcher_with_send):
			"""
			A connection to a Scatterer on another Worker.
			
			Receives the next (stage, key, value) tuple from the
			other Worker, and once received stores it into the
			apropriate buffer.
			"""
			parent = None	# Parent Gatherer
			buf = None	# cStringIO instance where we buffer the incoming packet, until it's complete

			def __init__(self, parent, sock, map):
				self.parent = parent
				self.buf = cStringIO.StringIO()

				asyncore.dispatcher_with_send.__init__(self, sock, map=map)

			def handle_read(self):
				# Receive as much as possible into self.buf, and attempt to
				# process it if the message is complete.
				data = self.recv(1*1024*1024)
				self.buf.write(data)
				self.process_buffer()

				if data == '':
					# The Scatterer has closed the connection.

					# TODO: Handle broken connections
					assert self.buf.getvalue() == '', "Connection broke?"

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
						(wurl, stage) = cPickle.loads(pkl_value)
						xmlrpclib.ServerProxy(wurl).ack_done_producing(stage)
					else:
						self.parent.append(stage, key, pkl_value)

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
			new_value_evts = None	# dict: key -> Event() which becomes set once new
						# values are available for the given key
			all_recvd = False	# True if the stage _before_ the one this Buffer is buffering
						# has ended (i.e., no more key/values are to be received).

			def __init__(self, size):
				self.lock = RLock()
				self.new_key_evt = Event("new_key_evt")

				self.chains = {}
				self.new_value_evts = {}

				self.mm = _make_buffer_mmap(size)

				# Ensure the first in a buffer chain is always the chain of keys
				self.append(KeyChainSentinel, cPickle.dumps((KeyChainSentinel, 0), -1))
				self.next_key = self.chains[KeyChainSentinel][1]

			def append(self, key, pkl_value):
				# Appent the pickled value to the chain of values
				# for key 'key'

				with self.lock:
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
					logger.debug("key=%s, offset=%s" % (key, mm.tell()))
					mm.write(struct.pack('<Q', len(pkl_value)))	# 1) Length of the data (uint64)
					mm.write(pkl_value)				# 2) The data (pickled)
					mm.write('\xff'*8)				# 3) Offset to next item in chain (or 0xFFFFFFFFFFFFFFFF, if end-of-chain)

					# Update the 'last' field in the self.chains dict
					self.chains[key] = self.chains[key][0], mm.tell() - 8

					# If this is a new chain, link to it from the primary chain
					# .. unless this is the initialization of the primary chain
					if new_key and key is not KeyChainSentinel:
						self.append(KeyChainSentinel, cPickle.dumps((key, self.chains[key][0]), -1))

						# Notify listener (Buffer.itervalues instances) that new keys are available
						self.new_key_evt.set()

					if key in self.new_value_evts:
						# Notify the active iterator that a new value was added
						self.new_value_evts[key].set()

			def iter_chain(self, key, first, last):
				"""
				A generator traversing a key chain from [first, last)

				first is the offset in self.mm where the [len] field of the
				first packet to be read is. last is the offset to the [offset-to-next]
				field of the last packet to be read.
				"""
				at = first
				while True:
					len, = struct.unpack('<Q', self.mm[at:at+8])

					with self.lock:
						at0 = self.mm.tell()
						try:
							self.mm.seek(at+8)
							v = cPickle.load(self.mm)
							at = self.mm.tell()
						finally:
							self.mm.seek(at0)

					yield v

					if at == last:
						# Reached the end of chain
						break

					with self.lock:
						# Load the position of the next packet
						offs, = struct.unpack('<Q', self.mm[at:at+8])
						at += offs + 8

			def itervalues(self, key, kevt, at):
				"""
				A blocking generator yielding values associated with 'key' until
				self.all_recvd is set.

				The caller guarantees that initially there's at least one key available,
				at offset 'at'. When no more values are available, the generator
				blocks until new values become available (and is signaled via kevt).
				"""
				while True:
					with self.lock:
						_, last = self.chains[key]

					# yield the available values
					for v in self.iter_chain(key, at, last):
						yield v

					while True:
						with self.lock:
							# See if more data showed up in the meantime.
							offs, = struct.unpack('<Q', self.mm[last:last+8])
							all_recvd = self.all_recvd
							kevt.clear()

						if offs == 0xFFFFFFFFFFFFFFFF:
							# No data currently available
							if all_recvd:
								return
							kevt.wait()
						else:
							# More data available
							at = last + 8 + offs
							break

			def iteritems(self):
				# Find keys not yet being processed, and yield
				# generators that will yield values for those keys

				# FIXME: TODO: take all_recvd into account
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
						self.new_key_evt.clear()

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

							# Mark this key as active, and create an Event for it
							# that will get triggered every time new data becomes
							# available for the key.
							kevt = self.new_value_evts[key] = Event("new_value_evt[%s]" % key)
							kevt.set()

							# Create the generator of values from this key chain
							value_gen = self.itervalues(key, kevt, first)
						else:
							# Check if this stage was marked as done, exit if so
							if self.all_recvd:
								return

					# Note: we're doing this outside the locked section
					if value_gen is not None:
						try:
							yield key, value_gen
						finally:
							# Remove the event
							with self.lock:
								del self.new_value_evts[key]
					else:
						# No keys to work on. Sleep waiting for a new one to be added.
						self.new_key_evt.wait()

			def all_received(self):
				# Expect to receive no more data in this buffer (because
				# the stage feeding it has ended)
				with self.lock:
					assert not self.all_recvd
					self.all_recvd = True

					# Wake up all threads waiting for new data to let
					# them know this is it
					self.new_key_evt.set()
					for evt in self.new_value_evts.itervalues():
						evt.set()

		parent = None		# Worker instance
		asyncore_map = None	# asyncore map that Gatherer participates in

		buffers = {}		# A dictionary of Buffer instances, keyed by stage
		port = None		# The port we're listening on for incoming Scatterer connections
		bufsize = BUFSIZE	# Buffer size (per stage)

		lock = None		# RLock guarding the buffers variable

		def __init__(self, parent, curl, map):
			self.parent = parent
			self.asyncore_map = map

			self.lock = RLock()
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

		def get_or_create_buffer(self, stage):
			"""
			Return or create a Buffer instance for stage
			"""
			with self.lock:
				# Find/create the mmap for the requested stage
				if stage in self.buffers:
					buf = self.buffers[stage]
				else:
					logger.debug("Creating buffer for stage=%d" % stage)
					buf = self.buffers[stage] = self.Buffer(self.bufsize)

			return buf

		def iteritems(self, stage):
			return self.get_or_create_buffer(stage).iteritems()

		def append(self, stage, key, pkl_value):
			"""
			Store a (key, pkl_value) pair into the buffer for
			the selected stage.
			"""
			return self.get_or_create_buffer(stage).append(key, pkl_value)

		def stage_ended(self, stage):
			# Notification from the coordinator that a particular
			# stage has ended. That means that all buffers one stage
			# later should expect no more data.
			self.buffers[stage+1].all_received()

		def handle_accept(self):
			# Accept a new connection to this Gatherer (presumably
			# by another Scatterer)
			pair = self.accept()
			if pair is not None:
				sock, _ = pair
				self.GathererChannel(self, sock, self.asyncore_map)

	server  = None		# XMLRPC server instance
	hostname = None		# The hostname of this machine
	port    = None		# The port on which we're listening
	url     = None		# The Worker's XMLRPC server URL

	kernels = None		# The list of kernels to execute
	locals  = None		# Local variables to be made available to the kernels (TODO: not implemented yet)
	
	coordinator = None	# XMLRPC Proxy to the Coordinator
	gatherer = None		# Gatherer instance
	scatterer = None	# Scatterer instance

	running_stage_threads = None	# defaultdict(int): stage -> number of worker threads processing the keys from stage

	def __init__(self, server, hostname):
		self.server = server

		self.hostname, self.port = hostname, server.server_address[1]
		self.url = 'http://%s:%s' % (self.hostname, self.port)

		self.running_stage_threads = defaultdict(int)
		self.lock      = RLock()

	def get_gatherer_addr(self):
		# Return the port on which our Gatherer listens
		return (self.hostname, self.gatherer.port)

	def initialize(self, curl, data):
		# Initialize the connection back to the Coordinator
		# WARNING: This routine MUST NOT call back the Coordinator
		#          or else it will deadlock
		self.coordinator = xmlrpclib.ServerProxy(curl)

		# Initialize Gatherer and Scatterer threads, with the asyncore
		# loop running in a separate thread
		asyncore_map = {}
		self.gatherer  = self.Gatherer(self, curl, map=asyncore_map)
		self.scatterer = self.Scatterer(self, curl, map=asyncore_map)
		th = threading.Thread(name='Scatter/Gather', target=asyncore.loop, kwargs={'timeout': 3600, 'map': asyncore_map})
		th.daemon = True
		th.start()

		# Initialize the task:
		fp = cStringIO.StringIO(b64decode(data))
		[ self.kernels, self.locals ] = cPickle.load(fp)

		# Place the (pickled) items on the gatherer's queue
		if False:
			self.gatherer.append(-1, 0, fp.read())
		else:
			items = cPickle.load(fp)
			for item in items:
				self.gatherer.append(-1, 0, cPickle.dumps(item))
		self.stage_ended(-2)
	
	def ack_done_producing(self, stage):
		# Called by a remote Gatherer to notify
		# our scatterer that it has received the AckDone
		# message
		self.scatterer.ack_done_producing(stage)

	class ResultHTTPRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
		def do_GET(self):
			# Just return the results to the user
			self.send_response(200)
			self.send_header("Content-type", "binary/octet-stream")
			self.end_headers()

			for bytes in self.server.valiter:
				self.wfile.write(bytes)

	def _serve_results(self, kv):
		"""
		Open a HTTP port and serve the results back to the client.
		"""
		_, v = kv
		httpd, port = start_server(BaseHTTPServer.HTTPServer, self.ResultHTTPRequestHandler, self.hostname, self.port)

		with self.lock:
			self.coordinator.notify_client_of_result("http://%s:%s" % (self.hostname, port))

		httpd.valiter = v	# The iterator returning the values
		httpd.handle_request()

	def _worker(self, stage):
		# Executes the kernel for a given stage in
		# a separate thread (called from run_stage)
		with self.lock:
			logger.info("Worker thread for stage %s active on %s" % (stage, self.url))
			# stage = -1 and stage = len(kernels) are special (feeder and collector kernels)
			if stage == -1:
				def K_start(kv):
					_, v = kv
					items = list(v)
					for k, item in enumerate(items):
						yield k, item
				K_fun, K_args = K_start, ()
			elif stage == len(self.kernels):
				K_fun, K_args = self.gatherer.serve_results, ()
			else:
				K_fun, K_args = unpack_callable(self.kernels[stage])

				if stage == 0:
					# stage = 0 kernel has a thin wrapper removing
					# the keys before passing the values to the mapper
					def K(kv, args, K_fun, K_args):
						_, v = kv
						return K_fun(v, *K_args)
					K_fun, K_args = K, (args, K_fun, K_args)

		# Do the actual work
		for key, valgen in self.gatherer.iteritems(stage):
			for kv in K_fun((key, valgen), *K_args):
				self.scatterer.queue(stage+1, kv)

		with self.lock:
			# Unregister ourselves from the list of active workers
			self.running_stage_threads[stage] -= 1
			
			# Let the coordinator know a stage thread has ended
			# (this is mostly for statistics/debugging purpuses for now)
			self.coordinator.stage_thread_ended(self.url, stage)
	
			if self.running_stage_threads[stage] == 0:
				# If this is the last thread left running this stage,
				# let the Scatterer know we're done producing for stage+1
				self.scatterer.done_producing(stage+1)

				# Delete the entry for this stage
				del self.running_stage_threads[stage]

	def run_stage(self, stage):
		# Start running a stage, in a separate thread
		# WARNING: This routine must not call back into the
		#          Coordinator (will deadlock)
		logger.debug("Starting stage %s on %s" % (stage, self.url))
		if stage != -1:
			return
		with self.lock:
			assert -1 <= stage <= len(self.kernels)
			self.running_stage_threads[stage] += 1
			th = threading.Thread(name='Stage %d' % stage, target=self._worker, args=(stage,))
			th.daemon = True
			th.start()
			
	def stage_ended(self, stage):
		# Notification from the coordinator that a particular
		# stage has ended
		self.gatherer.stage_ended(stage)

class Peer:
	class _Coordinator(object):
		class WorkerProxy(xmlrpclib.ServerProxy):
			url                   = None	# URL of the Worker
			purl		      = None	# URL of the Worker's Peer
			running_stage_threads = None	# defaultdict(int): the number of threads running each stage on the Worker
			process               = None	# subprocess.Popen class for local workers, None for remote workers

			def run_stage(self, stage):
				"""
				Thin layer over the run_stage RPC that records
				the run in running_stage_threads
				"""
				self.running_stage_threads[stage] += 1
				return self.__getattr__('run_stage')(stage)

			def __init__(self, url, purl, process=None):
				xmlrpclib.ServerProxy.__init__(self, url)

				self.url = url
				self.purl = purl
				self.process = process
			
				self.running_stage_threads = defaultdict(int)

		## _Coordinator ####################
		id	= None	# Unique task ID
		server  = None  # XMLRPCServer instance for this Coordinator (note: this is _not_ the Peer that launched it)
		hostname= None  # Our host name
		port    = None  # TCP port
		url     = None  # XMLRPC URL of our server
		purl	= None	# Parent peer URL

		spec    = None	# TaskSpec instance describing the task
		data    = None  # Serialized task data, for the workers

		workers = None	# dict: wurl -> WorkerProxy for workers working on the task
		queue   = None  # Queue for the Coordinator to message the launching Peer with task progress
		pserver = None  # XMLRPC Proxy to the parent Peer that launched us

		destinations = None	# destinations[stage][key] gives the WorkerProxy that receives (stage, key) data

		free_peers = None		# The set of currently unused peers
		free_peers_last_refresh = 0	# The last time when free_peers was refreshed from the Directory

		lock    = None	# Lock protecting instance variables

		def __init__(self, server, hostname, parent_url, id, spec, data):
			self.lock    = RLock()

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
			self.destinations = {}

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
			self.queue.put((cmd, msg))

		def _start_remote_worker(self, purl):
			"""
			Start a worker for this task on a peer purl.

			Returns an instance of Worker
			"""
			logger.debug("Conneecting to remote peer %s" % (purl,))
			if purl == self.purl:
				peer = self.pserver
			else:
				peer = xmlrpclib.ServerProxy(purl)

			# Launch the worker
			logger.debug("Launch worker for task %s" % (self.id,))
			wurl = peer.start_worker(self.id, self.spec.serialize())
			logger.debug("Connecting to worker %s" % (wurl,))
			worker = self.WorkerProxy(wurl, purl)
			logger.debug("Connected to %s" % (wurl,))

			# Store the worker into our list of workers
			with self.lock:
				self.workers[wurl] = worker

			# Initialize the task on the Worker
			logger.debug("Calling worker.initialize on %s" % (wurl,))
			worker.initialize(self.url, self.data)

			self._progress("WORKER_START", (purl, wurl))

			return worker

		def start(self):
			# Called by the Peer that launched us to start the 
			# first Worker and stage.
			self._progress("START", None)

			with self.lock:
				self._start_remote_worker(self.purl).run_stage(-1)

		def stage_thread_ended(self, wurl, stage):
			# Called by a Worker when one of its threads
			# executing the stage finishes.
			with self.lock:
				worker = self.workers[wurl]
				worker.running_stage_threads[stage] -= 1

				self._progress("THREAD_ENDED_ON_WORKER", (wurl, stage, worker.running_stage_threads[stage]))

				if worker.running_stage_threads[stage] == 0:
					del worker.running_stage_threads[stage]

		def stage_ended(self, wurl, stage):
			# Called by a Worker when all of its threads
			# executing the stage have finished, and when it has
			# gotten the acknowledgments from upstream Workers that
			# they've received all the stage+1 data it sent them.
			worker = self.workers[wurl]
			assert stage not in worker.running_stage_threads	# Should've been deleted in stage_thread_ended()

			self._progress("STAGE_ENDED_ON_WORKER", (wurl, stage))

			# Check if this stage is done on all workers
			for worker in self.workers.itervalues():
				if stage in worker.running_stage_threads:
					break
			else:
				# This stage was not found in any of the worker's
				# running_stage_threads. Means this stage for the
				# entire task has ended.
				self._progress("STAGE_ENDED", (wurl, stage))

				# Check if this means the whole task has ended
				# Note: this is not a mistake -- there is one more stage than 
				# the number of kernels - the last stage funnels the data back
				# to the user
				if stage == self.spec.nkernels:
					self._progress("DONE", None)
					self.shutdown()
				else:
					# Let Workers processing stage+1 know that the
					# previous stage has finished
					for worker in self.workers.itervalues():
						if stage+1 in worker.running_stage_threads:
							worker.stage_ended(stage)

					# Remove this stage from the destinations map
					try:
						del self.destinations[stage]
					except KeyError:
						# Note: having no self.destinations[stage] is legal; means
						# there were no results generated by the previous stage
						pass

		def notify_client_of_result(self, rurl):
			# Called by the last kernel, to notify the client
			# where to pick up the data
			self._progress("RESULT", rurl)

		def shutdown(self):
			# Called to shut down everything.
			with self.lock:
				assert len(self.destinations) == 0
				for worker in self.workers.itervalues():
					assert len(worker.running_stage_threads) == 0

			self.server.shutdown()

		def get_destination_for(self, stage, key):
			"""
			Get URL of a Gatherer to which to send (stage, key)-keyed items
			"""
			with self.lock:
				if stage not in self.destinations:
					self.destinations[stage] = {}
				if key in self.destinations[stage]:
					return self.destinations[stage][key].url

				# Refresh the list of unused peers every 60 sec or so...
				if self.free_peers_last_refresh + 60 < time.time():
					self.free_peers = list(set(self.pserver.list_peers()) - set(w.purl for w in self.workers.itervalues()))
					self.free_peers_last_refresh = time.time()
					logger.debug("Refreshed the list of free peers (%d free)" % len(self.free_peers))

				if len(self.free_peers):
					# Choose a Peer we're not yet running on
					purl = random.choice(self.free_peers)
					worker = self._start_remote_worker(purl)
					self.free_peers.remove(purl)
				else:
					# Randomly choose an existing worker
					worker = random.choice(self.workers.values())

				# Start the stage if not already running
				if stage not in worker.running_stage_threads:
					worker.run_stage(stage)

				# Remember the destination for this key
				self.destinations[stage][key] = worker

				# Return the URL
				return worker.url

	## Peer #####################
	def __init__(self, server, port):
		self.server   = server
		self.port     = port
		self.hostname = socket.gethostname()
		self.url      = "http://%s:%s" % (self.hostname, self.port)
		self.peer_id  = np.uint64(hash(self.hostname + str(time.time())) & 0xFFFFFFFF)

		# Register our availability
		self.directory       = 'peers'
		self.directory_entry = self.directory + '/' + self.hostname + ':' + str(port) + '.peer'

		# Initialize coordinated tasks array
		self.coordinators = {}
		self.coordinator_ctr = 0

		# The dictionary of workers (indexed by task_id)
		self.workers = {}

		# Global Peer Lock
		self.lock = RLock()

		# Time when we launched
		self.t_started = datetime.datetime.now()

	def stat(self):
		# Return basic statistics about this server
		return {
			'peer_id': str(self.peer_id),
			't_started': self.t_started,
			't_now': datetime.datetime.now(),
			'n_coordinators': len(self.coordinators),
			'n_workers': len(self.workers)
		}

	def _info(self):
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

		# Add info about the peers
		peers = self.list_peers()
		prows = [ 
			'<tr><th><a href="{purl}">{purl}</a></th>  <td>{peer_id}</td><td>{t_started}</td><td>{t_now}</td><td>{n_coordinators}</td><td>{n_workers}</td>  </tr>'
			.format(purl=purl, **xmlrpclib.ServerProxy(purl).stat()) for purl in peers ]
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

			server, _ = start_threaded_xmlrpc_server(SimpleXMLRPCServer.SimpleXMLRPCRequestHandler, 1023, self.hostname)
			coordinator = self.coordinators[task_id] = self._Coordinator(server, self.hostname, self.url, task_id, spec, data)
			server.register_instance(coordinator)
			server.register_introspection_functions()
			th = threading.Thread(name='Coordinator %d' % (self.coordinator_ctr-1,), target=server.serve_forever)
			th.daemon = True
			th.start()

		# Start the task in the spawned coordinator thread
		xmlrpclib.ServerProxy(coordinator.url).start()

		# Begin listening for notifications of events, yield them back to the user
		for msg in iter(coordinator.queue.get, ("DONE", None)):
			print "In loop: ", msg
			yield msg

		# delete this Coordinator task
		with self.lock:
			del self.coordinators[task_id]

	############################
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

			# Establish connection, record the worker (keyed by task_id)
			worker = self._Coordinator.WorkerProxy(wurl, self.url, worker_process)
			self.workers[task_id] = worker

			return wurl

	def _cleanup(self):
		with self.lock:
			for worker in self.workers.itervalues():
				if worker.process is not None:
					logger.info("Terminating process %d" % worker.process.pid)
					worker.process.terminate()
					worker.process.communicate()

	def shutdown(self):
		self.server.shutdown()
		return 0

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

class PeerRequestHandler(SimpleXMLRPCServer.SimpleXMLRPCRequestHandler):
	# Override do_POST() to permit stateful connections
	# for task submission and progress reporting
	def do_POST(self):
		logger.debug("New POST request")
		if self.path != '/execute':
			return SimpleXMLRPCServer.SimpleXMLRPCRequestHandler.do_POST(self)

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

	def do_GET(self):
		# Print out some statistics
		self.send_response(200)
		self.send_header("Content-type", "text/html")
		self.end_headers()

		self.wfile.write(self.server.instance._info())

	def _err_404(self, response):
		# Report a 404 error
		self.send_response(404)
		self.send_header("Content-type", "text/plain")
		self.send_header("Content-length", str(len(response)))
		self.end_headers()
		self.wfile.write(response)

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

def start_server(ServerClass, HandlerClass, port=1023, addr=''):
	# Find the next available port
	while port < 2**16:
		try:
			return ServerClass((addr, port), HandlerClass, allow_none=True), port
		except socket.error:
			port += 1

def start_threaded_xmlrpc_server(Handler, port=1023, addr=''):
	return start_server(core.ThreadedXMLRPCServer, Handler)

if __name__ == '__main__':
	## Setup logging ##
	format = '%(asctime)s.%(msecs)03d %(name)s[%(process)d] %(levelname)-8s {%(module)s:%(funcName)s}: %(message)s'
	datefmt = '%a, %d %b %Y %H:%M:%S'
	level = logging.DEBUG if (os.getenv("DEBUG", 0) == "1" or os.getenv("LOGLEVEL", "info") == "debug") else logging.INFO
	#filename = 'peer.log' if os.getenv("LOG", None) is None else os.getenv("LOG")
	#logging.basicConfig(filename=filename, format=format, datefmt=datefmt, level=level)
	logging.basicConfig(format=format, datefmt=datefmt, level=level)

	##logger.name = sys.argv[0].split('/')[-1]

	logger.info("Started %s", ' '.join(sys.argv))
	logger.debug("Debug messages turned ON")

	# Decide if we're launching a peer or a worker
	try:
		optlist, args = getopt.getopt(sys.argv[1:], 'w:', ['worker='])
	except getopt.GetoptError, err:
		print str(err)
		exit(-1)

	start_worker = False
	for o, a in optlist:
		if o in ('-w', '--worker'):
			start_worker = True
			hostname = a

	if start_worker:
		#import pydevd; pydevd.settrace(suspend=False, trace_only_current_thread=False)
		
		user_fn = args[0]
		argv = args[0:]

		# Start the worker server
		server, port = start_threaded_xmlrpc_server(SimpleXMLRPCServer.SimpleXMLRPCRequestHandler, 1023, hostname)
		worker = Worker(server, hostname)
		server.register_instance(worker)
		server.register_introspection_functions()

		###################
		# Import the user's code. Note: it's important this is done from
		# the main module!

		# Reset our argv to those of the user
		sys.argv = argv

		# Must prepend cwd to the module path, otherwise relative imports
		# from within the app won't work
		sys.path.insert(0, '.')

		# Load the user's Python app
		m = imp.load_source('_mrp2p_worker', user_fn)

		# Import its data to our __main__
		kws = ['__builtins__', '__doc__', '__file__', '__name__', '__package__', '__path__', '__version__']
		assert __name__ == '__main__'
		mself = sys.modules['__main__']
		for name in dir(m):
			if name in kws:
				continue
			setattr(mself, name, getattr(m, name))
		###################

		# Let the parent know where we're listening
		print worker.url
		sys.stdout.flush()

		# Start the XMLRPC server
		server.serve_forever()
	else:
		# Start the server
		logger.debug("Launching peer XMLRPC server")
		server, port = start_threaded_xmlrpc_server(PeerRequestHandler, 1023)
		peer = Peer(server, port)
		server.register_instance(peer)
		server.register_introspection_functions()

		try:
			# Register the Peer in the Peer directory
			peer._register()
			threading.current_thread().name = "Peer XMLRPC Server"
			server.serve_forever()
		except KeyboardInterrupt:
			pass;
		finally:
			peer._unregister()
			peer._cleanup()

		logging.debug("Remaining threads:")
		for th in threading.enumerate():
			logging.debug(th)

else:
	mrp2p_init()
	logger.debug("fn=%s, cwd=%s, argv=%s, len(env)=%s" % (fn, cwd, argv, len(env)))
