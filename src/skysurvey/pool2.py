#!/usr/bin/env python

from multiprocessing import Process, Queue, cpu_count
from collections import defaultdict
import cPickle as pickle
import os
import sys
import tempfile
import time
from utils import unpack_callable

def _worker(qin, qout):
	# Dispatch the call to the target function, collecting
	# back yielded results and returning them as a list
	for mapper, i, args in iter(qin.get, 'EXIT'):
		for result in mapper(*args):
			qout.put((i, result))
		qout.put("DONE")

def _decoder(message):
	# Decode a message encoded by the _worker and yield
	# back the results
	for (i, result) in message:
		return (i, result)

def _reduce_from_pickled(kw, pkl, reducer, args):
	# open the piclke jar, load the objects, pass them on to the
	# actual reducer
	d = open(pkl, 'rb')
	k, vd = kw
	va = []
	for offs in vd:
		d.seek(offs)
		obj = pickle.load(d)
		va.append(obj)
	d.close()

	return reducer((k, va), *args)

def progress_default(stage, step, input, index, result):
	self = progress_default

	if  step == 'begin':
		if '__len__' in dir(input):
			self.dispatch = progress_pct
		else:
			self.dispatch = progress_dots

	self.dispatch(stage, step, input, index, result)

def progress_pct(stage, step, input, index, result):
	self = progress_pct

	# Record the first 'begin' stage as the endstage
	if step == 'begin' and 't0' not in dir(self):
		self.t0 = time.time()
		self.endstage = stage
		self.head = 'm/r' if stage == 'mapreduce' else 'm'

	if step == 'begin' and (stage == 'map' or stage == 'reduce'):
			self.len = len(input)
			self.at = 0
			self.pct = 5

			if   stage == 'map':
				sys.stderr.write("[%s (%d elem): " % (self.head, self.len)),
				self.sign = ':'
			elif stage == "reduce":
				sys.stderr.write('|'),
				self.sign = '+'
	elif step == 'step':
		self.at = self.at + 1
		pct = 100. * self.at / self.len
		while self.pct <= pct:
			sys.stderr.write(self.sign)
			self.pct = self.pct + 5
	elif step == 'end':
		if stage == self.endstage:
			t = time.time() - self.t0
			sys.stderr.write(']  %.2f sec\n' % t)
			del self.t0

def progress_dots(stage, step, input, index, result):
	if step == 'begin':
		if   stage == 'map':
			sys.stderr.write("[map: "),
		elif stage == "reduce":
			sys.stderr.write(' [reduce: '),
	elif step == 'step':
		sys.stderr.write("."),
	elif step == 'end':
		sys.stderr.write(']')

def progress_pass(stage, step, input, index, result):
	pass

def where(cond, a, b):
	""" A readable C-ish ternary operator.
	"""
	return a if cond else b

class Pool:
	qin = None
	qout = None
	ps = []
	DEBUG = int(os.getenv('DEBUG', False))
	min_tasks_for_parallel = 3
	nworkers = int(os.getenv('NWORKERS', cpu_count()))

	def _create_workers(self):
		""" Lazily create workers, when needed. This routine
		    creates the worker processes when called the first
		    time.
		"""
		if len(self.ps) == self.nworkers:	# Already created?
			return

		self.qin = Queue()
		self.qout = Queue(self.nworkers*2)
		self.ps = [ Process(target=_worker, args=(self.qin, self.qout)) for _ in xrange(self.nworkers) ]

		for p in self.ps:
			p.daemon = True
			p.start()

	def __init__(self, nworkers = None):
		if nworkers != None:
			self.nworkers = nworkers

	def imap_unordered(self, input, mapper, mapper_args=(), return_index=False, progress_callback=None, progress_callback_stage='map'):
		""" Execute in parallel a callable <mapper> on all values of
		    iterable <input>, ensuring that no more than ~nworkers
		    results are pending in the output queue """
		if progress_callback == None:
			progress_callback = progress_default;

		progress_callback(progress_callback_stage, 'begin', input, None, None)

		# Try to optimize and not dispatch to workers if there are less
		# than self.min_tasks_for_parallel tasks
		try:
			parallel = len(input) >= self.min_tasks_for_parallel
		except TypeError:
			parallel = True

		parallel = parallel and self.nworkers > 1 and not self.DEBUG

		# Dispatch/execute
		if parallel:
			# Create workers (if not created already)
			self._create_workers()

			i = -1
			for (i, val) in enumerate(input):
				self.qin.put( (mapper, i, (val,) + mapper_args) )
			n = i + 1

			# yield the outputs
			k = 0
			while k != n:
				ret = self.qout.get()
				if isinstance(ret, str) and ret == 'DONE':
					k += 1
					progress_callback(progress_callback_stage, 'step', input, k, None)
					continue

				(i, result) = ret
				if return_index:
					yield (i, result)
				else:
					yield result
		else:
			# Execute in-thread, without external workers
			for (i, val) in enumerate(input):
				for result in mapper(val, *mapper_args):
					if return_index:
						yield (i, result)
					else:
						yield result
				progress_callback(progress_callback_stage, 'step', input, i, None)

		progress_callback(progress_callback_stage, 'end', input, None, None)

	def imap_reduce(self, input, mapper, reducer, mapper_args=(), reducer_args=(), progress_callback=None):
		""" A poor-man's map-reduce implementation.
		
		    Calls the mapper for each value in the <input> iterable. 
		    The mapper shall return a list of key/value pairs as a
		    result.  Once all mappers have run, reducers will be
		    called with a key, and a list of values associated with
		    that key, once for each key.  The reducer's return
		    values are yielded to the user.

		    Input: Any iterable
		    Output: Iterable
		    
		    Notes:
		    	- mapper must return a dictionary of (key, value) pairs
		    	- reducer must expect a (key, value) pair as the first
		    	  argument, where the value will be an iterable
		"""

		if progress_callback == None:
			progress_callback = progress_default
		
		progress_callback('mapreduce', 'begin', input, None, None)

		# Map step
		mresult = defaultdict(list)
		for r in self.imap_unordered(input, mapper, mapper_args, progress_callback=progress_callback, progress_callback_stage='map'):
			for (k, v) in r:
				mresult[k].append(v)

		# Reduce step
		for r in self.imap_unordered(mresult.items(), reducer, reducer_args, progress_callback=progress_callback, progress_callback_stage='reduce'):
			if len(r) > 2:
				print r
			yield r

		if progress_callback != None:
			progress_callback('mapreduce', 'end', None, None, None)

	def imap_reduce_big(self, input, mapper, reducer, mapper_args=(), reducer_args=(), progress_callback=None):
		#
		# Notes: same interface as imap_reduce, except that the outputs of
		#        map phase are assumed to be large and are cached on 
		#        the disk using cPickle. The (key->index on disk) mappings
		#        are still held in memory, so make sure those don't grow
		#        too large.
		#

		if progress_callback == None:
			progress_callback = progress_default
		
		progress_callback('mapreduce', 'begin', input, None, None)

		# Map step
		d = tempfile.NamedTemporaryFile(mode='wb', prefix='mapresults-', suffix='.pkl', delete=False)
		mresult = defaultdict(list)
		for r in self.imap_unordered(input, mapper, mapper_args, progress_callback=progress_callback, progress_callback_stage='map'):
			for (k, v) in r:
				mresult[k].append(d.tell())
				pickle.dump(v, d, -1)
		d.close()

		# Reduce step
		for r in self.imap_unordered(mresult.iteritems(), _reduce_from_pickled, (d.name, reducer, reducer_args), progress_callback=progress_callback, progress_callback_stage='reduce'):
			yield r

		os.unlink(d.name)

		if progress_callback != None:
			progress_callback('mapreduce', 'end', None, None, None)

	def map_reduce_chain(self, input, kernels, progress_callback=None):
		""" A poor-man's map-reduce implementation.
		
		    Calls the mapper for each value in the <input> iterable. 
		    The mapper shall return a list of key/value pairs as a
		    result.  Once all mappers have run, reducers will be
		    called with a key, and a list of values associated with
		    that key, once for each key.  The reducer's return
		    values are yielded to the user.

		    Input: Any iterable
		    Output: Iterable (generated)

		    Notes:
		    	- mapper must return a dictionary of (key, value) pairs
		    	- reducer must expect a (key, value) pair as the first
		    	  argument, where the value will be an iterable
		"""

		if progress_callback == None:
			progress_callback = progress_default

		progress_callback('mapreduce', 'begin', input, None, None)

		for i, K in enumerate(kernels):
			K_fun, K_args = unpack_callable(K)
			last_step = (i + 1 == len(kernels))
			stage = where(i == 0, 'map', 'reduce')

			mresult = defaultdict(list)
			for r in self.imap_unordered(input, K_fun, K_args, progress_callback=progress_callback, progress_callback_stage=stage):
				if last_step:
					# yield the final result
					yield r
				else:
					# Prepare for next reduction
					(k, v) = r
					mresult[k].append(v)

			input = mresult.items()

		if progress_callback != None:
			progress_callback('mapreduce', 'end', None, None, None)

#	def distributed_map_reduce_chain(self, input, kernels, partitioners, progress_callback=None):
#		""" A poor-man's map-reduce implementation.
#		
#		    Calls the mapper for each value in the <input> iterable. 
#		    The mapper shall return a list of key/value pairs as a
#		    result.  Once all mappers have run, reducers will be
#		    called with a key, and a list of values associated with
#		    that key, once for each key.  The reducer's return
#		    values are yielded to the user.
#
#		    Input: Any iterable
#		    Output: Iterable (generated)
#
#		    Notes:
#		    	- mapper must return a dictionary of (key, value) pairs
#		    	- reducer must expect a (key, value) pair as the first
#		    	  argument, where the value will be an iterable
#		"""
#
#		# Launch first mapping, and wait for the wave to finish
#		mr = MapReduceSwarm(16, kernels, partitioners)
#		mr.run()
#
#		for v in input:
#			mr.push(v)
#		for (msg, r) in mr:
#			if msg == 'result':
#				yield
#			else:
#				... progress report or something ...
#				... we could pass this to the application layer as well ...
#
#
#		class MapReduceSwarm:
#			def __init__(self, nmembers, kernels, partitioners):
#				self.comm = MPI.COMM_WORLD
#				self.rank = MPI.Get_rank()
#
#				if self.rank == 0:
#					# Make yourself a daemon, listening for user commands
#					# on a given socket
#					...
#				else:
#					# Listen for master's commands through MPI
#					(cmd, args) = comm.recv(source=0)
#					if cmd == 'run':
#						...
#
#			def push(self, val):
#				...
#
#		# - controller: receives messages from workers (progress or results)
#		# - workers: just work
#		#
#		# - Each process should have two threads (subprocesses) -- one for
#		#   communicating with MPI (monitor), one for the actual work.
#		#   - e.g., the monitor could then receive and store data sent
#		#     to it by nodes wishing to reduce
#		#   - might be good if these were two separate MPI instances?
#
#		def cell_partitioner(hosts, key):
#			# Cell partitioner with no location awareness (assumes shared storage)
#			# Always places the same key to the same host (assuming constant len(hosts))
#			hash = int(hashlib.md5(str(key)).hexdigest(), base=16)
#			return hosts[hash % len(hosts)]
#
#		def cell_partitioner(hosts, key, args):
#			# location-aware assignment partitioner might look something like this
#			(cat, occ) = args
#			
#			# Get a list of suitable hosts (this is all propagated at initialization time)
#			chosts = cat.get_hosts_containing_cell(key, hosts)
#
#			# Choose one at random
#			return chosts[random_integer(0, len(chosts)-1)]
#