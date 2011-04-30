#!/usr/bin/env python

from multiprocessing import Process, Queue, cpu_count
from collections import defaultdict
import cPickle as pickle
import cPickle
import os
import sys
import tempfile
import time
import mmap
import traceback
import platform
from utils import unpack_callable

RET_KEYVAL = 1
RET_KEYVAL_LIST = 2

BUFSIZE = 100 * 2**20 if platform.architecture()[0] == '32bit' else 200 * 2**30

# OS X HFS+ filesystem does not support sparse files
back_to_disk = platform.system() != 'Darwin'

def _worker(ident, qcmd, qin, qout):
	""" Waits for commands on qcmd. Possible commands are:
		MAP: On MAP, store mapper and mapper_args, and
		     begin listening on qin for a stream of
		     items to be passed to mapper, until a
		     message 'DONE' is encountered. Return the
		     results yielded by mapper via qout.
	"""
	try:
		for cmd, args in iter(qcmd.get, 'EXIT'):
			if cmd == 'MAP':
				mapper, mapper_args = args

				i, item, result = None, None, None
				for (i, item) in iter(qin.get, 'DONE'):
					try:
						for result in mapper(item, *mapper_args):
							qout.put((ident, 'RESULT', (i, result)))
						qout.put((ident, 'DONE', i))
					except KeyboardInterrupt:
						return
					except:
						type, value, tb = sys.exc_info()
						tb_str = traceback.format_tb(tb)
						del tb    # See docs for sys.exec_info() for why this has to be here
						qout.put((ident, 'EXCEPT', (type, value, tb_str)))

				# Announce we're done with this mapper
				qout.put((ident, 'MAPDONE', None))

				# Immediately release memory
				del result, i, item
				del mapper, mapper_args
				del args
	except KeyboardInterrupt:
		pass;

def _unserializer(file, offsets):
	# Helper for _reduce_from_pickle_jar -- takes a filename and
	# a list of offsets, and returns a generator unpickling objects
	# at given offsets
	with open(file) as f:
		mm = mmap.mmap(f.fileno(), 0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)

		for offs in offsets:
			##print "Seek to ", offs
			mm.seek(offs)
			yield cPickle.load(mm)

		mm.close()

def _reduce_from_pickle_jar(kw, file, reducer, reducer_args):
	# open the pickle jar, load the objects, pass them on to the
	# actual reducer
	key, offsets = kw
	for result in reducer((key, _unserializer(file, offsets)), *reducer_args):
		yield result

def _output_pickled_kv(item, K_fun, K_args):
	# return a pickled value, deduplicating if possible
	unique_objects = set()

	for (k, v) in K_fun(item, *K_args):
		p = cPickle.dumps(v, -1)

		hash = digest(p)
		if hash in unique_objects:
			p = None

		yield (k, (hash, p))

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
				sys.stderr.write(" "),
				self.sign = ':'
			elif stage == "reduce":
#				sys.stderr.write('|'),
				self.sign = '+'
			sys.stderr.write("[%d el.]" % self.len),
	elif step == 'step':
		self.at = self.at + 1
		pct = 100. * self.at / self.len
		while self.pct <= pct:
			sys.stderr.write(self.sign)
			self.pct = self.pct + 5
	elif step == 'end':
		if stage == self.endstage:
			t = time.time() - self.t0
			sys.stderr.write('>  %.2f sec\n' % t)
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
	qcmd = None
	qin = None
	qout = None
	ps = []
	min_tasks_for_parallel = 3
	DEBUG = None	# Filled in in __init__ from getenv
	nworkers = None	# Filled in in __init__ from getenv or cpu_count()

	def __del__(self):
		self.close()

	def close(self):
		if len(self.ps) == 0:
			return

		for q in self.qcmd:
			q.put('EXIT')

		kill = []
		for p in self.ps:
			try:
				p.join(5)
				if p.is_alive():
					kill.append(p)
			except:
				kill.append(p)
				pass

		for p in kill:
			p.terminate()

		self.qcmd = self.qin = self.qout = None
		del self.ps[:]

	def terminate(self):
		for p in self.ps:
			p.terminate()

		self.qcmd = self.qin = self.qout = None
		if len(self.ps):
			del self.ps[:]

	def _create_workers(self):
		""" Lazily create workers, when needed. This routine
		    creates the worker processes when called the first
		    time.
		"""
		if len(self.ps) == self.nworkers:	# Already created?
			return

		self.qin = Queue()
		self.qout = Queue(self.nworkers*2)
		self.qcmd = [ Queue() for _ in xrange(self.nworkers) ]
		self.ps = [ Process(target=_worker, args=(i, self.qcmd[i], self.qin, self.qout)) for i in xrange(self.nworkers) ]

		for p in self.ps:
			p.daemon = True
			p.start()

	def __init__(self, nworkers = None):
		self.DEBUG    = int(os.getenv('DEBUG', False))
		self.nworkers = int(os.getenv('NWORKERS', cpu_count()))

		if nworkers != None:
			self.nworkers = nworkers

	def imap_unordered(self, input, mapper, mapper_args=(), progress_callback=None, progress_callback_stage='map'):
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
			try:
				self._create_workers()

				# Initialize this map
				for q in self.qcmd:
					q.put( ('MAP', (mapper, mapper_args)) )

				# Queue the data to operate on
				i = -1
				for (i, item) in enumerate(input):
					self.qin.put( (i, item) )
				n = i + 1

				# Queue the end-of-map markers
				for _ in xrange(self.nworkers):
					self.qin.put('DONE')

				# yield the outputs
				k = 0
				wf = 0	# Number of workers that have finished
				while wf != self.nworkers:
					(ident, what, data) = self.qout.get()
					if what == 'RESULT':
						i, result = data
						yield result
					elif what == 'MAPDONE':
						wf += 1
						#print "MAPDONE----------", ident
					elif what == 'DONE':
						k += 1
						progress_callback(progress_callback_stage, 'step', input, k, None)
						#print "DONE with %s of %s" % (k, len(input)), ident
						continue
					elif what == 'EXCEPT':
						# Unhandled Exception was raised in one of the workers.
						# Terminate the workers and re-raise it.
						self.terminate()
						type, value, tb_str = data
						print >> sys.stderr, 'Remote Traceback (most recent call last):\n', ''.join(tb_str)
						raise value

				assert wf == self.nworkers	# All workers must have finished
				assert k == n			# All items must have been processed
			except:
				# Terminate the workers if an exception ocurred
				self.terminate()
				raise
		else:
			# Execute in-thread, without external workers
			for (i, item) in enumerate(input):
				for result in mapper(item, *mapper_args):
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

		if back_to_disk:
			unique_objects = {}
			fp, prev_fp = None, None

		for i, K in enumerate(kernels):
			K_fun, K_args = unpack_callable(K)
			last_step = (i + 1 == len(kernels))
			stage = where(i == 0, 'map', 'reduce')

			if back_to_disk:
				# Insert picklers/unpicklers
				if i != 0:
					# Insert unpickler
					K_fun, K_args = _reduce_from_pickle_jar, (prev_fp.name, K_fun, K_args)
#					if i == 2: exit()

				if not last_step:
					# Insert pickler
					K_fun, K_args = _output_pickled_kv, (K_fun, K_args)

					# Create a disk backing store for intermediate results
					fp = tempfile.NamedTemporaryFile(mode='wb', prefix='mapresults-', dir='/nfs_pan/mjuric', suffix='.pkl', delete=True)
					fd = fp.file.fileno()
					os.ftruncate(fd, BUFSIZE)
					mm = mmap.mmap(fd, 0)

			# Call the distributed mappers
			mresult = defaultdict(list)
			for r in self.imap_unordered(input, K_fun, K_args, progress_callback=progress_callback, progress_callback_stage=stage):
				if last_step:
					# yield the final result
					yield r
				else:
					(k, v) = r

					if back_to_disk:
						(hash, v) = v
						if hash in unique_objects:
							v = unique_objects[hash]
						else:
							# The output value has already been pickled (but not the key). Store the
							# pickled value into the pickle jar, and keep the (key, offset) tuple.
							offs = mm.tell()
							mm.write(v)
							assert len(v) == mm.tell() - offs
							v = offs
							unique_objects[hash] = offs

					# Prepare for next reduction
					mresult[k].append(v)

			input = mresult.items()

			if back_to_disk:
				# Close/clear the intermediate result backing store from the previous step
				if prev_fp is not None:
					prev_mm.resize(1)
					prev_mm.close()
					os.ftruncate(prev_fp.file.fileno(), 0)
					prev_fp.close()

				if fp is not None:
					prev_fp, prev_mm = fp, mm

		if progress_callback != None:
			progress_callback('mapreduce', 'end', None, None, None)

def digest(s):
	import hashlib
	#return hashlib.md5(s).hexdigest()
	#return hashlib.sha1(s).digest()
	return hashlib.md5(s).digest()

############ Unit tests

# ====
def _test_pool2_add(a, b):
	yield a+b
# ====
def _test_pool2_add1(a, b):
	yield a, a+b

def _test_pool2_subtract(kv, b, c):
	_, gen = kv
	for a in gen:
		yield a+b
		yield a-c
# ====
def _test_mapred2_map(a, b, c):
	yield a + b, 1
	yield a + c, a

def _test_mapred2_red1(kv):
	k, v = kv
	yield k, sum(v)

def _test_mapred2_red2(kv, d):
	k, v = kv
	for val in v:
		yield val + d
# ====

class Test_Pool:
	@classmethod
	def setUpClass(self):
		global np, sys
		import numpy as np
		import sys
		self.pool = Pool()

	def test_imap(self):
		""" Mapper """
		for k in [1, 2, 3, 4, 5, 7, 8, 10, 12, 16, 20, 50, 100, 200, 600]:
			arr = np.arange(k)
			b = 10

			it = self.pool.imap_unordered(arr, _test_pool2_add, (b,), progress_callback=progress_pass)
			res = np.fromiter(it, dtype=arr.dtype)

			res.sort()
			arr.sort()
			assert np.all(res == arr+b)

	def test_mapred1(self):
		""" Map-Reduce: simple """
		for k in [1, 2, 3, 4, 5, 7, 8, 10, 12, 16, 20, 50, 100, 200, 600]:
			arr = np.arange(k)
			b = 10
			c = -10

			it = self.pool.map_reduce_chain(arr, [ (_test_pool2_add1, b), (_test_pool2_subtract, b, c) ], progress_callback=progress_pass)
			res = np.fromiter(it, dtype=arr.dtype)
			res.sort()

			arr = arr + b
			arr = np.concatenate([arr + b, arr - c])
			arr.sort()

			assert np.all(res == arr)

	def test_mapred2(self):
		""" Map-Reduce: three-stage """
		for k in [1, 2, 3, 4, 5, 7, 8, 10, 12, 16, 20, 50, 100, 200, 600]:
			arr = np.arange(k)
			b = 10
			c = -4
			d = 3

			it = self.pool.map_reduce_chain(arr, [ (_test_mapred2_map, b, c), _test_mapred2_red1, (_test_mapred2_red2, d) ], progress_callback=progress_pass)
			res = np.fromiter(it, dtype=arr.dtype)
			res.sort()

			r1  = [ (a + b, 1) for a in arr ]
			r1 += [ (a + c, a) for a in arr ]
			r2 = defaultdict(np.int64)
			for k, v in r1:
				r2[k] += v
			r3 = np.array(r2.values(), dtype=np.int64) + d
			r3.sort()

			print res
			print r3
			assert np.all(res == r3)

