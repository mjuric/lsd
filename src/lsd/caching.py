#
# A generalized object cache
#

import fcntl
import cPickle
import tempfile
import getpass
import hashlib
import os.path
import errno
from contextlib  import contextmanager
from collections import OrderedDict

import functools

class CallResultCache(object):
	"""
		A persistent function result cache

		Note: !! Not thread safe !!
	"""
	cache_dir = None	# On-disk cache directory
	max_memcache = None	# Maximum (pickled) memsize of memory-cached objects (in bytes)

	__cache = None		# Memory cache
	__memcacesize = None	# Current approximate size of memory-cached data

	def __init__(self, cache_dir = None, max_memcache = 0.5 * 2**30):
		if cache_dir is None:
			cache_dir = tempfile.gettempdir() + '/_lsd_caching-' + getpass.getuser()

		self.cache_dir = cache_dir
		self.max_memcache = max_memcache
		
		self.__cache = OrderedDict()
		self.__memcachesize = 0

	@contextmanager
	def lock(self, fp):
		""" Lock the cache file """
		fcntl.lockf(fp, fcntl.LOCK_EX)

		yield fp

		fcntl.lockf(fp, fcntl.LOCK_UN)

	def cached(self, func):

		@functools.wraps(func)
		def wrapper(*args, **kwds):
			def stats():
				""" Return performance counters """
				return ((wrapper.hits, wrapper.misses),
					(wrapper.mem_hits, wrapper.mem_misses),
					(wrapper.disk_hits, wrapper.disk_misses))
			wrapper.stats = stats

			# Compute arg hash and cache filename
			funcname = func.__module__ + "." + func.__name__
			allargs = args + tuple(sorted(kwds.items()))
			hash = funcname + '-' + hashlib.md5(cPickle.dumps(allargs)).hexdigest()

			# See if it's in memory cache (LRU)
			try:
				result, pklsize = self.__cache.pop(hash)
				self.__memcachesize -= pklsize

				wrapper.hits += 1
				wrapper.mem_hits += 1
			except KeyError:
				wrapper.mem_misses += 1

				# Make sure the cache directory exists
				try:
					os.makedirs(self.cache_dir)
				except OSError as e:
					if e.errno != errno.EEXIST:
						raise

				# Disk cache lookup
				fn = self.cache_dir + '/' + hash + '.pkl'
				try:
					with file(fn, "a+b") as fp, self.lock(fp):
						if fp.tell() != 0:
							# Cache exists
							fp.seek(0)
							result = cPickle.load(fp)
							pklsize = fp.tell()

							wrapper.hits += 1
							wrapper.disk_hits += 1
						else:
							# New cache
							result = func(*args, **kwds)
							cPickle.dump(result, fp, -1)
							pklsize = fp.tell()

							wrapper.misses += 1
							wrapper.disk_misses += 1
				except IOError:
					# Something went wrong w. opening/reading the file
					# Evaluate but don't store the result
					result = func(*args, **kwds)
					pklsize =self.max_memcache

					wrapper.misses += 1
					wrapper.disk_misses += 1

				# Memcache sizing -- pop least recently used items to make
				# room for the new one
				if pklsize < self.max_memcache:
					while len(self.__cache) and (self.__memcachesize + pklsize > self.max_memcache):
						_key, (_, size) = self.__cache.popitem(0)
						self.__memcachesize -= size
						#print "Dropping %s [val=%s, size=%d]" % (_key, _, size)

			if self.__memcachesize + pklsize < self.max_memcache:
				self.__cache[hash] = result, pklsize
				self.__memcachesize += pklsize
				#print "Memcached  %s [val=%s, size=%d]" % (hash, result, pklsize)

			#print "memcachesize: ", self.__memcachesize
			return result

		wrapper.hits = wrapper.misses = 0
		wrapper.mem_misses = wrapper.mem_hits = 0
		wrapper.disk_misses = wrapper.disk_hits = 0

		return wrapper

## Default cache object
oc = CallResultCache()
cached = oc.cached

## Some testing code
if __name__ == "__main__":
	import numpy as np
	#oc = CallResultCache("mycache", 20000)
	print "cache_dir=", oc.cache_dir

	@cached
	def addnum(a, b):
		return a + b

	@cached
	def subnum(a, b):
		return a - b

	print addnum(2, 2)
	print subnum(4, 4)
	print addnum(2, 3)
	print addnum(2, 3)
	print addnum(2, 2)
	print len(addnum(np.arange(2**10), np.arange(2**10)))
	print len(addnum(np.arange(2**10)+1, np.arange(2**10)))
	print len(addnum(np.arange(2**10)+3, np.arange(2**10)))
	print len(addnum(np.arange(2**10), np.arange(2**10)))
	print len(addnum(np.arange(10*2**10), np.arange(10*2**10)))

	print addnum.stats()
	print subnum.stats()
