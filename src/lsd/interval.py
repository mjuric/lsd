#!/usr/bin/env python
"""
Basic interval arithmetic for LSD
"""
from bisect import bisect_left, bisect_right
import numpy as np

def intersect(i1, i2):
	# i1, i2 are two tuples
	if not i1 or not i2:
		return tuple()

	a1, b1 = i1
	a2, b2 = i2

	if b2 < a1 or b1 < a2:	# No overlap
		return tuple()

	# Compute the intersection
	a3 = a1 if a1 < a1 else a2
	b3 = b1 if b1 < b2 else b2
	
	return (a3, b3)

class intervalset:
	""" A simple class representing a set of closed intervals.

	    Methods of interest:
	    
	    add(i): adds a new interval i to the set
	    isInside(x): test whether x is in the interval
	    intersect(i): intersect with an interval i
	    
	    where interval i is a tuple (x0, x1)
	    
	    Also of interest are operators:
	    
	    |  : returns a union of two intervalsets
	    &  : returns an intersection of two intervalsets
	    == : tests for equality of intervalsets
	    
	    TODO: Rewrite this class in C (it's the bottleneck in
	          get_cells())
	"""
	def __init__(self, *ivals):
		self.data = []
		self.astuples_ = None
		for i in ivals:
			self.add(i)

	def add(self, i):
		# Add an interval to this interval set.
		if getattr(i, "__len__", None) == None:
			i = (i, i)
		elif len(i) == 1:
			i = (i[0], i[0])

		assert len(i) == 2
		assert i[0] <= i[1]

		# Find intervals overlaping this one
		il = bisect_left(self.data, i[0])
		ir = bisect_right(self.data, i[1])

		# Cut out all the points in the middle
		del self.data[il:ir]

		if il % 2 == 0:	# Left edge is outside an existing interval. Insert it
			self.data.insert(il, i[0])
			il += 1
		if ir % 2 == 0: # Right edge is outside an existing interval. Insert it
			self.data.insert(il, i[1])

		# Invalidate cache
		self.astuples_ = None

		assert len(self.data) % 2 == 0

	def __and__(self, iv):
		assert isinstance(iv, intervalset)

		ret = intervalset()
		for i in iv:
			for r in self.intersect(i):
				ret.add(r)
		return ret

	def __ior__(self, iv):
		assert isinstance(iv, intervalset)
		for i in iv:
			self.add(i)
		return self

	def intersect(self, i):
		""" Intersect this intervalset with a simple
		    tuple interval i
		"""
		assert len(i) == 2
		assert i[0] <= i[1]

		# Find intervals overlaping this one
		il = bisect_left(self.data, i[0])
		ir = bisect_right(self.data, i[1])

		# Cut out all the points in the middle
		iv = self.data[il:ir]

		if il % 2 == 1:	# Left edge is inside an existing interval. Insert it
			iv.insert(0, i[0])
		if ir % 2 == 1:
			iv.append(i[1])

		assert len(iv) % 2 == 0

		ret = intervalset()
		ret.data = iv
		return ret

	def isInside(self, x):
		""" Element-wise test if numpy array x is in the
		    interval set. Returns a bool NumPy array.
		    
		    Scales like O(len(Nivals)*len(Npts)) but since numpy
		    arrays are used, and for small interval sets, it's
		    usually fast enough (note that a Python-only "optimal"
		    O(log(Nivals)*len(Npts)) implementation would be many
		    times slower because of interpreter overhead)
		"""
		x = np.array(x, copy=False)
		ivals = np.array(self.data, copy=False)

		in_ = np.zeros(len(x), dtype=bool)
		for i in xrange(0, len(ivals), 2):
			in_ |= (ivals[i] <= x) & (x <= ivals[i+1])

		return in_

	def distance(self, x, dist_to_outside=False):
		x = np.asarray(x)
		
		ivals = np.zeros(len(self.data)+2)
		ivals[0]    = -np.inf
		ivals[1:-1] = self.data
		ivals[-1]   = np.inf

		at = np.searchsorted(ivals, x)
		d0 = ivals[at] - x
		d1 = x - ivals[at-1]
		dist = np.where(d0 < d1, d0, d1)	# Find nearest interval edge
		dist = np.where(at % 2 == int(dist_to_outside), 0, dist)	# Inside/outside the intervals
		
		return dist

	def __iter__(self):
		return iter(self.astuples())

	def __len__(self):
		return len(self.astuples())

	def __getitem__(self, idx):
		return self.astuples().__getitem__(idx)

	def astuples(self):
		if self.astuples_ == None:
			self.astuples_ = zip(self.data[::2], self.data[1::2])
		return self.astuples_

	def __eq__(self, b):
		if not isinstance(b, intervalset):
			return False
		return self.astuples() == b.astuples()

	def __str__(self):
		return str(self.astuples())
		
	def __repr__(self):
		return 'intervalset' + str(tuple(self.astuples()))

if __name__ == "__main__":
	iv = intervalset([1, 2], [3, 4], [-2, 0], [0.5, 1.5], [6, 6], [0.25, 0.25])
	print iv.data
	x = [-3, -2, -1, 0.5, 5, 2.5, 3.5, 0.25]
	print x
	print iv.isInside(x)
	assert intervalset([1, 2], [3, 4]) != intervalset([-2, 0], [0.5, 1.5])
	assert intervalset([1, 2], [3, 4]) == intervalset([1, 2], [3, 4])
	print "=== intersect tests ==="

	import numpy.random

	N = 10000; x0 = -4; x1 = 7;
	xa = zip(x0 + np.random.rand(N) * (x1-x0), x0 + np.random.rand(N)*(x1-x0))
	xa.extend([
		(-2, 0), (-2, 2), (0.25, 0.25), (1, 0.25), (-3, -2), (3, 7), (3, 6), (0.5, 3), (0.045, 0.045)
		])
	for x in xa:
		if x[0] > x[1]:
			x = (x[1], x[0])

		r1 = (iv & x).astuples()
		r2 = list((interval(*iv.astuples()) & interval(x))[:])
		##print iv, "&", x, "=\n   ", r1, "\n   ", r2
		assert r1 == r2, str(iv, "&", x, "=\n   ", r1, "\n   ", r2)
	print "OK"
