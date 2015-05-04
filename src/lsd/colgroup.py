#!/usr/bin/env python
"""
Implementation of ColGroup class

ColGroup is a functional equivalent of structured numpy.ndarray, the
difference being that a structured array is layed out row-by-row (in
memory), while a ColGroup is layed out column-by-column.

It's implemented as an ordered list of columns, with each column a ndarray
of equal length.
"""

import numpy as np
from utils import full_dtype
import copy
import tempfile, os, cPickle, sys

def make_record(dtype):
	# Construct a numpy record corresponding to a row of this table
	# There has to be a better way to construct a numpy.void of a given dtype, but I couldn't figure it out...
	tmp = np.zeros(1, dtype)
	return tmp[0].copy()

class RowIter:
	"""
	Iterator to permit iteration over ColGroup by row
	
	WARNING: TERRIBLY INEFFICIENT AND FLAWED AS DESIGNED. NEEDS
	A COMPLETE REWRITE. AVOID IF YOU CAN.
	"""
	def __init__(self, cgroup):
		self.cgroup = cgroup
		self.at = 0
		self.end = cgroup.nrows()

		if self.at < self.end:
			# Cache some useful data
			self.column_data = cgroup.column_data
			self.column_map = cgroup.keys()
#			self.row = cgroup.row.copy()
			self.row = make_record(cgroup.dtype)

	# Iterator protocol methods
	def __iter__(self):
		return self

	def next(self):
		if self.at >= self.end:
			raise StopIteration()

		for i, name in enumerate(self.column_map):
			val = self.column_data[i][self.at]
			# Special care for multidimensional columns
			if isinstance(val, np.ndarray) and self.row.dtype[name] != object:
				self.row[name][:] = val
			else:
				self.row[name] = val

		self.at = self.at + 1

		return self.row

class InfoInstance:
	"""
	Container for ColGroup.info. Defined outside ColGroup to be
	pickleable.
	"""
	pass;

class ColGroup(object):
	"""
	A structured array, stored by column instead of by row.

	ColGroup is a functional equivalent of structured numpy.ndarray, the
	difference being that a structured array is layed out row-by-row (in
	memory), while a ColGroup is layed out column-by-column.  It's
	implemented as a list of columns, with each column being a ndarray
	of equal length).
	
	The most frequently used operations are indexing and slicing
	(__getitem__), and addition/removal of new columns. Indexing/slicing
	works just as with ndarray; e.g.:

		>>> cg2 = cg[::3]
		
	will return a new ColGroup, containing every third row of the
	original one.

	Addition and removal of columns is fast. See add_column/add_columns
	and drop_column.

	To obtain an ordered list of all column names, use .keys().
	To obtain an ordered list of (name, column) tuples, use .items().
	To obtain the number of rows, use len(cg).
	
	Sorting is possible using .sort()
	"""

	column_map = None	# Map from name->pos and pos->name
	column_data = None	# A list of columns (individual numpy arrays)
	info = None		# A class holding any user-defined tags

	##
	## dict-like interface implementation
	##
	def keys(self):
		"""
		Returns an ordered list of columns.

		The columns appear in order in which they were added.
		"""
		return [ self.column_map[pos] for (pos, _) in enumerate(self.column_data) ]

	def items(self):
		"""
		Returns an ordered list of (column_name, column_data)
		
		The tuples appear in the order in which the columns were
		added. The 'column_data' element is the ndarray containing
		the column's data.
		"""
		items = [ (self.column_map[pos], self.column_data[pos]) for pos in xrange(len(self.column_data)) ]
		return items

	def __getattr__(self, name):
		"""
		Allow accessing columns as attributes

		'name' is the name of the column to return.
		"""
		if (self.column_map is None) or (name not in self.column_map):
			raise AttributeError('Column %s not found.' % name)

		return self.column_data[self.column_map[name]]

	def __setattr__(self, name, value):
		"""
		Allow adding (or replacing) columns as attributes.
		"""
		if name in ['column_map', 'column_data', 'info']:
			object.__setattr__(self, name, value)
		else:
			self[name] = value

	def __delattr__(self, name):
		"""
		Delete a column from the group.
		"""
		self.drop_column(name)

	def _get_slice(self, key):
		cg = ColGroup(info=self.info)
		cg.column_map = self.column_map.copy()
		cg.column_data = [ data[key] for data in self.column_data ]
		return cg

	def __getitem__(self, key):
		"""
		Slicing and indexing.
		
		This method is functionally equivalent to
		ndarray.__getitem__. It returns a ColGroup as a result,
		unless the key was a scalar in which case a numpy record
		is returned.

		All numpy fancy indexing and slicing will work just as it
		does with ndarrays. An extension is if a list or tuple of
		strings is passed as an index, e.g.:
		
			>>> cg2 = cg[("ra", "dec")]
			
		in which case a ColGroup consisting of only the named
		columns is returned as a result.
		"""
		# return a subcgroup if column names (or a tuple or
		# a list of them) were given
		if isinstance(key, str) or isinstance(key, list) or isinstance(key, tuple):
			return self.subset(key)

		# Compute the slice
		if len(self) == 0 and isinstance(key, np.ndarray):
			# Otherwise multidimensional arrays will throw an IndexError even if len(key) = 0
			# (I personally think that's an annoying inconsistency in numpy)
			assert len(key) == 0
			key = np.s_[:]

		# Return a ColGroup if the key was an instance of a slice or ndarray,
		# and numpy.void structure otherwise
		if isinstance(key, slice) or isinstance(key, np.ndarray):
			return self._get_slice(key)
		else:
			#row = self.row.copy()
			row = make_record(self.dtype)
			for pos, col in enumerate(self.column_data):
				name = self.column_map[pos]
				val = col[key]
				# Special care for vector columns
				if isinstance(val, np.ndarray):
					row[name][:] = val
				else:
					row[name] = val
			return row

	def drop_column(self, key):
		"""
		Remove the named column from the ColGroup.
		
		Note: This operation is computationally cheap, relative
		to the same operation with a structured ndarray.
		"""
		pos  = self.column_map[key] if     isinstance(key, str) else key
		cols = self.items()

		del self.column_data[pos]
		del cols[pos]

		self.column_map = dict(( (colname, pos)    for (pos, (colname, _)) in enumerate(cols) ))
		self.column_map.update(( (pos, colname)    for (pos, (colname, _)) in enumerate(cols) ))

	def __setitem__(self, key, value):
		"""
		Set the data within a ColGroup, or append a new column.
		
		This method is a functional equivalent of the eponymous
		method in ndarray.
		
		If the key is a string that is not one of the column name,
		the value is assumed to be a ndarray to be added to this
		ColGroup (see ColGroup.add_column())
		"""
		if isinstance(key, str):
			if key in self.column_map:
				self.column_data[self.column_map[key]][:] = value
			else:
				self.add_column(key, value)
		elif isinstance(key, slice):
			if isinstance(value, ColGroup):
				# fast but specialized
				assert value.keys() == self.keys()
				for (pos, _) in enumerate(self.column_data):
					self.column_data[pos][key] = value.column_data[pos]
			else:
				# slow but generic
				it_v = iter(value)
				for i in xrange(*key.indices(len(self))):
					self[i] = next(it_v)
		else:
			# Assume we're changing a single row
			for (pos, val) in enumerate(value):
				self.column_data[pos][key] = val

	def __contains__(self, column):
		"""
		Test if a column exists in the ColGroup
		"""
		return column in self.column_map

	#############

	@property
	def dtype(self):
		"""
		Return the dtype this column group would have if it was a
		numpy structured array
		"""
		return np.dtype([ (self.column_map[pos], full_dtype(col)) for (pos, col) in enumerate(self.column_data) ])

	@property
	def shape(self):
		"""
		Return the shape this column group would have if it was a
		numpy structured array
		"""
		return (len(self), )

	def column(self, idx):
		"""
		Return the column at index idx.
		"""
		return self.column_data[idx]

	def __init__(self, cols=[], dtype=None, size=0, info=None):
		"""
		Construct the column group.

		Build the column group out of an iterable, cols, that is
		expected to yield (colname, coldata) tuples, where coldata
		is a numpy array. All data elements in the tuples must be of
		the same length.

		Alternativey, if cols.items() exists, it will be called to
		yield the (colname, coldata) tuples. This allows you to pass
		a dict (or, more likely, an OrderedDict) as the cols
		argument.
		
		Finally, instead of specifying cols, you can specify the
		dtype and size of the ColGroup via dtype and size arguments.
		If dtype is specified, cols must be None or a zero-length
		array.
		"""
		self.column_map = dict()
		self.column_data = []
		if info is not None:
			self.info = copy.copy(info)
		else:
			self.info = InfoInstance()

		if dtype is not None:
			# construct cols from structured array dtype
			assert cols is None or len(cols) == 0
			# TODO: This is very inefficient (2x more RAM used than needed)
			template = np.zeros(size, dtype=dtype)
			cols = [ (name, template[name].copy()) for name in template.dtype.names ]

		# Detect dict() and ColGroup()s
		if getattr(cols, 'items', None) is not None:
			cols = cols.items()

		# Detect structured ndarrays
		if isinstance(cols, np.ndarray) and cols.dtype.names is not None:
			cols = [ (name, cols[name]) for name in cols.dtype.names ]

		self.add_columns(cols)

#		self._mk_row()

	def add_columns(self, cols):
		"""
		Add a list (iterable) of (colname, coldata) pairs
		to the ColGroup.
		
		This is a convenience method that repeatedly calls
		add_column() for each pair yielded from cols.
		"""
		for name, col in cols:
			self.add_column(name, col)

	def add_column(self, name, col, dtype=None):
		"""
		Append a column to the ColGroup.
		
		Parameters
		----------
		name : string
		    The name of the new column.
		col : ndarray or other
		    The data for the column. If ndarray is given, it must be of
		    the same length as existing columns (if any). If a
		    other is given, a vector will be constructed of the
		    same length as the existing columns (or one, if there
		    are none), and the element replicated everywhere.
		dtype : numpy.dtype
		    If dtype is given, and col is a scalar, dtype


		Examples
		--------
		Add a simple column, from ndarray

		>>> x = np.array([1, 2, 3])
		>>> cg = c.ColGroup();
		>>> cg.add_column('x', x)

		Add a column from a scalar

		>>> cg.add_column('s', 2.)

		Add a column from an array

		>>> cg.add_column('a', [1, 3, 4])
		>>> cg['a']
		array([[1, 3, 4],
		[1, 3, 4],
		[1, 3, 4]])
		>>> cg['a'].dtype
		>>> dtype('int64')
		"""
		if not isinstance(col, np.ndarray):
			# permit scalars to initialize new columns
			col = np.array([col], dtype=dtype, ndmin=1)
			col = np.resize(col, (len(self),) + col.shape[1:]) # This will resize, replicating the first element everywhere

		# sanity check: require numpy arrays
		assert isinstance(col, np.ndarray)
		assert self.ncols() == 0 or (len(col) == self.nrows()), str(len(col)) + " != " + str(self.nrows())
		assert name not in self.column_map

		# Add a column to the end
		pos = len(self.column_data)
		self.column_map[name] = pos
		self.column_map[pos] = name
		self.column_data.append(col)

#		if not supress_row_update:
#			self._mk_row()

#	def _mk_row(self):
#		if self.ncols() != 0:
#			self.row = make_record(self.dtype)

	def resize(self, size, refcheck=True):
		"""
		Resize the columns to new size
		
		Resizes each column in the ColGroup to the requested size.
		If the new size is greater than the current length, the
		contents of newly created elements is undefined
		(== implementation-dependend and may change in the future).
		"""
		for (pos, _) in enumerate(self.column_data):
			col = self.column_data[pos].copy()
			col.resize((size,) + col.shape[1:])
			self.column_data[pos] = col

#	def append_rows(self, other):
#		# Sanity checks:
#		#	1. All columns from the cset must be in cgroup
#		#	2. No extra columns are allowed to be in cgroup
#		#	3. The lengths of all columns must be the same
#		l = None
#		for (pos, col) in enumerate(self.column_data):
#			name = self.column_map[pos]
#			if name not in other:
#				raise Exception('Column "%s" not found in the set of rows to be appended.' % name)
#			if l == None:
#				l = len(other[name])
#			if l != len(other[name]):
#				raise Exception('Length of column "%s" different from the others.' % name)
#
#		if self.ncols() != other.ncols():
#			c1 = self.keys()
#			c2 = other.keys()
#			print sorted(c1)
#			print sorted(c2)
#			raise Exception('Extra columns found in the set of rows to be appended.')
#
#		# Append data
#		for (pos, col) in enumerate(self.column_data):
#			name = self.column_map[pos]
#			self.column_data[pos] = np.append(col, other[name])
#
#		return self

	def subset(self, key):
		"""
		Internal: use cg[key] for forward compatibility.
		"""
		# Return a subset or a column

		# Return a single column
		if type(key) is str:
			return self.column_data[self.column_map[key]]

		# Return a subset cgroup
		return ColGroup((   (name, self.column_data[self.column_map[name]]) for name in key ), info=self.info)

	def nrows(self):
		"""
		Internal: return the number of rows in the ColGroup.
		
		Use len(cg) for forward compatibility.
		"""
		return 0 if len(self.column_data) == 0 else len(self.column_data[0])

	def ncols(self):
		"""
		Return the number of columns in the ColGroup
		"""
		return len(self.column_data)

	def __len__(self):
		return self.nrows()

	def __iter__(self):
		"""
		Iterate through rows of the ColGroup
		
		NOTE: The implementation of RowIterator is horrible at the
		moment, so iterating row-by-row should be avoided if
		possible.
		"""
		return RowIter(self)

	def as_columns(self):
		"""
		Return an iterable yielding columns (the ndarrays)
		
		Allows one to write code like:
		>>> x, y, z = cg.as_columns()
		
		TODO: Add a 'cols' parameter, to restrict the list of
		columns that gets returned so that something like:
		
		>>> x, y = cg.as_columns(['x','y'])
		
		becomes possible.
		"""
		return iter(self.column_data)
	
	def as_ndarray(self):
		"""
		Return a structured ndarray with columns of this ColGroup
		"""
		rows = np.empty(self.nrows(), dtype=self.dtype)
		for name in self.keys():
			rows[name] = self[name]
		return rows

	def __str__(self):
		"""
		Prints the head/tail of the columns in the group.
		"""
		if len(self.column_data) == 0:
			return ''

		ret = ''
		for row in self[0:min(len(self.column_data[0]), 10)]:
			ret = ret + str(row) + '\n'

		return ret[:-1] if ret != '' else ''

	def sort(self, cols=()):
		"""
		Sort the cgroup by one or more (or all) columns
		
		The names of the columns are to be given in the cols
		iterable.
		"""
		if len(cols) == 0:
			cols = self.keys()

		sort_cols = [ self[name] for name in reversed(cols) ]
		idx = np.lexsort(sort_cols)

		for col in self.column_data:
			col[:] = col[idx]

	def __eq__(self, y):
		"""
		Emulate ndarray per-element equality comparison

		Works for comparing cgroup-to-cgroup and cgroup-to-structured ndarray
		Comparison to anything else returns False.
		"""
		if not isinstance(y, ColGroup) and not isinstance(y, np.ndarray):
			return False

		assert self.dtype == y.dtype, str(self.dtype) + str(y.dtype)
		assert len(self) == len(y)

		res = np.ones(len(self), dtype=bool)
		for name in self.keys():
			# Support multi-dimensional subfields, by flattening them
			# successively over every axis
			tmp = self[name] == y[name]
			for _ in xrange(len(tmp.shape)-1):
				tmp = np.all(tmp, axis=-1)

			res &= tmp

		return res

	def copy(self, expr=None):
		"""
		Return a copy of this object, potentially keeping only some
		rows.

		Copies the info object as well.

		If expr is not None, keeps only the rows for which expr
		evaulates to True.
		"""
		ret = ColGroup(info=self.info)

		if expr is None:
			for name in self.keys():
				ret.add_column(name, self[name].copy())
		else:
			for name in self.keys():
				ret.add_column(name, self[name][expr])
		return ret

	def cut(self, expr):
		"""
		Remove rows for which expr evaluates to False
		
		Returns self.

		Convenient for interactive work.
		"""

		self.column_data = [ col[expr] for col in self.column_data ]

		return self


def fromiter(it, dtype=None, blocks=False):
	"""
	Load a ColGroup from an iterable.
	"""
	assert blocks == True, "blocks==False not implemented yet."

	buf = None
	for rows in it:
		if buf is None:
			# Just copy the first one
			buf = rows.copy()
			at  = len(buf)
		else:
			at2 = at + len(rows)
			if at2 > len(buf):
				# Next higher power of two
				newsize = 1 << int(np.ceil(np.log2(at2)))
				buf.resize(newsize)

			# append
			buf[at:at+len(rows)] = rows
			at = at + len(rows)

	# Truncate the buffer to output size
	if buf is not None:
		buf.resize(at)
	else:
		buf = ColGroup(dtype=dtype, size=0)

	return buf

def count_unique(v):
	""" Given a ndarray v, return a tuple k, ct
	    where k is an array of unique elements of v
	    and ct is the number of occurrences of each
	    element in v (frequency)
	"""
	v = np.sort(v.flatten())

	# Unique values
	keep = np.concatenate(([True], v[1:] != v[:-1]))
	u = v[keep]

	# Their count
	ct = np.searchsorted(v, u, side='right') - np.searchsorted(v, u, side='left')

	return u, ct

def partitioned_fromiter(it, keycol, maxrows, dtype=None, blocks=False):
	assert blocks == True, "blocks==False not implemented yet."

	try:
		keys = []
		buf = None
		fp = None
		for rows in it:
#			print "XXX:", rows.dtype, len(rows)
			if buf is None:
				# Just copy the first one
				buf = rows.copy()
				if dtype is None:
					dtype = rows.dtype
				at  = len(buf)
			else:
				at2 = at + len(rows)
				if at2 > len(buf):
					# Next higher power of two
					newsize = 1 << int(np.ceil(np.log2(at2)))
					buf.resize(newsize)

				# append
				buf[at:at+len(rows)] = rows
				at = at + len(rows)
				
				# deallocate rows
				rows = None

			# If buf has exceeded the maximum partition size,
			# back it to disk and continue with an empty one
			if at > maxrows:
				buf.resize(at)
#				print "Writing block, len=", len(buf)
				if fp is None:
					fp = tempfile.NamedTemporaryFile(mode='r+b', prefix='fromiter-', dir=os.getenv('LSD_TEMPDIR'), suffix='.pkl', delete=True)
				keys.append(buf[keycol].copy())
				cPickle.dump(buf, fp, -1)
				buf = None

		# Truncate the buffer to output size
		if buf is not None:
			buf.resize(at)
		else:
			buf = ColGroup(dtype=dtype, size=0)

		# If there was no backing to disk, just return
		if fp is None:
			yield buf
			return

		# Store the last buf to disk
#		print "Writing last block, len=", len(buf)
#		print "LAST:", buf.dtype, len(buf)
		keys.append(buf[keycol].copy())
		cPickle.dump(buf, fp, -1)
#		print "nblocks = ", len(keys)

		# yield (horizontally) partitioned pieces
		nsaved = len(keys)
		keys = np.concatenate(keys)
		ukeys, counts = count_unique(keys)
#		ccounts = np.concatenate(([0], np.cumsum(counts)))
		ccounts = np.cumsum(counts)
#		print len(ukeys), ukeys[:10]
#		print len(counts), counts[:10]
#		print len(ccounts), ccounts[:10]
		i0 = 0
		c0 = 0
		while i0 < len(ccounts):
			i1 = np.searchsorted(ccounts, c0 + maxrows, side='right')
			if i1 == i0:
				i1 += 1
			c1 = ccounts[i1-1]
			keep_keys = ukeys[i0:i1]
			size = c1 - c0
#			print >>sys.stderr, "CUT:", i0, i1, size, keep_keys
			i0 = i1
			c0 = c1

			# Stream through the stored result and select
			# out only the given block of keys
			fp.seek(0)
			buf = None
			for _ in xrange(nsaved):
				rows = cPickle.load(fp)
				keep = np.in1d(rows[keycol], keep_keys)
				if not np.any(keep):
					continue

				rows = rows[keep]
				if buf is None:
					buf = ColGroup(dtype=rows.dtype, size=size)
					at = 0
				buf[at:at+len(rows)] = rows
				at += len(rows)
			assert at == len(buf)

			print >>sys.stderr, "YIELD", len(buf)
			yield buf
	finally:
		if fp is not None:
			fp.close()

############################################################
# Unit tests

class Test_ColGroup__getitem__:
	def setUp(self):
		# Create a numpy structured array from which ColGroup instances will be made
		global random, sys, itertools
		import numpy.random as random
		import sys, itertools

		random.seed(42)
		dtype = [ ('int4', 'i4'), ('float8', 'f8'), ('float4', 'f4'),
			  ('4float4', '4f4'), ('a10', 'a10'), ('cube', (np.float32, (4, 4)))
			]
		self.a = np.empty(10000, dtype=dtype)
		for name, dtype in dtype:
			self.a[name].flat = np.random.random(self.a[name].size)*self.a[name].size

	def test_getitem_column_subset(self):
		""" __getitem__: subset of columns """
		cg = ColGroup(self.a)

		random.seed(42)
		for names in itertools.combinations(self.a.dtype.names, 3):
			names = list(names)
			random.shuffle(names)

			#print >>sys.stderr, names
			cg2 = cg[names]
			for name in names:
				assert np.all(cg2[name] == self.a[name])

	def test_getitem_column_name(self):
		""" __getitem__: by column name """
		cg = ColGroup(self.a)

		random.seed(78)
		names = list(self.a.dtype.names)
		random.shuffle(names)

		for name in names:
			#print >>sys.stderr, name
			col = cg[name]
			assert np.all(col == self.a[name])

	def test_getitem_single_row(self):
		""" __getitem__: get single row """
		cg = ColGroup(self.a)

		random.seed(79)
		ii = np.append(random.random_integers(0, self.a.size-1, 10), [0, self.a.size-1])

		for i in ii:
			row   = cg[i]
			rownp = self.a[i]

			# Test column-by-column
			for name in self.a.dtype.names:
				assert np.all(row[name] == rownp[name])

			# Test via __eq__ operator
			assert np.all(row == rownp)

	def _test_getitem_by(self, idx):
		# Aux. func for test_getitem_by_index and _by_boolean
		cg = ColGroup(self.a)

		cg2 = cg[idx]
		a2 = self.a[idx]

		# Test column-by-column
		for name in self.a.dtype.names:
			#print >>sys.stderr, idx
			assert np.all(cg2[name] == a2[name])

		# Test via __eq__ operator
		assert np.all(cg2 == a2)		

	def test_getitem_by_index(self):
		""" __getitem__: by integer index """
		random.seed(55)
		idx = random.random_integers(0, self.a.size-1, self.a.size // 2)

		self._test_getitem_by(idx)

	def test_getitem_by_boolean(self):
		""" __getitem__: by boolean index """
		random.seed(65)
		idx = random.random_integers(0, self.a.size-1, self.a.size) > (self.a.size // 2)
		assert idx.dtype == bool

		self._test_getitem_by(idx)

class Test_fromiter:
	"""fromiter: comprehensive tests"""
	def setUp(self):
		global izip
		from itertools import izip

	def test_empty(self):
		"""fromiter: empty iterator"""
		# No element
		cg1 = fromiter([], blocks=True)
		assert np.all(cg1 == ColGroup())
		assert len(cg1) == 0

	def test_single(self):
		"""fromiter: single element"""
		cg1 = ColGroup()
		cg1.f1 = np.arange(10)
		cg1.f2 = np.arange(10)
		cg2 = fromiter([cg1], blocks=True)
		assert np.all(cg1 == cg2)

	def _mkarray(self, begin, end):
		cg = ColGroup()
		cg.f1 = np.arange(begin, end)
		cg.f2 = np.arange(begin, end, dtype='f4')
		return cg

	def _mkarray_keyed(self, begin, end, mod):
		cg = ColGroup()
		cg.k = np.arange(begin, end) % mod
		cg.f1 = np.arange(begin, end)
		cg.f2 = np.arange(begin, end, dtype='f4')
		return cg

	def test_general(self):
		"""fromiter: multiple elements"""
		# More than one block of varying sizes
		s = [20, 2, 4, 570, 13, 44, 56, 88, 9999, 1003000, 0, 1]
		cg2 = fromiter(( self._mkarray(end-len, end) for (len, end) in izip(s, np.cumsum(s)) ), blocks=True)
		cg1 = self._mkarray(0, sum(s))
		assert np.all(cg1 == cg2)

	def test_special_pow2(self):
		"""fromiter: powers of two sizes"""
		# Special sizes
		s = [0, 2, 2, 4, 8, 16, 32, 32, 32, 1]
		cg2 = fromiter(( self._mkarray(end-len, end) for (len, end) in izip(s, np.cumsum(s)) ), blocks=True)
		cg1 = self._mkarray(0, sum(s))
		assert np.all(cg1 == cg2)

	def test_fuzz(self):
		"""fromiter: fuzzing"""
		# Fuzzing
		np.random.seed(99)
		for _ in xrange(100):
			s = np.random.random_integers(0, 10000, 20)
			cg2 = fromiter(( self._mkarray(end-len, end) for (len, end) in izip(s, np.cumsum(s)) ), blocks=True)
			cg1 = self._mkarray(0, sum(s))
			assert np.all(cg1 == cg2)

	def test_partitioned_fromiter(self):
		"""partitioned_fromiter"""
		s    = [20, 2, 4, 570, 13, 44, 56, 88, 9999, 10030, 0, 1]
		mods = [2,  5, 1, 30,  2,  10, 3,  22, 100,  110,   2, 1]
		slist = [ self._mkarray_keyed(end-llen, end, mod) for (llen, end, mod) in izip(s, np.cumsum(s), mods) ] 
		cg1 = ColGroup(np.concatenate([v.as_ndarray() for v in slist]))

		# Quick (and incomplete) test for count_unique		
		k, ct = count_unique(cg1.k)
		assert np.all(k == np.unique(cg1.k))
		assert sum(ct) == len(cg1.k)

		cg1.sort('k')

		# Try for a few different maxrows
		for maxrows in [10000, 1000, 100, 10, 1]:
			print "maxrows=%d" % maxrows
			rlist = []
			for rows in partitioned_fromiter(slist, 'k', maxrows, blocks=True):
				assert len(rows) <= maxrows or np.all(rows['k'] == rows['k'][0])
				print len(rows),
				rlist.append(rows)
			print ''
			cg2 = fromiter(rlist, blocks=True)
			cg2.sort('k')

			assert np.all(cg1 == cg2)

if __name__ == "__main__":
	test_fromiter()
	exit()

	# Multi-dimensional arrays
	ra   = np.arange(10, dtype=np.dtype('f8'))
	dec  = np.arange(40, dtype='i4').reshape((10, 4))
	blob = np.array([ [ 'blob' + str(j) + '_' + str(i) for i in xrange(3) ] for j in xrange(10)], dtype=np.object_)
	tbl0  = ColGroup([("ra", ra), ("dec", dec), ("blob", blob)])

	print "Single tbl0 row:", tbl0[3], '\n'
	tbl0['blob'][3,2] = 'new thing'
	print "Modified tbl0 row:", tbl0[3], '\n'
	print "tbl0 slice:", tbl0[3:6], '\n'

	tbl  = tbl0[('ra', 'dec')]
	print tbl[3:5], '\n'
	tbl[3] = (42, (99, 88, 77, 66))
	print tbl[3], '\n'
	tbl[6:8] = [ (77, (99, 88, 77, 66)), (77, (94, 84, 74, 64)) ]
	print tbl, '\n'
	tbl['dec'][6:8] = np.zeros((2, 4))
	print tbl0, '\n'
	exit()

	# Simple arrays
	ra   = np.arange(1000, dtype=np.dtype('f8'))
	dec  = np.arange(1000, dtype=np.dtype('f8'))
	id   = np.arange(1000, dtype=np.dtype('i8'))
	blob = np.array([ 'blob' + str(i) for i in xrange(1000) ], dtype=np.object_)
	tbl  = ColGroup([("ra", ra), ("dec", dec), ("id", id), ("blob", blob)])

	ra2   = np.arange(1000, dtype=np.dtype('f8'))
	dec2  = np.arange(1000, dtype=np.dtype('f8'))
	id2   = np.arange(1000, dtype=np.dtype('i8'))
	blob2 = np.array([ 'blob' + str(i) for i in xrange(1000) ], dtype=np.object_)
	tbl2  = ColGroup([("ra", ra2), ("dec", dec2), ("id", id2), ("blob", blob2)])

	tbl2['ra'] += 22
	tbl2['dec'] += 33
	tbl2['id'] += 11
	tbl2['blob'] += 'aaa'

	print "Selecting a single row"
	row = tbl[1]
	print "ROW:", row, type(row)
	print ""

	print "Changing an entry in a single row"
	print "Before:", tbl[5]
	tbl[5] = row
	print "After:", tbl[5], '\n'

	print tbl, '\n'
	print tbl2, '\n'
	tbl[2:4] = tbl2[2:4]
	print tbl, '\n'
	print tbl2, '\n'

	#t2  = tbl[tbl["id"] == 5]
	#t2 = tbl[("ra", "dec", "blob")]
	#print t2
