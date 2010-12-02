#!/usr/bin/env python

import numpy as np
from utils import full_dtype

def make_record(dtype):
	# Construct a numpy record corresponding to a row of this table
	# There has to be a better way to construct a numpy.void of a given dtype, but I couldn't figure it out...
	tmp = np.zeros(1, dtype)
	return tmp[0].copy()

class RowIter:
	""" Iterator to permit iteration over ColGroup by row
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

class ColGroup(object):
	column_map = None	# Map from name->pos and pos->name
	column_data = None	# A list of columns (individual numpy arrays)

	row = None		# A np.void instance with the correct dtype for a row in this cgroup

	##
	## dict-like interface implementation
	##
	def keys(self):
		# Return the list of columns, in order of appearance
		return [ self.column_map[pos] for (pos, _) in enumerate(self.column_data) ]

	def items(self):
		# Return a list of (name, column) tuples
		items = [ (self.column_map[pos], self.column_data[pos]) for pos in xrange(len(self.column_data)) ]
		return items

	def __getitem__(self, key):
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
		ret = [ (self.column_map[pos], col[key])   for (pos, col) in enumerate(self.column_data) ]

		# Return a ColGroup if the key was an instance of a slice or ndarray,
		# and numpy.void structure otherwise
		if isinstance(key, slice) or isinstance(key, np.ndarray):
			return ColGroup(ret)
		else:
			#row = self.row.copy()
			row = make_record(self.dtype)
			for name, val in ret:
				# Special care for vector columns
				if isinstance(val, np.ndarray):
					row[name][:] = val
				else:
					row[name] = val
			return row

	def drop_column(self, key):
		pos  = self.column_map[key] if     isinstance(key, str) else key
		cols = self.items()

		del self.column_data[pos]
		del cols[pos]

		self.column_map = dict(( (colname, pos)    for (pos, (colname, _)) in enumerate(cols) ))
		self.column_map.update(( (pos, colname)    for (pos, (colname, _)) in enumerate(cols) ))

	def __setitem__(self, key, value):
		# Append a column to the cgroup, or replace data
		# in an existing column or entire cgroup slice
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
		# Test if a column exists in the cgroup
		return column in self.column_map

	#############

	@property
	def dtype(self):
		# Return the dtype this column set would have if it was a numpy structured array
		return np.dtype([ (self.column_map[pos], full_dtype(col)) for (pos, col) in enumerate(self.column_data) ])

	def __init__(self, cols=[], dtype=None, size=0):
		self.column_map = dict()
		self.column_data = []

		if dtype is not None:
			# construct cols from structured array dtype
			assert cols is None or len(cols) == 0
			template = np.zeros(size, dtype=dtype)
			cols = [ (name, template[name]) for name in template.dtype.names ]

		# Detect dict() and ColGroup()s
		if getattr(cols, 'items', None) is not None:
			cols = cols.items()

		for (name, col) in cols:
			self.add_column(name, col)

#		self._mk_row()

	def add_columns(self, cols):
		for name, col in cols:
			self.add_column(name, col)

	def add_column(self, name, col, dtype=None):
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
		for (pos, _) in enumerate(self.column_data):
			col = self.column_data[pos]
			self.column_data[pos] = np.resize(col, (size,) + col.shape[1:])

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
		# Return a subset or a column

		# Return a single column
		if type(key) is str:
			return self.column_data[self.column_map[key]]

		# Return a subset cgroup
		return ColGroup((   (name, self.column_data[self.column_map[name]]) for name in key ))

	def nrows(self):
		return 0 if len(self.column_data) == 0 else len(self.column_data[0])

	def ncols(self):
		return len(self.column_data)

	def __len__(self):
		return self.nrows()

	def __iter__(self):
		# Iterate through the list of rows
		return RowIter(self)

	def as_columns(self):
		return iter(self.column_data)
	
	def column_at(self, idx):
		return self.column_data[idx]

	def as_ndarray(self):
		rows = np.empty(self.nrows(), dtype=self.dtype)
		for name in self.keys():
			rows[name] = self[name]
		return rows

	def __str__(self):
		# Print head/tail of the cgroup
		if len(self.column_data) == 0:
			return ''

		ret = ''
		for row in self[0:min(len(self.column_data[0]), 10)]:
			ret = ret + str(row) + '\n'

		return ret[:-1] if ret != '' else ''

	def sort(self, cols=()):
		""" Sort the cgroup by one or more (or all) columns
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

			Everything else returns False
		"""
		if not isinstance(y, ColGroup) and not isinstance(y, np.ndarray):
			return False

		assert self.dtype == y.dtype, str(self.dtype) + str(y.dtype)
		assert len(self) == len(y)

		res = np.ones(len(self), dtype=bool)
		for name in self.keys():
			res &= (self[name] == y[name])

		return res

if __name__ == "__main__":
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
