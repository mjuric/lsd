#!/usr/bin/env python

import numpy as np

class RowIter:
	""" Iterator to permit iteration over Table by row
	"""
	def __init__(self, table):
		self.table = table
		self.at = 0
		self.end = table.nrows()

		if self.at < self.end:
			tmp = np.empty(1, table.dtype)
			self.row = tmp[0]

			self.column_data  = table.column_data
			self.column_map = table.keys()

	# Iterator protocol methods
	def __iter__(self):
		return self

	def next(self):
		if self.at >= self.end:
			raise StopIteration()

		for i, name in enumerate(self.column_map):
			self.row[name] = self.column_data[i][self.at]
		self.at = self.at + 1

		return self.row

class Table(object):
	column_map = None	# Map from name->pos and pos->name
	column_data = None	# A list of columns (individual numpy arrays)

	##
	## dict-like interface implementation
	##
	def keys(self):
		# Return the list of columns, in order of appearance
		return [ self.column_map[pos] for (pos, col) in enumerate(self.column_data) ]

	def items(self):
		# Return a list of (name, column) tuples
		items = [ (self.column_map[pos], self.column_data[pos]) for pos in xrange(len(self.column_data)) ]
		return items

	def __getitem__(self, key):
		# return a subtable if column names (or a tuple or
		# a list of them) were given
		if isinstance(key, str) or isinstance(key, list) or isinstance(key, tuple):
			return self.subtable(key)

		# Return a slice of the table
		cols = Table()
		for (pos, col) in enumerate(self.column_data):
			name = self.column_map[pos]
			rcol = col[key]
			if type(rcol) != np.ndarray:			# Ensure we always return a numpy array
				rcol = np.array([rcol], col.dtype)
			cols.add_column(name, rcol)
		return cols

	def __setitem__(self, key, value):
		# Append a column to the table, or replace data
		# in an existing column or entire table slice
		if isinstance(key, str):
			if key in self.column_map:
				self.column_data[self.column_map[key]][:] = value
			else:
				self.add_column(key, value)
		elif isinstance(key, slice):
			assert isinstance(value, Table)
			assert value.keys() == self.keys()

			for (pos, col) in enumerate(self.column_data):
				self.column_data[pos][key] = value.column_data[pos]
		else:
			print type(key)
			raise TypeError()

	def __contains__(self, column):
		# Test if a column exists in the table
		return column in self.column_map

	#############

	@property
	def dtype(self):
		# Return the dtype this column set would have if it was a numpy structured array
		return np.dtype([ (self.column_map[pos], col.dtype.str) for (pos, col) in enumerate(self.column_data) ])

	def __init__(self, cols=[]):
		self.column_map = dict()
		self.column_data = []
		for (name, col) in cols:
			self.add_column(name, col)

	def add_column(self, name, col):
		# sanity check: require numpy arrays
		assert isinstance(col, np.ndarray)
		assert self.ncols() == 0 or (len(col) == self.nrows())
		assert name not in self.column_map

		# Add a column to the end
		pos = len(self.column_data)
		self.column_map[name] = pos
		self.column_map[pos] = name
		self.column_data.append(col)

	def resize(self, size):
		for (pos, col) in enumerate(self.column_data):
			self.column_data[pos] = np.resize(self.column_data[pos], size)

#	def append_rows(self, other):
#		# Sanity checks:
#		#	1. All columns from the cset must be in table
#		#	2. No extra columns are allowed to be in table
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

	def subtable(self, key):
		# Return a subtable or a column

		# Return a single column
		if type(key) is str:
			return self.column_data[self.column_map[key]]

		# Return a subtable
		cols = Table()
		for name in key:
			cols.add_column( name, self.column_data[self.column_map[name]] )
		return cols

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

	def as_ndarray(self):
		rows = np.empty(self.nrows(), dtype=self.dtype)
		for name in self.keys():
			rows[name] = self[name]
		return rows

	def __str__(self):
		# Print head/tail of the table
		if len(self.column_data) == 0:
			return ''

		ret = ''
		for row in self[0:min(len(self.column_data[0]), 10)]:
			ret = ret + str(row) + '\n'

		return ret

	def sort(self, cols=()):
		""" Sort the table by one or more (or all) columns
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
			Works for comparing table-to-table and table-to-structured ndarray

			Everything else returns False
		"""
		if not isinstance(y, Table) and not isinstance(y, np.ndarray):
			return False

		assert self.dtype == y.dtype, str(self.dtype) + str(y.dtype)
		assert len(self) == len(y)

		res = np.ones(len(self), dtype=bool)
		for name in self.keys():
			res &= (self[name] == y[name])

		return res

if __name__ == "__main__":
	ra   = np.arange(1000, dtype=np.dtype('f8'))
	dec  = np.arange(1000, dtype=np.dtype('f8'))
	id   = np.arange(1000, dtype=np.dtype('i8'))
	blob = np.array([ 'blob' + str(i) for i in xrange(1000) ], dtype=np.object_)
	tbl  = Table([("ra", ra), ("dec", dec), ("id", id), ("blob", blob)])

	ra2   = np.arange(1000, dtype=np.dtype('f8'))
	dec2  = np.arange(1000, dtype=np.dtype('f8'))
	id2   = np.arange(1000, dtype=np.dtype('i8'))
	blob2 = np.array([ 'blob' + str(i) for i in xrange(1000) ], dtype=np.object_)
	tbl2  = Table([("ra", ra2), ("dec", dec2), ("id", id2), ("blob", blob2)])

	tbl2['ra'] += 22
	tbl2['dec'] += 33
	tbl2['id'] += 11
	tbl2['blob'] += 'aaa'

	print tbl2
	print tbl
	tbl[2:4] = tbl2[2:4]
	print tbl
	print tbl2

	#t2  = tbl[tbl["id"] == 5]
	#t2 = tbl[("ra", "dec", "blob")]
	#print t2
