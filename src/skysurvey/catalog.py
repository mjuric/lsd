#!/usr/bin/env python

import query_parser as qp
from collections import defaultdict
from contextlib import contextmanager
import subprocess
import pickle
import tables
import numpy.random as ran
import numpy as np
import pyfits
from math import *
import bhpix
import Polygon.Shapes
import itertools as it
from itertools import izip, imap
from multiprocessing import Pool
import multiprocessing as mp
import time
import sys
import pool2
import os, errno, glob
import json
import fcntl
import Polygon.IO
import utils
import footprint
from table import Table
from utils import astype, gc_dist, unpack_callable, full_dtype, is_scalar_of_type, isiterable
from intervalset import intervalset
from StringIO import StringIO
from pixelization import Pixelization
from native import table_join

def vecmd5(x):
	import hashlib
	l = np.empty(len(x), dtype='a32')
	for i in xrange(len(x)):
		l[i] = hashlib.md5(x[i]).hexdigest()
	return l

def veclen(x):
	l = np.empty(len(x), dtype=int)
	for i in xrange(len(x)):
		l[i] = len(x[i])
	return l

# Special return type used in _mapper() and Catalog.map_reduce
# to denote that the returned value should not be yielded to
# the user
# Impl. note: The class is intentionally derived from list, and
#             the value is intentionally [], for it to
#             be compatible with the map/reduce mode (i.e.,
#             an empty list will be ignored when constructing
#             the list of values to reduce)
class EmptySpecial(list):
	pass
Empty = EmptySpecial()	# Marker for mapreduce

# Constants with special meaning
Automatic = None
Default = None
All = []

class Catalog:
	""" A spatially and temporally partitioned object catalog.
	
	    The usual workhorses are Catalog.fetch, Catalog.iterate
	    and Catalog.map_reduce methods.
	"""
	path = '.'
	pix  = Pixelization(level=6, t0=54335, dt=1)
				# t0: default starting epoch (== 2pm HST, Aug 22 2007 (night of GPC1 first light))
				# t1: default temporal resolution (in days)
	__nrows = 0

	NULL = 0		# The value for NULL in JOINed rows that had no matches

	tables  = None		# Tables in catalog ( dict of lists of (tablename, schema, primary_key) tuples)
	hidden_tables = None	# Tables in catalog that do not participate in direct joins/fetches (e.g., xmatch tables)
	primary_table = None	# Primary table of this catalog ( the one holding the IDs and spatial/temporal keys)

	joined_catalogs = None	# External, cross-matched catalogs (see add_joined_catalog for schema of this dict)

	def all_columns(self):
		# Return the list of all columns in all tables of the catalog
		# Return the primary table's columns first, followed by other tables in alphabetical order
		cols = list(self.tables[self.primary_table]['columns'])	# Note: wrapped in list() to get a copy
		for name in sorted(self.tables.keys()):
			if name == self.primary_table:
				continue
			cols += self.tables[name]['columns']
		# Extract just the names
		return [ name for name, _ in cols ]

	### File name/path related methods
	def get_table_data_path(self, table):
		""" Allow individual tables to override where they're placed
		    This comes in handy for direct JOINs.
		"""
		schema = self._get_schema(table)
		return schema.get('path', '%s/data' % (self.path))

	def _tablet_filename(self, table):
		""" The filename of a tablet in a cell """
		return '%s.%s.h5' % (self.name, table)

	def _cell_prefix(self, cell_id):
		return '%s/%s/%s' % (self.get_table_data_path(self.primary_table), self.pix.path_to_cell(cell_id), self.name)

	def _tablet_file(self, cell_id, table):
		return '%s/%s/%s' % (self.get_table_data_path(table), self.pix.path_to_cell(cell_id), self._tablet_filename(table))

	def tablet_exists(self, cell_id, table=None):
		""" Return True if the given tablet exists in cell_id """
		if table is None:
			table = self.primary_table

		assert (table in self.tables) or (table in self.hidden_tables)

		fn = self._tablet_file(cell_id, table)
		return os.access(fn, os.R_OK)

	def static_if_no_temporal(self, cell_id, table):
		""" Try to locate tablet 'table' in cell_id. If there's
		    no such tablet, return a corresponding static sky
		    cell_id
		"""
		if not self.pix.is_temporal_cell(cell_id):
			return cell_id

		if self.tablet_exists(cell_id, table):
			##print "Temporal cell found!", self._cell_prefix(cell_id)
			return cell_id

		# return corresponding static-sky cell
		cell_id = self.pix.static_cell_for_cell(cell_id)
		#print "Reverting to static sky", self._cell_prefix(cell_id)
		return cell_id

	def get_cells(self, bounds, return_bounds=False):
		""" Return a list of cells
		"""
		data_path = self.get_table_data_path(self.primary_table)
		pattern   = self._tablet_filename(self.primary_table)

		return self.pix.get_cells(data_path, pattern, bounds, return_bounds=return_bounds)

	def is_cell_local(self, cell_id):
		""" Returns True if the cell is reachable from the
		    current machine. A placeholder for if/when I decide
		    to make this into a true distributed database.
		"""
		return True

	#############

	def _load_dbinfo(self):
		data = json.loads(file(self.path + '/dbinfo.json').read())

		self.name = data["name"]
		self.__nrows = data["nrows"]

		######################
		# Backwards compatibility
		level, t0, dt = data["level"], data["t0"], data["dt"]
		self.pix = Pixelization(level, t0, dt)

		#################################### Remove at some point
		# Backwards compatibility
		if "columns" in data:
			data["tables"] = \
			{
				"catalog":
				{
					"columns":   data["columns"],
					"primary_key": "id",
					"spatial_keys": ("ra", "dec"),
					"cached_flag": "cached",
					"exposure_key": "exp_id",
					"blobs": dict()
				}
			}
			data["primary_table"] = 'catalog'
		if "hidden_tables" not in data: data["hidden_tables"] = {}
		if "joined_catalogs" not in data: data["joined_catalogs"] = {}
		###############################

		# Load table definitions
		self.tables = data["tables"]
		self.hidden_tables = data["hidden_tables"]
		self.primary_table = data["primary_table"]
		self.joined_catalogs = data["joined_catalogs"]

		# Postprocessing: fix cases where JSON restores arrays instead
		# of tuples, and tuples are required
		for _, schema in self.tables.iteritems():
			schema['columns'] = [ tuple(val) for val in schema['columns'] ]
		for _, schema in self.hidden_tables.iteritems():
			schema['columns'] = [ tuple(val) for val in schema['columns'] ]

	def _store_dbinfo(self):
		data = dict()
		data["level"], data["t0"], data["dt"] = self.pix.level, self.pix.t0, self.pix.dt
		data["nrows"] = self.__nrows
		data["tables"] = self.tables
		data["hidden_tables"] = self.hidden_tables
		data["primary_table"] = self.primary_table
		data["name"] = self.name
		data["joined_catalogs"] = self.joined_catalogs

		f = open(self.path + '/dbinfo.json', 'w')
		f.write(json.dumps(data, indent=4, sort_keys=True))
		f.close()

	def create_table(self, table, schema, ignore_if_exists=False, hidden=False):
		# Create a new table and set it as primary if it
		# has a primary_key
		if ((table in self.tables) or (table in self.hidden_tables)) and not ignore_if_exists:
			raise Exception('Trying to create a table that already exists!')

		tables = self.tables if not hidden else self.hidden_tables
		tables[table] = schema

		if 'primary_key' in schema:
			assert not hidden
			if 'spatial_keys' not in schema:
				raise Exception('Trying to create a primary table with no spatial keys!')
			if self.primary_table is not None:
				raise Exception('Trying to create a primary table while one already exists!')
			self.primary_table = table

		if 'blobs' in schema:
			cols = dict(schema['columns'])
			for blobcol in schema['blobs']:
				assert is_scalar_of_type(cols[blobcol], np.int64)

		self._store_dbinfo()

	### Cell locking routines
	def _lock_cell(self, cell_id, retries=-1):
		# create directory if needed
		fn = self._cell_prefix(cell_id) + '.lock'

		path = fn[:fn.rfind('/')];
		if not os.path.exists(path):
			utils.mkdir_p(path)

		utils.shell('/usr/bin/lockfile -1 -r%d "%s"' % (retries, fn) )
		return fn

	def _unlock_cell(self, lockfile):
		os.unlink(lockfile)

	#### Low level tablet creation/access routines. These employ no locking
	def _create_tablet(self, fn, table):
		# Create a tablet at a given path, for table 'table'
		assert os.access(fn, os.R_OK) == False

		# Find the schema of the requested table
		schema = self._get_schema(table)

		# Create the cell directory if it doesn't exist
		path = fn[:fn.rfind('/')];
		if not os.path.exists(path):
			utils.mkdir_p(path)

		# Create the tablet
		fp  = tables.openFile(fn, mode='w')
		fp.createTable('/main', 'table', np.dtype(schema["columns"]), expectedrows=20*1000*1000, createparents=True)

		if 'primary_key' in schema:
			seqname = '_seq_' + schema['primary_key']
			fp.createArray('/main', seqname, np.array([1], dtype=np.uint64))

		if 'blobs' in schema:
			for blobcol in schema['blobs']:
				fp.createVLArray('/main/blobs', blobcol, tables.ObjectAtom(), "BLOBs", createparents=True)
				fp.root.main.blobs.__getattr__(blobcol).append(None)	# ref=0 should be pointed to by no real element (equivalent to NULL pointer)

		return fp

	def _open_tablet(self, cell_id, table, mode='r'):
		""" Open a given tablet in read or write mode, autocreating
		    if necessary.
		    
		    No locking of any kind.
		"""
		fn = self._tablet_file(cell_id, table)

		if mode == 'r':
			fp = tables.openFile(fn)
		elif mode == 'w':
			if not os.path.isfile(fn):
				fp = self._create_tablet(fn, table)
			else:
				fp = tables.openFile(fn, mode='a')
		else:
			raise Exception("Mode must be one of 'r' or 'w'")

		return fp

	def _drop_tablet(self, cell_id, table):
		# Remove a tablet file. No locking of any kind.
		#
		if not self.tablet_exists(cell_id, table):
			return

		fn = self._tablet_file(cell_id, table)
		os.unlink(fn)

	def _append_tablet(self, cell_id, table, rows):
		# Append a set of rows to a tablet. No locking of any kind
		#
		fp  = self._open_tablet(cell_id, mode='w', table=table)

		fp.root.main.table.append(rows)

		fp.close()

	### Public methods
	def __init__(self, path, mode='r', name=None, level=Automatic, t0=Automatic, dt=Automatic):
		if mode == 'c':
			assert name is not None
			self.create_catalog(name, path, level, t0, dt)
		else:
			self.path = path
			self._load_dbinfo()

	def create_catalog(self, name, path, level, t0, dt):
		""" Create a new catalog and store its definition.
		"""
		self.path = path

		utils.mkdir_p(self.path)
		if os.path.isfile(self.path + '/dbinfo.json'):
			raise Exception("Creating a new catalog in '%s' would overwrite an existing one." % self.path)

		self.tables = {}
		self.hidden_tables = {}
		self.joined_catalogs = {}
		self.name = name

		if level == Automatic: level = self.pix.level
		if    t0 == Automatic: t0 = self.pix.t0
		if    dt == Automatic: dt = self.pix.dt
		self.pix = Pixelization(level, t0, dt)

		self._store_dbinfo()

	def update(self, table, keys, rows):
		raise Exception('Not implemented')

	def resolve_alias(self, colname):
		""" Return the real column name for special column
		    aliases.
		"""
		schema = self._get_schema(self.primary_table);

		if colname == '_ID'     and 'primary_key'  in schema: return schema['primary_key']
		if colname == '_LON'    and 'spatial_keys' in schema: return schema['spatial_keys'][0]
		if colname == '_LAT'    and 'spatial_keys' in schema: return schema['spatial_keys'][1]
		if colname == '_TIME'   and 'temporal_key' in schema: return schema['temporal_key']
		if colname == '_EXP'    and 'exposure_key' in schema: return schema['exposure_key']
		if colname == '_CACHED' and 'cached_flag'  in schema: return schema['cached_flag']

		return colname

	def append(self, cols):
		""" Insert a set of rows into a table in the database. Protects against
		    multiple writers simultaneously inserting into the same file.

		    If table being inserted into has spatial_keys, the rows being
		    inserted MUST contain the primary key column.

		    Return: array of primary keys of inserted rows
		"""

		# make a copy and perform some sanity checks
		cols = dict(cols)
		assert len(cols)
		n = None
		for _, col in cols.iteritems():
			if n is None: n = len(col)
			assert n == len(col), 'n=%d len(col)=%d' % (n, len(col))

		# Resolve aliases
		cols = dict(( (self.resolve_alias(name), col) for name, col in cols.iteritems()  ))

		# Locate cells into which we're going to store the results
		schema = self._get_schema(self.primary_table)
		raKey, decKey = schema["spatial_keys"]
		key           = schema["primary_key"]
		if key not in cols:	# if the primary key column has not been supplied, autoadd it
			cols[key] = np.empty(n, dtype=np.dtype(dict(schema['columns'])[key]))

		ra, dec = cols[raKey], cols[decKey]
		if "temporal_key" in schema:
			t = cols[schema["temporal_key"]]
		else:
			t = None
		cells     = self.cell_for_pos(ra, dec, t)
		cols[key] = self.id_from_pos(ra, dec, t)

		ntot = 0
		unique_cells = list(set(cells))
		while unique_cells:
			# Find a cell that is ready to be written to (that isn't locked
			# by another writer) and lock it
			for k in xrange(3600):
				try:
					i = k % len(unique_cells)
					cell_id = unique_cells[i]

					# Try to acquire a lock for the entire cell
					lock = self._lock_cell(cell_id, retries=0)

					unique_cells.pop(i)
					break
				except subprocess.CalledProcessError:
					#print err
					pass
			else:
				raise Exception('Appear to be stuck on a lock file!')

			# Extract rows belonging to this cell
			incell = cells == cell_id
			nrows = sum(incell)
			cols2 = {}
			for name, col in cols.iteritems():
				cols2[name] = col[incell]

			# Store them in their tablets
			for table, schema in self.tables.iteritems():
				fp    = self._open_tablet(cell_id, mode='w', table=table)
				t     = fp.root.main.table
				blobs = schema['blobs'] if 'blobs' in schema else dict()

				if table == self.primary_table:
					id_seq = fp.root.main.__getattr__('_seq_' + key)
					cols[key][incell] += np.arange(id_seq[0], id_seq[0] + nrows, dtype=np.uint64)
					cols2[key] = cols[key][incell]
					id_seq[0] += nrows

				# Construct a compatible numpy array, that will leave
				# unspecified columns set to zero
				rows = np.zeros(nrows, dtype=np.dtype(schema['columns']))
				for colname in rows.dtype.names:
					if colname not in cols2:
						continue
					if colname not in blobs:
						# Simple column
						rows[colname] = cols2[colname]
					else:
						# BLOB column - find unique objects, insert them
						# into the BLOB VLArray, and put the indices to these
						# into the actual table
						assert cols2[colname].dtype == np.object_
						uobjs, _, ito = np.unique(cols2[colname], return_index=True, return_inverse=True)	# Note: implicitly flattens multi-D input arrays
						ito = ito.reshape(rows[colname].shape)	# De-flatten the output indices

						# Offset indices
						barray = fp.root.main.blobs.__getattr__(colname)
						bsize = len(barray)
						ito = ito + bsize

						# Remap any None values to index 0 (where None is stored by fiat)
						# We use the fact that None will be sorted to the front of the unique sequence, if exists
						if len(uobjs) and uobjs[0] is None:
							##print "Remapping None", len((ito == bsize).nonzero()[0])
							uobjs = uobjs[1:]
							ito -= 1
							ito[ito == bsize-1] = 0

						rows[colname] = ito

						# Check we've correctly mapped everything
						uobjs2 = np.append(uobjs, [None])
						assert (uobjs2[np.where(rows[colname] != 0, rows[colname]-bsize, len(uobjs))] == cols2[colname]).all()

						for obj in uobjs:
							barray.append(obj)

				t.append(rows)
				fp.close()

			self._unlock_cell(lock)

			#print '[', nrows, ']'
			self.__nrows = self.__nrows + nrows
			ntot = ntot + nrows

		assert ntot == n, 'ntot != n, ntot=%d, n=%d, cell_id=%d' % (ntot, n, cell_id)
		assert len(np.unique1d(cols[key])) == n, 'len(np.unique1d(cols[key])) != n in cell %d' % cell_id

		return cols[key]

	def nrows(self):
		return self.__nrows

	def close(self):
		pass

	def __str__(self):
		""" Return some basic (human readable) information about the
		    catalog.
		"""
		i =     'Path:          %s\n' % self.path
		i = i + 'Partitioning:  level=%d\n' % (self.pix.level)
		i = i + '(t0, dt):      %f, %f \n' % (self.pix.t0, self.pix.dt)
		i = i + 'Objects:       %d\n' % (self.nrows())
		i = i + 'Tables:        %s' % str(self.tables.keys())
		i = i + 'Hidden tables: %s' % str(self.hidden_tables.keys())
		i = i + '\n'
		s = ''
		for table, schema in dict(self.tables, *self.hidden_tables).iteritems():
			s = s + '-'*31 + '\n'
			s = s + 'Table \'' + table + '\':\n'
			s = s + "%20s %10s\n" % ('Column', 'Type')
			s = s + '-'*31 + '\n'
			for col in schema["columns"]:
				s = s + "%20s %10s\n" % (col[0], col[1])
			s = s + '-'*31 + '\n'
		return i + s

	def _get_schema(self, table):
		if table in self.tables: return self.tables[table]
		return self.hidden_tables[table]

	def _smart_load_blobs(self, barray, refs):
		""" Load an ndarray of BLOBs from a set of refs refs,
		    taking into account not to instantiate duplicate
		    objects for the same BLOBs.
		    
		    The input array of refs must be one-dimensional.
		    The output is a 1D array of blobs, corresponding to the refs.
		"""
		##return np.ones(len(refs), dtype=object);
		assert len(refs.shape) == 1

		ui, _, idx = np.unique(refs, return_index=True, return_inverse=True)
		assert (ui >= 0).all()	# Negative refs are illegal. Index 0 means None

		objlist = barray[ui]
		if len(ui) == 1 and tables.__version__ == '2.2':
			# bug workaround -- PyTables 2.2 returns a scalar for length-1 arrays
			objlist = [ objlist ]

		# Note: using np.empty followed by [:] = ... (as opposed to
		#       np.array) ensures a 1D array will be created, even
		#       if objlist[0] is an array (in which case np.array
		#       misinterprets it as a request to create a 2D numpy
		#       array)
		blobs    = np.empty(len(objlist), dtype=object)
		blobs[:] = objlist
		blobs = blobs[idx]

		#print >> sys.stderr, 'Loaded %d unique objects for %d refs' % (len(objlist), len(idx))

		return blobs

	def fetch_blobs(self, cell_id, table, column, refs, include_cached=False):
		""" Fetch blobs from column 'column' in a tablet 'table'
		    of cell cell_id, given a vector of references 'refs'

		    If the cell_id has a temporal component, and there's no
		    tablet in that cell, a static sky cell corresponding
		    to it is tried next.
		"""
		# short-circuit if there's nothing to be loaded
		if len(refs) == 0:
			return np.empty(refs.shape, dtype=np.object_)

		# revert to static sky cell if cell_id is temporal, but there's no such tablet
		cell_id = self.static_if_no_temporal(cell_id, table)

		# Flatten refs; we'll deflatten the blobs in the end
		shape = refs.shape
		refs = refs.reshape(refs.size)

		# load the blobs arrays
		with self.get_cell(cell_id) as cell:
			with cell.open(table) as fp:
				b1 = fp.root.main.blobs.__getattr__(column)
				if include_cached and 'cached' in fp.root:
					# We have cached objects in 'cached' group -- read the blobs
					# from there as well. blob refs of cached objects are
					# negative.
					b2 = fp.root.cached.blobs.__getattr__(column)

					blobs = np.empty(len(refs), dtype=object)
					blobs[refs >= 0] = self._smart_load_blobs(b1,   refs[refs >= 0]),
					blobs[ refs < 0] = self._smart_load_blobs(b2,  -refs[ refs < 0]),
				else:
					blobs = self._smart_load_blobs(b1, refs)

		blobs = blobs.reshape(shape)
		return blobs

	def fetch_tablet(self, cell_id, table=None, include_cached=False):
		""" Load and return all rows from a given tablet in
		    a given cell_id.

		    If the cell_id has a temporal component, and there's no
		    tablet in that cell, a static sky cell corresponding
		    to it is tried next.
		"""
		if table is None:
			table = self.primary_table

		# revert to static sky cell if cell_id is temporal, but there's no such tablet
		cell_id = self.static_if_no_temporal(cell_id, table)

		if self.tablet_exists(cell_id, table):
			with self.get_cell(cell_id) as cell:
				with cell.open(table) as fp:
					rows = fp.root.main.table.read()
					if include_cached and 'cached' in fp.root:
						rows2 = fp.root.cached.table.read()
						rows = np.append(rows, rows2)
		else:
			schema = self._get_schema(table)
			rows = np.empty(0, dtype=np.dtype(schema['columns']))

		return rows

#	def query_cell(self, cell_id, query='*', include_cached=False):
#		""" Execute a query on a local cell.
#
#		    If the cell_id has a temporal component, and there are no
#		    tablets in that cell, a static sky cell corresponding
#		    to it will be tried.
#		"""
#		assert self.is_cell_local(cell_id)
#
#		return self.fetch(query, cell_id, include_cached=include_cached, progress_callback=pool2.progress_pass);
#
#	def fetch(self, query='*', foot=All, include_cached=False, testbounds=True, nworkers=None, progress_callback=None, filter=None):
#		""" Return a table (numpy structured array) of all rows within a
#		    given footprint. Calls 'filter' callable (if given) to filter
#		    the returned rows. Returns None if there are no rows to return.
#
#		    The 'filter' callable should expect a single argument, rows,
#		    being the set of rows (numpy structured array) to filter. It
#		    must return the set of filtered rows (also as numpy structured
#		    array). E.g., identity filter function would be:
#		    
#		    	def identity(rows):
#		    		return rows
#
#		    while a function filtering on column 'r' may look like:
#		    
#		    	def r_filter(rows):
#		    		return rows[rows['r'] < 21.5]
#		   
#		    The filter callable must be piclkeable. Extra arguments to
#		    filter may be given in 'filter_args'
#		"""
#
#		filter, filter_args = unpack_callable(filter)
#
#		ret = None
#		for rows in self.map_reduce(query=query, mapper=(_iterate_mapper, filter, filter_args), _pass_empty=True, foot=foot, testbounds=testbounds, include_cached=include_cached, nworkers=nworkers, progress_callback=progress_callback):
#			# ensure enough memory has been allocated (and do it
#			# intelligently if not)
#			if ret is None:
#				ret = rows
#				nret = 0
#
#			while len(ret) < nret + len(rows):
#				ret.resize(2*(len(ret)+1))
#				#print "Resizing to", len(ret)
#
#			# append
#			ret[nret:nret+len(rows)] = rows
#			nret = nret + len(rows)
#
#		ret.resize(nret)
#		#print "Resizing to", len(ret)
#		return ret
#
#	def iterate(self, query='*', foot=All, include_cached=False, testbounds=True, nworkers=None, progress_callback=None, filter=None, return_blocks=False):
#		""" Yield rows (either on a row-by-row basis if return_blocks==False
#		    or in chunks (numpy structured array)) within a
#		    given footprint. Calls 'filter' callable (if given) to filter
#		    the returned rows.
#
#		    See the documentation for Catalog.fetch for discussion of
#		    'filter' callable.
#		"""
#
#		filter, filter_args = unpack_callable(filter)
#
#		for rows in self.map_reduce(query, (_iterate_mapper, filter, filter_args), foot=foot, testbounds=testbounds, include_cached=include_cached, nworkers=nworkers, progress_callback=progress_callback):
#			if return_blocks:
#				yield rows
#			else:
#				for row in rows:
#					yield row

	class CellProxy:
		cat     = None
		cell_id = None
		mode    = None

		def __init__(self, cat, cell_id, mode):
			self.cat = cat
			self.cell_id = cell_id
			self.mode = mode

		@contextmanager
		def open(self, table=None):
			if table is None:
				table = self.cat.primary_table

			fp = self.cat._open_tablet(self.cell_id, mode=self.mode, table=table)

			yield fp

			fp.close()

	@contextmanager
	def get_cell(self, cell_id, mode='r', retries=-1):
		""" Open and return a proxy object for the given cell, that allows
		    one to open individual tablets stored there.

		    If mode is not 'r', the entire cell will be locked
		    for the duration of this context manager, and automatically
		    unlocked upon exit.
		"""
		lockfile = None if mode == 'r' else self._lock_cell(cell_id, retries=retries)

		yield Catalog.CellProxy(self, cell_id, mode=mode)

		if lockfile != None:
			self._unlock_cell(lockfile)

	def build_neighbor_cache(self, margin_x_arcsec=30):
		""" Cache the objects found within margin_x (arcsecs) of
		    each cell into neighboring cells as well, to support
		    efficient nearest-neighbor lookups.

		    This routine works in tandem with _cache_maker_mapper
		    and _cache_maker_reducer auxilliary routines.
		"""
		margin_x = sqrt(2.) / 180. * (margin_x_arcsec/3600.)

		# Find out which columns are our spatial keys
		schema = self._get_schema(self.primary_table)
		raKey, decKey = schema["spatial_keys"]
		query = "%s, %s" % (raKey, decKey)

		ntotal = 0
		ncells = 0
		for (cell_id, ncached) in self.map_reduce(query, (_cache_maker_mapper, margin_x), _cache_maker_reducer):
			ntotal = ntotal + ncached
			ncells = ncells + 1
			#print self._cell_prefix(cell_id), ": ", ncached, " cached objects"
		print "Total %d cached objects in %d cells" % (ntotal, ncells)

	def compute_summary_stats(self):
		""" Compute frequently used summary statistics and
		    store them into the dbinfo file. This should be called
		    to refresh the stats after insertions.
		"""
		from tasks import compute_counts
		self.__nrows = compute_counts(self)
		self._store_dbinfo()

	def get_spatial_keys(self):
		# Find out which columns are our spatial keys
		return self._get_schema(self.primary_table).get("spatial_keys", (None, None))

	def get_primary_key(self):
		# Find out which columns are our spatial keys
		return self._get_schema(self.primary_table)["primary_key"]

	def get_temporal_key(self):
		return self._get_schema(self.primary_table).get("temporal_key", None)

	def define_join(self, cat, table_from, table_to, column_from, column_to):
		# Schema:
		#	catalog_name: {
		#		'path': path,
		#		'table_from': table_name
		#		'table_to':   table_name
		#		'column_from':  'id1'
		#		'column_to':    'id2'
		#	}
		#
		# A simple case is where table_from = table_to, with only column_from and
		# column_to as their columns.
		self.joined_catalogs[cat.name] = \
		{
			'path':			cat.path,	# The path to the foreign catalog

			'table_from':		table_from,	# The table in _this_ catalog with the 'column_from' column
			'table_to':		table_to,	# The table in _this_ catalog with the 'column_to' column
			'column_from':		column_from,	# The primary keys of rows in this catalog, to be joined with the other catalog
			'column_to':		column_to	# The primary keys of corresponding rows in the foreign catalog
		}
		self._store_dbinfo()

###############################################################
# Aux functions implementing Catalog.iterate and Catalog.fetch
# functionallity
def _iterate_mapper(rows, filter, filter_args):
	if filter != None:
		rows = filter(rows, *filter_args)
	return rows

###############################################################
# Aux functions implementing Catalog.map_reduce functionallity
def _reducer(kw, reducer, cat, reducer_args):
	reducer.CATALOG = cat
	return reducer(kw[0], kw[1], *reducer_args)

def extract_columns(rows, cols=All):
	""" Given a structured array rows, extract and keep
	    only the list of columns given in cols.
	"""
	if cols == All:
		return rows

	rcols = [ (col, rows.dtype[col].str) for col in cols ]
	ret   = np.empty(len(rows), np.dtype(rcols))
	for col in cols: ret[col] = rows[col]

	return ret

def in_array(needles, haystack):
	""" Return a boolean array of len(needles) set to 
	    True for each needle that is found in the haystack.
	"""
	s = np.sort(haystack)
	i = np.searchsorted(s, needles)

	i[i == len(s)] = 0
	in_arr = s[i] == needles

	return in_arr

def tstart():
	return [ time.time() ]
	
def tick(s, t):
	tt = time.time()
	dt = tt - t[0]
	print >> sys.stderr, s, ":", dt
	t[0] = tt

def _fitskw_dumb(hdrs, kw):
	# Easy way
	res = []
	for ahdr in hdrs:
		hdr = pyfits.Header( txtfile=StringIO(ahdr) )
		res.append(hdr[kw])
	return res

def fits_quickparse(header):
	""" An ultra-simple FITS header parser. Does not support
	    CONTINUE statements, HIERARCH, or anything of the sort;
	    just plain vanilla:
	    	key = value / comment
	    one-liners. The upshot is that it's fast.

	    Assumes each 80-column line has a '\n' at the end
	"""
	res = {}
	for line in header.split('\n'):
		at = line.find('=')
		if at == -1 or at > 8:
			continue

		# get key
		key = line[0:at].strip()

		# parse value (string vs number, remove comment)
		val = line[at+1:].strip()
		if val[0] == "'":
			# string
			val = val[1:val[1:].find("'")]
		else:
			# number or T/F
			at = val.find('/')
			if at == -1: at = len(val)
			val = val[0:at].strip()
			if val.lower() in ['t', 'f']:
				# T/F
				val = val.lower() == 't'
			else:
				# Number
				val = float(val)
				if int(val) == val:
					val = int(val)
		res[key] = val
	return res;

def fitskw(hdrs, kw):
	""" Intelligently extract a keyword kw from an arbitrarely
	    shaped object ndarray of FITS headers.
	"""
	shape = hdrs.shape
	hdrs = hdrs.reshape(hdrs.size)

	res = []
	cache = dict()
	for ahdr in hdrs:
		ident = id(ahdr)
		if ident not in cache:
			if ahdr is not None:
				#hdr = pyfits.Header( txtfile=StringIO(ahdr) )
				hdr = fits_quickparse(ahdr)
				cache[ident] = hdr[kw]
			else:
				cache[ident] = None
		res.append(cache[ident])

	#assert res == _fitskw_dumb(hdrs, kw)

	res = np.array(res).reshape(shape)
	return res

###################################################################
## Auxilliary functions implementing Catalog.build_neighbor_cache
## functionallity
def _cache_maker_mapper(rows, margin_x):
	# Map: fetch all objects to be mapped, return them keyed
	# by cell ID and table
	self         = _cache_maker_mapper
	cat          = self.CATALOG
	cell_id      = self.CELL_ID

	p, _ = cat.cell_bounds(cell_id)

	# Find all objects within 'margin_x' from the cell pixel edge
	# The pixel can be a rectangle, or a triangle, so we have to
	# handle both situations correctly.
	(x1, x2, y1, y2) = p.boundingBox()
	d = x2 - x1
	(cx, cy) = p.center()
	if p.nPoints() == 4:
		s = 1. - 2*margin_x / d
		p.scale(s, s, cx, cy)
	elif p.nPoints() == 3:
		if (cx - x1) / d > 0.5:
			ax1 = x1 + margin_x*(1 + 2**.5)
			ax2 = x2 - margin_x
		else:
			ax1 = x1 + margin_x
			ax2 = x2 - margin_x*(1 + 2**.5)

		if (cy - y1) / d > 0.5:
			ay2 = y2 - margin_x
			ay1 = y1 + margin_x*(1 + 2**.5)
		else:
			ay1 = y1 + margin_x
			ay2 = y2 - margin_x*(1 + 2**.5)
		p.warpToBox(ax1, ax2, ay1, ay2)
	else:
		raise Exception("Expecting the pixel shape to be a rectangle or triangle!")

	# Now reject everything not within the margin, and
	# (for simplicity) send everything within the margin,
	# no matter close to which edge it actually is, to
	# all neighbors.
	(ra, dec) = rows.as_columns()
	(x, y) = bhpix.proj_bhealpix(ra, dec)
	#in_ = np.fromiter( (not p.isInside(px, py) for (px, py) in izip(x, y)), dtype=np.bool, count=len(x))
	in_ = ~p.isInsideV(x, y)

	if not in_.any():
		return Empty

	# Load full rows, across all tablets, keeping only
	# those with in_ == True
	data = load_full_rows(cat, cell_id, in_)

	# Mark these to be replicated all over the neighborhood
	res = []
	if len(data):
		for neighbor in cat.pix.neighboring_cells(cell_id):
			res.append( (neighbor, data) )

	##print "Scanned margins of %s (%d objects)" % (cat._tablet_file(self.CELL_ID, table=cat.primary_table), len(data[cat.primary_table]['rows']))

	return res

def load_full_rows(cat, cell_id, in_):
	""" Load all rows for all tablets, keeping only those with in_ == True.
	    Return a nested dict:

	    	ret = {
		    	table_name: {
		    		'rows': rows (ndarray)
		    		'blobs': {
		    			blobcolname: blobs (ndarray)
		    			...
		    		}
			}
		}

	    Any blobs referred to in rows will be stored in blobs, to be indexed
	    like:

	       blobs = ret[table]['blobs'][blobcol][ ret[table][blobcolref] ]
	"""
	data = { }
	for table in cat.tables:
		data[table] = {}

		# load all rows
		rows = cat.fetch_tablet(cell_id, table)[in_]

		# load all blobs
		data[table]['blobs'] = {}
		schema = cat._get_schema(table)
		if 'blobs' in schema:
			for bcolname in schema['blobs']:
				# Get only unique blobs, and reindex accordingly
				blobrefs, _, idx = np.unique(rows[bcolname], return_index=True, return_inverse=True)
				idx = idx.reshape(rows[bcolname].shape)
				rows[bcolname] = idx
				assert rows[bcolname].min() == 0
				assert rows[bcolname].max() == len(blobrefs)-1

				# Fetch unique blobs
				blobs    = cat.fetch_blobs(cell_id, table, bcolname, blobrefs)

				# In the end, blobs will contain N unique blobs, while rows[bcolname] will
				# have 0-based indices to those blobs
				data[table]['blobs'][bcolname] = blobs

		# This must follow the blob resolution, as it may
		# modify the indices in the rows
		data[table]['rows'] = rows

	return data

def extract_full_rows_subset(allrows, in_):
	# Takes allrows in the format returned by load_full_rows and
	# extracts those with in_==True, while correctly reindexing any
	# BLOBs that are in there.
	#
	# Also works if in_ is a ndarray of indices.
	ret = {}
	for table, data in allrows.iteritems():
		rows = data['rows'][in_]
		xblobs = {}
		for (bcolname, blobs) in data['blobs'].iteritems():
			# reindex blob refs
			blobrefs, _, idx = np.unique(rows[bcolname], return_index=True, return_inverse=True)
			idx = idx.reshape(rows[bcolname].shape)

			xblobs[bcolname] = blobs[blobrefs];
			rows[bcolname]   = idx

			assert rows[bcolname].min() == 0
			assert rows[bcolname].max() == len(blobrefs)-1
			assert (xblobs[bcolname][ rows[bcolname] ] == blobs[ data['rows'][in_][bcolname] ]).all()

		ret[table] = { 'rows': rows, 'blobs': xblobs }
	return ret

def write_neighbor_cache(cat, cell_id, nborblocks):
	# Store a list of full rows (as returned by load_full_rows)
	# to neighbor tables of tablets in cell cell_id
	# of catalog cat

	assert cat.is_cell_local(cell_id)

	ncached = 0
	with cat.get_cell(cell_id, mode='w') as cell:
		for table, schema in cat.tables.iteritems():
			if 'cached_flag' in schema:
				cachedFlag = schema['cached_flag']
			else:
				cachedFlag = None

			with cell.open(table=table) as fp:
				# Drop existing cache
				if 'cached' in fp.root:
					fp.removeNode('/', 'cached', recursive=True);

				# Create destinations for rows and blobs
				fp.createGroup('/', 'cached', title='Cached objects from neighboring cells')
				fp.root.main.table.copy('/cached', 'table', start=0, stop=0, createparents=True)
				blobs = set(( name for nbor in nborblocks for (name, _) in nbor[table]['blobs'].iteritems() ))
				for name in blobs:
					fp.createVLArray('/cached/blobs', name, tables.ObjectAtom(), "BLOBs", createparents=True)
					fp.root.cached.blobs.__getattr__(name).append(0)	# ref=0 should be pointed to by no real element (equivalent to NULL pointer)
				haveblobs = len(blobs) != 0

				# Write records (rows and blobs)
				for nbor in nborblocks:
					rows  = nbor[table]['rows']

					if haveblobs:
						# Append cached blobs, and adjust the offsets
						rows = rows.copy()		# Need to do this, so that modifications to rows[name] aren't permanent
						blobs = nbor[table]['blobs']
						for (name, data) in blobs.iteritems():
							barray = fp.root.cached.blobs.__getattr__(name)
							rows[name] += len(barray)
							rows[name] *= -1		# Convention: cached blob refs are negative
							for obj in data:
								barray.append(obj)

					# Append cached rows
					if cachedFlag:
						rows[cachedFlag] = True

					fp.root.cached.table.append(rows)

				# sanity
				if ncached == 0:
					ncached = fp.root.cached.table.nrows
				assert ncached == fp.root.cached.table.nrows

	return ncached

def _cache_maker_reducer(cell_id, nborblocks):
	self = _cache_maker_reducer
	cat          = self.CATALOG

	#print "Would write to %s." % (cat._tablet_file(cell_id, table=cat.primary_table));
	#exit()

	ncached = write_neighbor_cache(cat, cell_id, nborblocks);

	# Return the number of new rows cached into this cell
	return (cell_id, ncached)

###################################################################
