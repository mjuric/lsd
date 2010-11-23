#!/usr/bin/env python

import subprocess
import tables
import numpy as np
import pyfits
import bhpix
import time
import sys
import os
import json
import utils
import cPickle
from utils import is_scalar_of_type
from StringIO import StringIO
from pixelization import Pixelization
from collections import OrderedDict
from contextlib import contextmanager
from table import Table

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

class BLOBAtom(tables.ObjectAtom):
    """
    Sema as tables.ObjectAtom, except that it uses the highest available
    pickle protocol to serialize the objects.
    """
    def _tobuffer(self, object_):
        return cPickle.dumps(object_, -1)

class ColumnType(object):
	""" A simple record representing columns. Built at runtime
	    from _tables entries, and stored in Catalog.columns
	"""
	name    = None
	table   = None
	dtype   = None
	is_blob = False

class Catalog:
	""" A spatially and temporally partitioned object catalog.
	
	    The usual workhorses are Catalog.fetch, Catalog.iterate
	    and Catalog.map_reduce methods.
	"""

#	class TableSchema(object):
#		""" A simple record representing column groups.
#		"""
#		name    = None
#		columns = None

	path = '.'
	pix  = Pixelization(level=6, t0=54335, dt=1)
				# t0: default starting epoch (== 2pm HST, Aug 22 2007 (night of GPC1 first light))
				# t1: default temporal resolution (in days)
	_nrows = 0

	NULL = 0		# The value for NULL in JOINed rows that had no matches

	_tables  = None		# Tables in catalog ( dict of lists of (tablename, schema, primary_key) tuples)

	columns       = None
	primary_table = None	# Primary table of this catalog ( the one holding the IDs and spatial/temporal keys)
	primary_key   = None
	spatial_keys  = None
	temporal_key  = None

	### File name/path related methods
	def _get_table_data_path(self, table):
		""" Allow individual tables to override where they're placed
		    This comes in handy for direct JOINs.
		"""
		schema = self._get_schema(table)
		return schema.get('path', '%s/data' % (self.path))

	def _tablet_filename(self, table):
		""" The filename of a tablet in a cell """
		return '%s.%s.h5' % (self.name, table)

	def _tablet_file(self, cell_id, table):
		return '%s/%s/%s' % (self._get_table_data_path(table), self.pix.path_to_cell(cell_id), self._tablet_filename(table))

	def tablet_exists(self, cell_id, table=None):
		""" Return True if the given tablet exists in cell_id """
		if table is None:
			table = self.primary_table

		assert table in self._tables

		fn = self._tablet_file(cell_id, table)
		return os.access(fn, os.R_OK)

	def _cell_prefix(self, cell_id):
		return '%s/%s/%s' % (self._get_table_data_path(self.primary_table), self.pix.path_to_cell(cell_id), self.name)

	def static_if_no_temporal(self, cell_id):
		""" See if we have data in cell_id. If not, return a
		    corresponding static sky cell_id. Useful when evaluating
		    static-temporal JOINs
		"""
		if not self.pix.is_temporal_cell(cell_id):
			return cell_id

		if self.tablet_exists(cell_id):
			##print "Temporal cell found!", self._cell_prefix(cell_id)
			return cell_id

		# return corresponding static-sky cell
		cell_id = self.pix.static_cell_for_cell(cell_id)
		#print "Reverting to static sky", self._cell_prefix(cell_id)
		return cell_id

	def get_cells(self, bounds, return_bounds=False):
		""" Return a list of cells
		"""
		data_path = self._get_table_data_path(self.primary_table)
		pattern   = self._tablet_filename(self.primary_table)

		return self.pix.get_cells(data_path, pattern, bounds, return_bounds=return_bounds)

	def is_cell_local(self, cell_id):
		""" Returns True if the cell is reachable from the
		    current machine. A placeholder for if/when I decide
		    to make this into a true distributed database.
		"""
		return True

	#############

	def _load_schema(self):
		data = json.loads(file(self.path + '/schema.cfg').read(), object_pairs_hook=OrderedDict)

		self.name = data["name"]
		self._nrows = data.get("nrows", None)

		######################
		# Backwards compatibility
		level, t0, dt = data["level"], data["t0"], data["dt"]
		self.pix = Pixelization(level, t0, dt)

		# Load table definitions
		if isinstance(data['tables'], dict):
			# Backwards compatibility, keeps ordering because of objecct_pairs_hook=OrderedDict above
			self._tables = data["tables"]
		else:
			self._tables = OrderedDict(data['tables'])

		# Postprocessing: fix cases where JSON restores arrays instead
		# of tuples, and tuples are required
		for _, schema in self._tables.iteritems():
			schema['columns'] = [ tuple(val) for val in schema['columns'] ]

		self._rebuild_internal_schema()

	def _rebuild_internal_schema(self):
		# Rebuild internal representation of the schema from self._tables
		# OrderedDict
		self.columns = OrderedDict()
		self.primary_table = None

		for table, schema in self._tables.iteritems():
			for colname, dtype in schema['columns']:
				assert colname not in self.columns
				self.columns[colname] = ColumnType()
				self.columns[colname].name  = colname
				self.columns[colname].dtype = np.dtype(dtype)
				self.columns[colname].table = table

			if self.primary_table is None:
				self.primary_table = table
				if 'primary_key'  in schema:
					self.primary_key  = self.columns[schema['primary_key']]
				if 'temporal_key' in schema:
					self.temporal_key = self.columns[schema['temporal_key']]
				if 'spatial_keys' in schema:
					(lon, lat) = schema['spatial_keys']
					self.spatial_keys = (self.columns[lon], self.columns[lat])
			else:
				# If any of these are defined, they must be defined in the
				# primary table
				assert 'primary_key'  not in schema
				assert 'spatial_keys' not in schema
				assert 'temporak_key' not in schema

			if 'blobs' in schema:
				for colname in schema['blobs']:
					assert self.columns[colname].dtype.base == np.int64, "Data structure error: blob reference columns must be of int64 type"
					self.columns[colname].is_blob = True

	def _store_schema(self):
		data = dict()
		data["level"], data["t0"], data["dt"] = self.pix.level, self.pix.t0, self.pix.dt
		data["nrows"] = self._nrows
		data["tables"] = self._tables.items()
		data["name"] = self.name

		f = open(self.path + '/schema.cfg', 'w')
		f.write(json.dumps(data, indent=4, sort_keys=True))
		f.close()

	###############

	def create_table(self, table, schema, ignore_if_exists=False):
		# Create a new table and set it as primary if it
		# has a primary_key
		if table in self._tables and not ignore_if_exists:
			raise Exception('Trying to create a table that already exists!')

		self._tables[table] = schema

		if 'primary_key' in schema:
			if 'spatial_keys' not in schema:
				raise Exception('Trying to create a primary table with no spatial keys!')
			if self.primary_table is not None:
				raise Exception('Trying to create a primary table ("%s") while one ("%s") already exists!' % (table, self.primary_table))
			self.primary_table = table

		if 'blobs' in schema:
			cols = dict(schema['columns'])
			for blobcol in schema['blobs']:
				assert is_scalar_of_type(cols[blobcol], np.int64)

		self._rebuild_internal_schema()
		self._store_schema()

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
	def _get_row_group(self, fp, group, table):
		""" Obtain a handle to the given HDF5 node, autocreating
		    it if necessary.
		"""
		assert group in ['main', 'cached']

		g = getattr(fp.root, group, None)

		if g is None:
			schema = self._get_schema(table)

			# Table
			fp.createTable('/' + group, 'table', np.dtype(schema["columns"]), expectedrows=20*1000*1000, createparents=True)
			g = getattr(fp.root, group)
			

			# Primary key sequence
			if group == 'main' and 'primary_key' in schema:
				seqname = '_seq_' + schema['primary_key']
				fp.createArray(g, seqname, np.array([1], dtype=np.uint64))

			# BLOB storage arrays
			if 'blobs' in schema:
				for blobcol in schema['blobs']:
					fp.createVLArray('/' + group +'/blobs', blobcol, BLOBAtom(), "BLOBs", createparents=True)
					getattr(g.blobs, blobcol).append(None)	# ref=0 always points to None
		return g

	def drop_row_group(self, cell_id, group):
		""" Delete the given HDF5 group and all its children
		    from all tablets of cell cell_id
		"""
		with self.get_cell(cell_id, mode='w') as cell:
			for table in self._tables:
				with cell.open(table) as fp:
					if group in fp.root:
						fp.removeNode('/', group, recursive=True);

	def _create_tablet(self, fn, table):
		# Create a tablet at a given path, for table 'table'
		assert os.access(fn, os.R_OK) == False

		# Create the cell directory if it doesn't exist
		path = fn[:fn.rfind('/')];
		if not os.path.exists(path):
			utils.mkdir_p(path)

		# Create the tablet
		fp  = tables.openFile(fn, mode='w')

		# Force creation of the main subgroup
		self._get_row_group(fp, 'main', table)

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
			if not os.path.isdir(self.path):
				raise Exception('Cannot access table: "%s" is inexistant or not readable.' % (path))
			self._load_schema()

	def create_catalog(self, name, path, level, t0, dt):
		""" Create a new catalog and store its definition.
		"""
		self.path = path

		utils.mkdir_p(self.path)
		if os.path.isfile(self.path + '/dbinfo.json'):
			raise Exception("Creating a new catalog in '%s' would overwrite an existing one." % self.path)

		self._tables = OrderedDict()
		self.columns = OrderedDict()
		self.name = name

		if level == Automatic: level = self.pix.level
		if    t0 == Automatic: t0 = self.pix.t0
		if    dt == Automatic: dt = self.pix.dt
		self.pix = Pixelization(level, t0, dt)

		self._store_schema()

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

	def append(self, cols_, group='main', cell_id=None):
		""" Insert a set of rows into a table in the database. Protects against
		    multiple writers simultaneously inserting into the same file.

		    If table being inserted into has spatial_keys, the rows being
		    inserted MUST contain the primary key column.

		    Return: array of primary keys of inserted rows
		"""

		assert group in ['main', 'cached']

		# Resolve aliases in the input, and prepare a Table()
		cols = Table()
		if getattr(cols_, 'items', None) is not None:
			cols_ = cols_.items()
		for name, col in cols_:
			cols.add_column(self.resolve_alias(name), col)
		assert cols.ncols()

		# if the primary key column has not been supplied by the user, add it
		key = self.get_primary_key()
		if key not in cols:
			cols[key] = np.zeros(len(cols), dtype=self.columns[key].dtype)

		# Locate the cells into which we're going to store the rows
		# - if <cell_id> is not None: insert as-is into the requested cell. The user is responsible for ensuring <prim_key> existence and uniqueness.
		# - elif <primary_key> column exists and not all zeros: compute destination cells from it
		# - elif <spatial_keys> columns exist: use them to determine destination cells
		genKeys = False
		if cell_id is not None:
			# Explicit destination cell(s) have been provided
			cells = np.array(cell_id, copy=False, ndmin=1)
			if len(cells) == 1:
				cells = np.resize(cells, len(cols))
			assert len(cells) == len(cols)
		elif key in cols and np.any(cols[key]):
			# Deduce destination cells from IDs
			assert 0, "Test this codepath!"
			cells = self.pix.cell_for_id(cols[key])
		else:
			# Deduce destination cells from spatial and temporal keys (the fallback)
			assert group == 'main'

			lonKey, latKey = self.get_spatial_keys()
			assert lonKey and latKey, "The table must have at least the spatial keys!"
			assert lonKey in cols and latKey in cols, "The input must contain at least the spatial keys!"

			tKey = self.get_temporal_key()
			t = cols[tKey] if tKey is not None else None

			cols[key][:] = self.pix.obj_id_from_pos(cols[lonKey], cols[latKey], t)
			cells        = self.pix.cell_for_id(cols[key])

			# TODO: Debugging, remove when happy
			assert np.all(self.pix.cell_id_for_pos(cols[lonKey], cols[latKey], t) == cells)

			genKeys = True

		####
		#schema = self._get_schema(self.primary_table)
		#raKey, decKey = schema["spatial_keys"]
		#key           = schema["primary_key"]
		#if key not in cols:	# if the primary key column has not been supplied, autoadd it
		#	cols[key] = np.empty(n, dtype=np.dtype(dict(schema['columns'])[key]))
		#
		#ra, dec = cols[raKey], cols[decKey]
		#if "temporal_key" in schema:
		#	t = cols[schema["temporal_key"]]
		#else:
		#	t = None
		#cells     = self.pix.cell_id_for_pos(ra, dec, t)
		#cols[key] = self.pix.obj_id_from_pos(ra, dec, t)
		####

		# TODO: Debugging, remove when confident
		#tmp = self.pix.cell_for_id(cols[key])
		##print self.pix.str_id(cells)
		##print self.pix.str_id(cols[key][:10])
		#assert np.all(tmp == cells)

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
			cols2  = cols[incell]
			nrows  = len(cols2)

			# Store cell groups into their tablets
			for table, schema in self._tables.iteritems():
				fp    = self._open_tablet(cell_id, mode='w', table=table)
				g     = self._get_row_group(fp, group, table)
				t     = g.table
				blobs = schema['blobs'] if 'blobs' in schema else dict()

				if table == self.primary_table and genKeys:
					id_seq = getattr(g, '_seq_' + key, None)
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
						assert cols2[colname].dtype == object
						uobjs, _, ito = np.unique(cols2[colname], return_index=True, return_inverse=True)	# Note: implicitly flattens multi-D input arrays
						ito = ito.reshape(rows[colname].shape)	# De-flatten the output indices

						# Offset indices
						barray = getattr(g.blobs, colname)
						bsize = len(barray)
						ito = ito + bsize

						# Remap any None values to index 0 (where None is stored by convention)
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
			self._nrows = self._nrows + nrows
			ntot = ntot + nrows

		assert ntot == len(cols), 'ntot != len(cols), ntot=%d, len(cols)=%d, cell_id=%d' % (ntot, len(cols), cell_id)
		assert len(np.unique1d(cols[key])) == len(cols), 'len(np.unique1d(cols[key])) != len(cols) (%s != %s) in cell %s' % (len(np.unique1d(cols[key])), len(cols), cell_id)

		return cols[key]

	def nrows(self):
		return self._nrows

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
		i = i + 'Tables:        %s' % str(self._tables.keys())
		i = i + '\n'
		s = ''
		for table, schema in dict(self._tables).iteritems():
			s = s + '-'*31 + '\n'
			s = s + 'Table \'' + table + '\':\n'
			s = s + "%20s %10s\n" % ('Column', 'Type')
			s = s + '-'*31 + '\n'
			for col in schema["columns"]:
				s = s + "%20s %10s\n" % (col[0], col[1])
			s = s + '-'*31 + '\n'
		return i + s

	def _get_schema(self, table):
		return self._tables[table]

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

	def fetch_blobs(self, cell_id, column, refs, include_cached=False):
		""" Fetch blobs from column 'column'
		    in cell cell_id, given a vector of references 'refs'

		    If the cell_id has a temporal component, and there's no
		    tablet in that cell, a static sky cell corresponding
		    to it is tried next.
		"""
		# short-circuit if there's nothing to be loaded
		if len(refs) == 0:
			return np.empty(refs.shape, dtype=np.object_)

		# Get the table for this column
		table = self.columns[column].table

		# revert to static sky cell if cell_id is temporal but
		# unpopulated (happens in static-temporal JOINs)
		cell_id = self.static_if_no_temporal(cell_id)

		# Flatten refs; we'll deflatten the blobs in the end
		shape = refs.shape
		refs = refs.reshape(refs.size)

		# load the blobs arrays
		with self.get_cell(cell_id) as cell:
			with cell.open(table) as fp:
				b1 = getattr(fp.root.main.blobs, column)
				if include_cached and 'cached' in fp.root:
					# We have cached objects in 'cached' group -- read the blobs
					# from there as well. blob refs of cached objects are
					# negative.
					b2 = getattr(fp.root.cached.blobs, column)

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

		# revert to static sky cell if cell_id is temporal but
		# unpopulated (happens in static-temporal JOINs)
		cell_id = self.static_if_no_temporal(cell_id)

		if self.tablet_exists(cell_id, table):
			with self.get_cell(cell_id) as cell:
				with cell.open(table) as fp:
					rows = fp.root.main.table.read()
					if include_cached and 'cached' in fp.root:
						rows2 = fp.root.cached.table.read()
						# Make any neighbor cache BLOBs negative (so that fetch_blobs() know to
						# look for them in the cache, instead of 'main')
						schema = self._get_schema(table)
						if 'blobs' in schema:
							for blobcol in schema['blobs']:
								rows[blobcol] *= -1
						# Append the data from cache to the main tablet
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

	def get_spatial_keys(self):
		# Find out which columns are our spatial keys
		return (self.spatial_keys[0].name, self.spatial_keys[1].name) if self.spatial_keys is not None else (None, None)

	def get_primary_key(self):
		# Find out which columns are our spatial keys
		return self.primary_key.name

	def get_temporal_key(self):
		return self.temporal_key.name if self.temporal_key is not None else None

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
