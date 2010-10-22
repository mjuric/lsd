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
from slalib import *
import Polygon
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
from utils import astype, gc_dist

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
	level = 6
	t0 = 47892		# default starting epoch (MJD, 47892 == Jan 1st 1990)
	dt = 90			# default temporal resolution (in days)
	__nrows = 0

	NULL = 0		# The value for NULL in JOINed rows that had no matches

	tables  = None		# Tables in catalog ( dict of lists of (tablename, schema, primary_key) tuples)
	hidden_tables = None	# Tables in catalog that do not participate in direct joins/fetches (e.g., xmatch tables)
	primary_table = None	# Primary table of this catalog ( the one holding the IDs and spatial/temporal keys)

	xmatched_catalogs = None	# External, cross-matched catalogs (see add_xmatched_catalog for schema of this dict)

	def all_columns(self):
		# Return the list of all columns in all tables of the catalog
		cols = []
		for _, schema in self.tables.iteritems():
			cols += schema['columns']
		return [ name for name, _ in cols ]

	def cell_for_id(self, id):
		# Note: there's a more direct way of doing this,
		#       but I don't have time to think about it right now
		x, y, t, _ = self.unpack_id(id)
		cell_id = self._id_from_xy(x, y, t, self.level)
		return cell_id

	def cell_for_pos(self, ra, dec, t=None):
		return self.id_from_pos(ra, dec, t, self.level)

	def id_from_pos(self, ra, dec, t=None, level=10):
		# find the bhealpix coordinates and time slice
		(x, y) = bhpix.proj_bhealpix(ra, dec)

		return self._id_from_xy(x, y, t, level)

	def _id_from_xy(self, x, y, t=None, level=10):
		if t == None:
			t = np.array([self.t0])

		(x, y) = bhpix.xy_center(x, y, level)		# round to requested pixelization level
		ct = astype((t - self.t0) / self.dt, int)
		if type(ct) == np.ndarray:
			ct[ct < 0] = 0
		elif ct < 0:
			ct = 0

		# construct the 32bit ID prefix from the above
		# Prefix format: 10bit x + 10bit y + 12bit time
		ix   = astype((1 + x) / 2. * 2**10, np.uint64)
		iy   = astype((1 + y) / 2. * 2**10, np.uint64)
		id   = ix << 22
		id  |= iy << 12
		id  |= ct & 0xFFF
		id <<= 32

		# NOTE: Test tranformation correctness (comment this out for production code)
		#(ux, uy, ut, ui) = self.unpack_id(id, level)
		#cc = bhpix.xy_center(x, y, self.level)
		#cu = bhpix.xy_center(ux, uy, self.level)
		#ct = ct * self.dt + self.t0
		#if np.any(cc[0] != cu[0]) or np.any(cc[1] != cu[1]) or np.any(ct != ut) or np.any(ui != 0):
		#	print cc, "==", cu, ct, "==", ut
		#	raise Exception("**** Bug detected ****")			

		return id

	def unpack_id(self, id, level = 10):
		# return (approximate) healpix position and
		# time slice for the given id
		id = astype(id, np.uint64)
		ci = id & 0xFFFFFFFF
		id >>= 32
		cx = 2 * astype(id >> 22, float)              / 2**10 - 1
		cy = 2 * astype((id & 0x3FF000) >> 12, float) / 2**10 - 1
		ct = astype(id & 0xFFF, float) * self.dt + self.t0
		(cx, cy) = bhpix.xy_center(cx, cy, level)
		return (cx, cy, ct, ci)

	def cell_bounds(self, cell_id):
		"""
			Return the bounding polygon and time
			for a given cell.
		"""
		x, y, t, _ = self.unpack_id(cell_id, self.level)
		bounds = self._cell_bounds_xy(x, y)
		return (bounds, t)

	def _cell_bounds_xy(self, x, y, dx = None):
		if dx == None:
			dx = bhpix.pix_size(self.level)

		bounds = Polygon.Shapes.Rectangle(dx)
		bounds.shift(x - 0.5*dx, y - 0.5*dx);

		if fabs(fabs(x) - fabs(y)) == 0.5:
			# If it's a "halfpixel", return a triangle
			# by clipping agains the sky
			bounds &= footprint.ALLSKY
		return bounds

	def _cell_prefix(self, cell_id):
		(x, y, t, rank) = self.unpack_id(cell_id, self.level)
		subpath = bhpix.get_path(x, y, self.level)

		if t >= self.t0 + self.dt:
			prefix = '%s/%s/mjd%05d%+d' % (self.path, subpath, t, self.dt)
		else:
			prefix = '%s/%s/static' % (self.path, subpath)

		return prefix

	def _tablet_file(self, cell_id, table):
		return "%s.%s.h5" % (self._cell_prefix(cell_id), table)

	def tablet_exists(self, cell_id, table=None):
		if table == None:
			table = self.primary_table

		assert (table in self.tables) or (table in self.hidden_tables)

		fn = self._tablet_file(cell_id, table)
		return os.access(fn, os.R_OK)

	def _load_dbinfo(self):
		data = json.loads(file(self.path + '/dbinfo.json').read())

		self.name = data["name"]
		self.level = data["level"]
		self.t0 = data["t0"]
		self.dt = data["dt"]
		self.__nrows = data["nrows"]

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
					"cached_flag": "cached"
				}
			}
			data["primary_table"] = 'catalog'
		if "hidden_tables" not in data: data["hidden_tables"] = {}
		if "xmatched_catalogs" not in data: data["xmatched_catalogs"] = {}
		###############################

		# Load table definitions
		self.tables = data["tables"]
		self.hidden_tables = data["hidden_tables"]
		self.primary_table = data["primary_table"]
		self.xmatched_catalogs = data["xmatched_catalogs"]

		# Postprocessing: fix cases where JSON restores arrays instead
		# of tuples, and tuples are required
		for table, schema in self.tables.iteritems():
			schema['columns'] = [ tuple(val) for val in schema['columns'] ]
		for table, schema in self.hidden_tables.iteritems():
			schema['columns'] = [ tuple(val) for val in schema['columns'] ]

	def _store_dbinfo(self):
		data = dict()
		data["level"], data["t0"], data["dt"] = self.level, self.t0, self.dt
		data["nrows"] = self.__nrows
		data["tables"] = self.tables
		data["hidden_tables"] = self.hidden_tables
		data["primary_table"] = self.primary_table
		data["name"] = self.name
		data["xmatched_catalogs"] = self.xmatched_catalogs

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

		# Create the tablet
		fp  = tables.openFile(fn, mode='w')
		fp.createTable('/', 'table', np.dtype(schema["columns"]), expectedrows=20*1000*1000)
		if 'primary_key' in schema:
			seqname = '_seq_' + schema['primary_key']
			fp.createArray('/', seqname, np.array([1], dtype=np.uint64))

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

		fp.root.table.append(rows)

		fp.close()

	## Cell enumeration routines
	def _get_cells_recursive(self, cells, foot, pix):
		""" Helper for _get_cells(). See documentation of
		    _get_cells() for usage
		"""
		# Check for nonzero overlap
		lev = bhpix.get_pixel_level(pix[0], pix[1])
		dx = bhpix.pix_size(lev)
		#box = Polygon.Shapes.Rectangle(dx)
		#box.shift(pix[0] - 0.5*dx, pix[1] - 0.5*dx);
		box = self._cell_bounds_xy(pix[0], pix[1], dx)
		foot = foot & box
		if not foot:
			return

		# Check for existence of leaf file(s). There can be
		# more than one file in catalogs with a time component
		prefix = self.path + '/' + bhpix.get_path(pix[0], pix[1], lev)
		fn = None
		pattern = "%s/*.%s.h5" % (prefix, self.primary_table)
		for fn in glob.iglob(pattern):
			if(foot.area() == box.area()):
				foot = None

			# parse out the time, construct cell ID
			fname = fn[fn.rfind('/')+1:];
			t = fname.split('.')[-3]
			t = self.t0 if t == 'static' else float(t)
			cell_id = self._id_from_xy(pix[0], pix[1], t, self.level)

			cells += [ (cell_id, foot) ]
		if fn != None:
			return

		# Check if the directory node exists (and stop subdividing if it doesn't)
		if not os.path.isdir(prefix):
			return

		# Recursively subdivide the four subpixels
		dx = dx / 2
		for d in np.array([(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]):
			self._get_cells_recursive(cells, foot & box, pix + dx*d)

	def _get_cells(self, foot = All):
		""" Return a list of (cell_id, footprint) tuples completely
		    covering the requested footprint
		"""
		# Handle all-sky
		if foot == All:
			foot = footprint.ALLSKY
		else:
			# Restrict to valid sky
			foot = foot & footprint.ALLSKY

		# Divide and conquer to find the database cells
		cells = []
		self._get_cells_recursive(cells, foot, (0., 0.))
		return cells

	def _prep_query(self, cols):
		#
		# Need: dtype for output table, list of tables and their fields
		#
		if cols == All: cols = [ '*' ]
		if type(cols) == str: cols = cols.split()

		# At this point, cols is an array of tokens (columns, wildcards)
		# Construct the output dtype as well
		tables = defaultdict(dict)
		dtype = []
		cats = dict()
		for token in cols:
			# get table/column
			p = token.split('.')
			if len(p) == 1:
				table = ''; column = p[0];
			else:
				table = p[0]; column = p[1]

			# Locate this catalog
			if table not in cats:
				cats[table] = self if table == '' else self.get_xmatched_catalog(table)
			cat = cats[table]

			# Check for asterisk
			if column == '*':
				columns = [ name for (name, _) in cat.columns ]
			else:
				columns = [ column ]

			# Add output columns, ignoring those already inserted
			colspec = dict(cat.columns)
			for column in columns:
				outname = table + '.' + column if table != '' else column
				if column not in tables[table]:
					tables[table][column] = outname
					dtype += [ (outname, colspec[column]) ]

		# Verify there's at least one column referencing the main catalog
		if '' not in tables:
			raise Exception('There has to be at least one column referencing the main catalog')

		ret = (dtype, tables)

		return ret

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
		self.xmatched_catalogs = {}
		self.name = name

		if level != Automatic: self.level = level
		if    t0 != Automatic: self.t0 = t0
		if    dt != Automatic: self.dt = dt

		self._store_dbinfo()

	def update(self, table, keys, rows):
		raise Exception('Not implemented')

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
			if n == None: n = len(col)
			assert n == len(col), 'n=%d len(col)=%d' % (n, len(col))

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
				except subprocess.CalledProcessError as err:
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
				fp  = self._open_tablet(cell_id, mode='w', table=table)
				t   = fp.root.table

				if table == self.primary_table:
					id_seq = fp.root.__getattr__('_seq_' + key)
					cols[key][incell] += np.arange(id_seq[0], id_seq[0] + nrows, dtype=np.uint64)
					cols2[key] = cols[key][incell]
					id_seq[0] += n

				rows = np.zeros(nrows, dtype=np.dtype(schema['columns']))
				for col in rows.dtype.names:
					if col not in cols2:
						continue
					rows[col] = cols2[col]

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
		i = i + 'Partitioning:  level=%d\n' % (self.level)
		i = i + '(t0, dt):      %f, %f \n' % (self.t0, self.dt)
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

	def fetch_cell(self, cell_id, table=None, include_cached=False):
		""" Load and return all rows from a given tablet
		"""
		if table == None:
			table = self.primary_table

		if self.tablet_exists(cell_id, table):
			with self.get_cell(cell_id) as cell:
				with cell.open(table) as fp:
					rows = fp.root.table.read()
					if include_cached and 'cached' in fp.root:
						rows2 = fp.root.cached.read()
						rows = np.append(rows, rows2)
		else:
			schema = self._get_schema(table)
			rows = np.empty(0, dtype=np.dtype(schema['columns']))

		return rows

	def fetch(self, cols=All, foot=All, where=None, testbounds=True, include_cached=False, join_type='outer', nworkers=None, progress_callback=None, filter=None, filter_args=()):
		""" Return a table (numpy structured array) of all rows within a
		    given footprint. Calls 'filter' callable (if given) to filter
		    the returned rows.
		    
		    The 'filter' callable should expect a single argument, rows,
		    being the set of rows (numpy structured array) to filter. It
		    must return the set of filtered rows (also as numpy structured
		    array). E.g., identity filter function would be:
		    
		    	def identity(rows):
		    		return rows

		    while a function filtering on column 'r' may look like:
		    
		    	def r_filter(rows):
		    		return rows[rows['r'] < 21.5]
		   
		    The filter callable must be piclkeable. Extra arguments to
		    filter may be given in 'filter_args'
		"""

		files = self._get_cells(foot)

		ret = None
		for rows in self.map_reduce(_iterate_mapper, mapper_args=(filter, filter_args), cols=cols, foot=foot, where=where, join_type=join_type, testbounds=testbounds, include_cached=include_cached, nworkers=nworkers, progress_callback=progress_callback):
			# ensure enough memory has been allocated (and do it
			# intelligently if not)
			if ret == None:
				ret = np.empty_like(rows)
				nret = 0

			while len(ret) < nret + len(rows):
				ret.resize(2*(len(ret)+1))
				#print "Resizing to", len(ret)

			# append
			ret[nret:nret+len(rows)] = rows
			nret = nret + len(rows)

		ret.resize(nret)
		#print "Resizing to", len(ret)
		return ret

	def iterate(self, cols=All, foot=All, where=None, testbounds=True, return_array=False, include_cached=False, join_type='outer', nworkers=None, progress_callback=None, filter=None, filter_args=()):
		""" Yield rows (either on a row-by-row basis if return_array==False
		    or in chunks (numpy structured array)) within a
		    given footprint. Calls 'filter' callable (if given) to filter
		    the returned rows.

		    See the documentation for Catalog.fetch for discussion of
		    'filter' callable.
		"""

		files = self._get_cells(foot)

		for rows in self.map_reduce(_iterate_mapper, mapper_args=(filter, filter_args), cols=cols, foot=foot, where=where, join_type=join_type, testbounds=testbounds, include_cached=include_cached, nworkers=nworkers, progress_callback=progress_callback):
			if return_array:
				yield rows
			else:
				for row in rows:
					yield row

	def map_reduce(self, mapper, reducer=None, query='*', foot=All, testbounds=True, include_cached=False, join_type='outer', mapper_args=(), reducer_args=(), nworkers=None, progress_callback=None):
		""" A MapReduce implementation, where rows from individual cells
		    get mapped by the mapper, with the result reduced by the reducer.
		    
		    Mapper, reducer, and all *_args must be pickleable.
		    
		    The mapper must be a callable expecting at least one argument, 'rows'.
		    'rows' is always the first argument; if any extra arguments are passed 
		    via mapper_args, they will come after it.
		    'rows' will be a numpy array of table records (with named columns)

		    The mapper must return a sequence of key-value pairs. All key-value
		    pairs will be merged by key into (key, [values..]) pairs that shall
		    be passed to the reducer.

		    The reducer must expect two parameters, the first being the key
		    and the second being a sequence of all values that the mappers
		    returned for that key. The return value of the reducer is passed back
		    to the user and is user-defined.
   
		    If the reducer is None, only the mapping step is performed and the
		    return value of the mapper is passed to the user.
		"""
		# slice up the job down to individual cells
		partspecs = self._get_cells(foot)

		# tell _mapper not to test polygon boundaries if the user requested so
		if not testbounds:
			partspecs = [ (part_id, None) for (part_id, bounds) in partspecs ]

		# start and run the workers
		pool = pool2.Pool(nworkers)
		if reducer == None:
			for result in pool.imap_unordered(
					partspecs, _mapper,
					mapper_args = (mapper, self, query, include_cached, mapper_args),
					progress_callback = progress_callback):

				if type(result) != type(Empty):
					yield result
		else:
			for result in pool.imap_reduce(
					partspecs, _mapper, _reducer,
					mapper_args  = (mapper, self, query, include_cached, mapper_args),
					reducer_args = (reducer, self, reducer_args),
					progress_callback = progress_callback):
				yield result

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
			if table == None:
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

	def neighboring_cells(self, cell_id, include_self=False):
		""" Returns the cell IDs for cells neighboring
		    the requested one both in space and time.
		    
		    If the cell_id is for static sky (i.e., it's time
		    bits are all zero), we return no temporal neighbors
		    (as this would be an infinite set).
		    
		    We do not check if the returned neighbor cells 
		    actually have any objects (exist).
		"""
		x, y, t, _ = self.unpack_id(cell_id, self.level)

		ncells = bhpix.neighbors(x, y, self.level, include_self)
		for (cx, cy) in ncells:
			if fabs(fabs(cx) - fabs(cy)) > 0.5:
				print "PROBLEM: ", x, y, cx, cy
				print ncells

		nhood = [ self._id_from_xy(x, y, t, self.level) for (x, y) in ncells ]

		# TODO: Remove once we're confident it works
		rrr = set([ self.unpack_id(cid, self.level)[0:2] for cid in nhood ])
		assert rrr == ncells


		# Add the time component unless this is a static-sky catalog
		if t != self.t0:
			nhood += [ self._id_from_xy(x, y, t + self.dt, self.level) for (x, y) in ncells ]
			nhood += [ self._id_from_xy(x, y, t - self.dt, self.level) for (x, y) in ncells ]

		return nhood

	def is_cell_local(self, cell_id):
		""" Returns True if the cell is reachable from the
		    current machine. A placeholder for if/when I decide
		    to make this into a true distributed database.
		"""
		return True

	def build_neighbor_cache(self, margin_x_arcsec=30, margin_t_days=0):
		""" Cache the objects found within margin_x (arcsecs) of
		    each cell into neighboring cells as well, to support
		    efficient nearest-neighbor lookups.

		    This routine works in tandem with _cache_maker_mapper
		    and _cache_maker_reducer auxilliary routines.
		"""
		margin_x = sqrt(2.) / 180. * (margin_x_arcsec/3600.)
		margin_t = margin_t_days

		# Find out which columns are our spatial keys
		schema = self._get_schema(self.primary_table)
		raKey, decKey = schema["spatial_keys"]
		query = "%s, %s" % (raKey, decKey)

		ntotal = 0
		ncells = 0
		for (cell_id, ncached) in self.map_reduce(_cache_maker_mapper, _cache_maker_reducer, query=query, mapper_args=(margin_x, margin_t)):
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
		return self._get_schema(self.primary_table)["spatial_keys"]

	def get_primary_key(self):
		# Find out which columns are our spatial keys
		return self._get_schema(self.primary_table)["primary_key"]

	def _fetch_xmatches(self, cell_id, ids, cat_to_name):
		"""
			Return a list of crossmatches corresponding to ids
		"""
		table = self.xmatched_catalogs[cat_to_name]['xmatch_table']

		if len(ids) == 0 or not self.tablet_exists(cell_id, table):
			return ([], [])

		rows = self.fetch_cell(cell_id, table)

		# drop all links where id1 is not in ids
		sids = np.sort(ids)
		res = np.searchsorted(sids, rows['id1'])
		res[res == len(sids)] = 0
		ok = sids[res] == rows['id1']
		rows = rows[ok]

		return (rows['id1'], rows['id2'])

	def add_xmatched_catalog(self, cat, xmatch_table):
		# Schema:
		#	catalog_name: {
		#		'path': path,
		#		'xmatch_table': table_name
		#	}
		assert xmatch_table in self.hidden_tables
		self.xmatched_catalogs[cat.name] = \
		{
			'path':		cat.path,
			'xmatch_table':	xmatch_table
		}
		self._store_dbinfo()

	def get_xmatched_catalog(self, catname):
		assert catname in self.xmatched_catalogs
		return Catalog(self.xmatched_catalogs[catname]['path'])

###############################################################
# Aux functions implementing Catalog.iterate and Catalog.fetch
# functionallity
def _iterate_mapper(rows, filter = None, filter_args = ()):
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

def table_join(id1, id2, m1, m2, join_type='outer'):
	# The algorithm assumes id1 and id2 have no
	# duplicated elements
	if False:
		x,y,t,_ = table_join.cat.unpack_id(table_join.cell_id)
		radec = bhpix.deproj_bhealpix(x, y)

		if len(np.unique1d(id1)) != len(id1):
			print "XXXXXXXXXXX"
			print "len(np.unique1d(id1)) != len(id1): ", len(np.unique1d(id1)), len(id1)
			print "cell_id = ", table_join.cell_id
			print "ra,dec =", radec
			assert len(np.unique1d(id1)) == len(id1)
		if len(np.unique1d(id2)) != len(id2):
			print "XXXXXXXXXXX"
			print "len(np.unique1d(id2)) != len(id2): ", len(np.unique1d(id2)), len(id2)
			print "cell_id = ", table_join.cell_id
			print "ra,dec =", radec
			assert len(np.unique1d(id2)) == len(id2)
		assert len(m1) == len(m2)

	if len(id1) != 0 and len(id2) != 0 and len(m1) != 0:
		i1 = id1.argsort()
		sid1 = id1[i1]
		res1 = np.searchsorted(sid1, m1)
		res1[res1 == len(sid1)] = 0

#		ok1 = sid1[res1] == m1
#		print 'id1:  ', id1
#		print 'i1:   ', i1
#		print 'sid1: ', sid1
#		print 'm1:   ', m1
#		print 'res1: ', res1
#		print 'ok1:  ', ok1 
#		print 'idx1: ', i1[res1[ok1]]
#		print 'i1_se:', id1[i1[res1[ok1]]]
#		print ''

		# Cull all links that we don't have in id2
		i2 = id2.argsort()
		sid2 = id2[i2]
		res2 = np.searchsorted(sid2, m2)
		res2[res2 == len(sid2)] = 0

#		ok2 = sid2[res2] == m2
#		print 'id2:  ', id2
#		print 'i2:   ', i2
#		print 'sid2: ', sid2
#		print 'm2:   ', m2
#		print 'res2: ', res2
#		print 'ok2:  ', ok2 
#		print 'idx2: ', i2[res2[ok2]]
#		print 'i2_se:', id2[i2[res2[ok2]]]
#		print ''

		# Now map links in m to indices in id1 and id2
		ok = (sid1[res1] == m1) & (sid2[res2] == m2)
#		print 'ok: ', ok1 & ok2

		idx1 = i1[res1[ok]]
		idx2 = i2[res2[ok]]
	else:
		idx1 = np.empty(0, dtype=int)
		idx2 = np.empty(0, dtype=int)

	if join_type == 'outer':
		# Add rows from table 1 that have no match in table 2
		# have them nominally link to row 0 of table 2, but note their status in isnull column
		i = np.arange(len(id1))
		i[idx1] = -1
		i = i[i != -1]
		idx1 = np.concatenate((idx1, i))
		idx2 = np.concatenate((idx2, np.zeros(len(i), int)))
		isnull = np.zeros(len(idx1), dtype=np.dtype('bool'))
		isnull[len(idx1)-len(i):] = True
	else:
		isnull = np.zeros(len(idx1), dtype=np.dtype('bool'))

	# Sort by idx1, to have all idx1 rows appear consecutively
	i = idx1.argsort()
	idx1 = idx1[i]
	idx2 = idx2[i]
	isnull = isnull[i]

#	print 'links:'
#	print id1[idx1]
#	print id2[idx2]
#	print isnull

	return (idx1, idx2, isnull)

def in_array(needles, haystack):
	""" Return a boolean array of len(needles) set to 
	    True for each needle that is found in the haystack.
	"""
	s = np.sort(haystack)
	i = np.searchsorted(s, needles)

	i[i == len(s)] = 0
	in_arr = s[i] == needles

	return in_arr

#def in_array(needles, haystack, return_indices=False):
#	i = haystack.argsort()
#	s = haystack[i]
#	k = np.searchsorted(s, needles)
#	k[l == len()] = 0
#
#	in_arr = s[i] == needles
#	
#	if return_indices:
#		hi = i[k]
#		hi[in_arr == False] = -1
#	else:
#		return in_arr

def tstart():
	return [ time.time() ]
	
def tick(s, t):
	tt = time.time()
	dt = tt - t[0]
	print >> sys.stderr, s, ":", dt
	t[0] = tt

class TableColsProxy:
	cat = None

	def __init__(self, cat):
		self.cat = cat

	def __getitem__(self, catname):
		# Return a list of columns in catalog catname
		if catname == '': return self.cat.all_columns()
		return self.cat.get_xmatched_catalog(catname).all_columns()

class ColDict:
	catalogs = None		# Cache of loaded catalogs and tables
	columns  = None		# Cache of already referenced columns
	cell_id  = None		# cell_id on which we're operating
	
	primary_catalog = None	# the primary catalog
	include_cached = None	# whether we should include the cached data within the cell

	orig_rows= None		# Debugging/sanity checking: dict of catname->number_of_rows that any tablet of this catalog correctly fetched with fetch_cell() should have

	def __init__(self, query, cat, cell_id, bounds, include_cached):

		self.cell_id = cell_id
		self.columns = {}
		self.primary_catalog = cat.name
		self.include_cached = include_cached

		# parse query
		(select_clause, where_clause, from_clause) = qp.parse(query, TableColsProxy(cat))
		#print (query, select_clause, where_clause, from_clause)
		#exit()

		# Fetch all rows of the base table, including the cached ones (if requested)
		rows2 = cat.fetch_cell(cell_id=cell_id, table=cat.primary_table, include_cached=include_cached)
		idx2  = np.arange(len(rows2))
		self.orig_rows = { cat.name: len(rows2) }

		# Reject objects out of bounds
		if bounds != None and len(rows2):
			raKey, decKey = cat.tables[cat.primary_table]["spatial_keys"]
			ra, dec = rows2[raKey], rows2[decKey]

			(x, y) = bhpix.proj_bhealpix(ra, dec)
			in_ = np.fromiter( (bounds.isInside(px, py) for (px, py) in izip(x, y)), dtype=np.bool)

			idx2  = idx2[in_]

		# Initialize catalogs and table lists, plus the array of primary keys
		self.catalogs = \
		{
			cat.name:
			{
				'cat':    cat,
				'join' :  (idx2, np.zeros(len(idx2), dtype=bool)),
				'tables':
				{
					cat.primary_table: rows2
				}
			}
		}
		self.keys     = rows2[idx2][cat.tables[cat.primary_table]["primary_key"]]

		# Load catalog indices to be joined
		for (catname, join_type) in from_clause:
			if cat.name == catname:
				continue
			assert catname not in self.catalogs, "Same catalog, '%s', listed twice in XMATCH clause" % catname;

			assert include_cached == False, "include_cached=True in JOINs is a recipe for disaster. Don't do it"

			# load the xmatch table and instantiate the second catalog object
			(m1, m2) = cat._fetch_xmatches(cell_id, self.keys, catname)
			cat2     = cat.get_xmatched_catalog(catname)
			rows2    = cat2.fetch_cell(cell_id=cell_id, table=cat2.primary_table, include_cached=True)
			id2      = rows2[cat2.tables[cat2.primary_table]["primary_key"]]
			self.orig_rows[cat2.name] = len(rows2)

			#print len(m1), len(m2)
			#print len(self.keys), len(rows2)

			# Join the tables (jmap and rows2), using (m1, m2) linkage information
			table_join.cell_id = cell_id	# debugging
			table_join.cat = cat		# debugging
			(idx1, idx2, isnull) = table_join(self.keys, id2, m1, m2, join_type=join_type)

			# update the keys and index maps for already joined catalogs
			self.keys = self.keys[idx1]
			for (catname, v) in self.catalogs.iteritems():
				(idx, isnull_) = v['join']
				v['join']      = idx[idx1], isnull_[idx1]

			#print "XX:", len(self.keys), len(rows2), idx1, idx2, isnull

			# add the newly joined catalog
			self.catalogs[cat2.name] = \
			{
				'cat': 		cat2,
				'join':		(np.arange(len(rows2)+1)[idx2], isnull),	# The +1 is for outer joins where len(rows2) = 0, but table_join returns idx2=0 (with isnull=True)
				'tables':
				{
					cat2.primary_table: rows2
				}
			}
			assert len(self.catalogs[cat2.name]['join'][0]) == len(self.keys)

		# Filter out only the rows remaining after the JOIN in the cached tables
		for catname in self.catalogs:
			tables = self.catalogs[catname]['tables']
			for table, rows in tables.iteritems():
				#if(len(self.keys)): print 'Before(%s.%s): %d' % (catname, table, len(rows))
				tables[table] = self._filter_joined(rows, catname)
				#if(len(self.keys)): print 'After(%s.%s): %d' % (catname, table, len(tables[table]))

		#if(len(self.keys)):
		#	print "Nonzero match!", len(self.keys)
		#	exit()

		# eval individual columns in select clause to slurp them up from disk
		# and have them ready for the where clause
		self.dtype = []
		nrows = 0
		for (asname, name) in select_clause:
			col = eval(name, {}, self)
			self[asname] = col
			self.dtype += [ (asname, str(col.dtype)) ]
			nrows = len(self[asname])

		# eval the WHERE clause, to obtain the final filter
		self.in_    = np.empty(nrows, dtype=bool)
		self.in_[:] = eval(where_clause, {}, self)
		#print self.dtype
		#exit()

	def rows(self):
		# Extract out the filtered rows
		self.t0 = time.time()
		rows = np.empty(sum(self.in_), dtype=np.dtype(self.dtype))
		for name, _ in self.dtype:
			col = self[name][self.in_]
			rows[name] = col
		##rows = np.empty(len(self.in_), dtype=np.dtype(self.dtype))
		##for name, _ in self.dtype:
		##	rows[name] = self[name]
		##rows = rows[self.in_]
		##print 'Loaded %d rows in %f sec (%d columns).' % (len(rows), time.time() - self.t0, len(rows.dtype.names))
		
		#if len(rows):
		#	(ra1, dec1, ra2, dec2, id2) = utils.as_columns(rows)
		#	rows = rows[id2 != 0]
		#	#d = gc_dist(ra1, dec1, ra2, dec2)*3600
		#	#at = 0
		#	#for i in xrange(len(d)):
		#	#	if id2[i] == 0: continue
		#	#	assert d[i] < 1, "Distance > 1arcsec: d=%f, row=%s" % (d[i], str(rows[i]))
		#	#	#print rows[i], d[i]
		#	#	#at = at + 1
		#	#	#if at == 10: exit()
		return rows

	def _filter_joined(self, rows, catname):
		# Join
		cat = self.catalogs[catname]['cat']
		idx, isnull = self.catalogs[catname]['join']
		if len(rows) == 0:
			rows = np.zeros(1, dtype=rows.dtype)
		rows = rows[idx]
		for name in rows.dtype.names:
			rows[name][isnull] = cat.NULL
		return rows

	def load_column(self, name, table, catname):
		# Load the column from table 'table' of the catalog 'catname'
		# Also cache the loaded tablet, for future reuse

		# See if we have already loaded the required tablet
		if table in self.catalogs[catname]['tables']:
			return self.catalogs[catname]['tables'][table][name]

		# Load
		cat = self.catalogs[catname]['cat']
		include_cached = self.include_cached if catname == self.primary_catalog else True
		rows = cat.fetch_cell(cell_id=self.cell_id, table=table, include_cached=include_cached)
		assert len(rows) == self.orig_rows[catname]

		# Join
		rows = self._filter_joined(rows, catname)

		# Cache
		self.catalogs[catname]['tables'][table] = rows

		# return the requested column (further caching is the responsibility of the caller)
		return rows[name]

	def __getitem__(self, name):
		# An already loaded column?
		if name in self.columns:
			return self.columns[name]

		# A yet unloaded column? Try to find it in tables of joined catalogs
		# May be prefixed by catalog name, in which case we force lookup of only
		#     that catalog
		if name.find('.') == -1:
			cats = self.catalogs
			colname = name
		else:
			(catname, colname) = name.split('.')
			cats = { catname: self.catalogs[catname] }
		for (catname, v) in cats.iteritems():
			cat = v['cat']
			for (table, schema) in cat.tables.iteritems():
				columns = set(( name for name, _ in schema['columns'] ))
				if colname in columns:
					self[name] = self.load_column(colname, table, catname)
					#print "Loaded column %s.%s.%s for %s (len=%s)" % (catname, table, colname, name, len(self.columns[name]))
					return self.columns[name]

		# A name of a catalog? Return a proxy object
		if name in self.catalogs:
			return CatProxy(self, name)

		# This object is unknown to us -- let it fall through, it may
		# be a global/Python variable or function
		raise KeyError(name)

	def __setitem__(self, key, val):
		if len(self.columns):
			assert len(val) == len(self.columns.values()[0]), "%s: %d != %d" % (key, len(val), len(self.columns.values()[0]))

		self.columns[key] = val

class CatProxy:
	coldict = None
	prefix = None

	def __init__(self, coldict, prefix):
		self.coldict = coldict
		self.prefix = prefix

	def __getattr__(self, name):
		return self.coldict[self.prefix + '.' + name]

def _mapper(partspec, mapper, cat, query, include_cached, mapper_args):
	(cell_id, bounds) = partspec

	# pass on some of the internals to the mapper
	mapper.CELL_ID = cell_id
	mapper.CATALOG = cat
	mapper.BOUNDS = bounds

	# Load, join, select
	rows = ColDict(query, cat, cell_id, bounds, include_cached).rows()

	# Pass on to mapper, unless empty
	if len(rows) != 0:
		result = mapper(rows, *mapper_args)
	else:
		# Catalog.map_reduce will not pass this back to the user (or to reduce)
		result = Empty

	return result

###################################################################
## Auxilliary functions implementing Catalog.build_neighbor_cache
## functionallity
def _cache_maker_mapper(rows, margin_x, margin_t):
	# Map: fetch all objects to be mapped, return them keyed
	# by cell ID and table
	self         = _cache_maker_mapper
	cat          = self.CATALOG
	cell_id      = self.CELL_ID

	p, t = cat.cell_bounds(cell_id)

	# Find all objects within 'margin_x' from the cell pixel edge
	# The pixel can be a rectangle, or a triangle, so we have to
	# handle both situations correctly.
	(x1, x2, y1, y2) = p.boundingBox()
	d = x2 - x1
	(cx, cy) = p.center()
	stop = 0
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
	(ra, dec) = utils.as_columns(rows)
	(x, y) = bhpix.proj_bhealpix(ra, dec)
	in_ = np.fromiter( (not p.isInside(px, py) for (px, py) in izip(x, y)), dtype=np.bool)
	if margin_t != 0:
		tcol = utils.as_columns(rows, 2)
		in_ &= np.fromiter( ( 0. < fabs(pt - t) - 0.5*dt < margin_t for pt in tcol ), dtype=np.bool)

	if not in_.any():
		return Empty

	# Now load all tablets, and keep only the neighbors within
	# neighborhood
	rows = {}
	for table in cat.tables:
		rows[table] = cat.fetch_cell(cell_id=cell_id, table=table)[in_]

	# Mark these to be replicated all over the neighborhood
	res = []
	if len(rows):
		for neighbor in cat.neighboring_cells(cell_id):
			res.append( (neighbor, rows) )

	#print "Scanned margins of %s (%d objects)" % (cat._tablet_file(self.CELL_ID, table=cat.primary_table), len(rows[cat.primary_table]))

	return res

def _cache_maker_reducer(cell_id, nborblocks):
	# Reduce: the key is the cell ID, the value is
	# a list of objects to be copied there.
	# 1. copy all existing non-cached objects to a temporary table
	# 2. append cached objects
	# 3. remove the original table
	# 4. rename the cached table
	self = _cache_maker_reducer
	cat          = self.CATALOG

	assert cat.is_cell_local(cell_id)

	# Update all tables
	ncached = 0
	with cat.get_cell(cell_id, mode='w') as cell:
		for table, schema in cat.tables.iteritems():
			if 'cached_flag' in schema:
				cachedFlag = schema['cached_flag']
			else:
				cachedFlag = None

			with cell.open(table=table) as fp:
				# Drop existing cached table, create an empty one
				if 'cached' in fp.root:
					fp.root.cached.remove()
				fp.root.table.copy('/', 'cached', start=0, stop=0)

				# Append cached rows
				for nbor in nborblocks:
					newrows = nbor[table]
					if cachedFlag:
						newrows[cachedFlag] = True
					fp.root.cached.append(newrows)

				# sanity
				if ncached == 0:
					ncached = fp.root.cached.nrows
				assert ncached == fp.root.cached.nrows

	# Return the number of new rows cached into this cell
	return (cell_id, ncached)
###################################################################

# Refresh neighbor cache
if __name__ == "x__main__":
	cat = Catalog('sdss')
	cat.build_neighbor_cache()

	exit()

# MapReduce examples
if __name__ == "__main__":
	cat = Catalog('sdss')

	# Simple mapper, counts the number of objects in each file
#	ntotal = 0
#	for (file, nobjects) in cat.map_reduce(ls_mapper, include_cached=False, nworkers=4):
#		ntotal = ntotal + nobjects
#		print file, nobjects
#	print "Total of %d objects in catalog." % ntotal

	# Computes the histogram of counts vs. declination
#	for (k, v) in sorted(cat.map_reduce(deccount_mapper, deccount_reducer)):
#		print k, v

	# Computes and plots the sky coverage at a given resolution
	sky_coverage = coverage(dx=0.25)
	pyfits.writeto('foot.fits', sky_coverage.astype(float).transpose()[::-1,], clobber=True)

	exit()


if __name__ == "x__main__":

	#importDVO('ps1', '/raid14/panstarrs/dvo-201008');
	#importSDSS('sdss', '/raid14/sweeps/sdss3/2009-11-16.v2/301/');
	importSDSS('sdss', '/data/sdss/sdss3/2009-11-16.v2/301/');
	exit()

	cat = Catalog('sdss')
	n = 0;
	sky = np.zeros((360,180))
	allsky = Polygon.Polygon([(1,1),(-1,1),(-1,-1),(1,-1)])
	foot = footprint.rectangle(0, -80, 360, 90, coordsys='gal')
	foot = allsky

#	for (ra, dec) in cat.select('ra dec', foot, testbounds=True):
#		n = n + 1
#		sky[int(ra), int(90-dec)] += 1

	###mr = MapReduce_Coverage()
	###cat.mapreduce(test_map, mr.reduce, foot, testbounds=True)
	###n = mr.sum;
	###sky = mr.sky

	test_reduce.sky = None
	cat.mapreduce(test_map, test_reduce, foot, testbounds=True, mapargs=(0.5,))
	sky = test_reduce.sky
	n = sky.sum()

	#from PIL import Image
	#img = Image.fromarray(sky.astype(np.int32));
	#img.save('foot.png')
	pyfits.writeto('foot.fits', sky.astype(float).transpose()[::-1,], clobber=True)

	print 'rows=', n
	#plt.imshow(sky.transpose(), interpolation='nearest');
	#plt.show();

	exit()

