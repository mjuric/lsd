#!/usr/bin/env python

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
from utils import astype

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
	columns = []
	t0 = 47892		# default starting epoch (MJD, 47892 == Jan 1st 1990)
	dt = 90			# default temporal resolution (in days)
	__nrows = 0

	NULL = 0		# The value for NULL in JOINed rows that had no matches

	def cell_for_id(self, id):
		# Note: there's a more direct way of doing this,
		#       but I don't have time to think about it right now
		x, y, t, _ = self.unpack_id(id)
		cell_id = self._id_from_xy(x, y, t, self.level)
		return cell_id

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

	def _file_for_id(self, id, table_type='catalog'):
		(x, y, t, rank) = self.unpack_id(id, self.level)

		if t >= self.t0 + self.dt:
			fn = '%s/%s.%s.MJD%05d%+d.h5' % (self.path, bhpix.get_path(x, y, self.level), table_type, t, self.dt)
		else:
			fn = '%s/%s.%s.static.h5' % (self.path, bhpix.get_path(x, y, self.level), table_type)
		return fn

	def cell_exists(self, cell_id, table_type='catalog'):
		fn = self._file_for_id(cell_id, table_type)
		return os.access(fn, os.R_OK)

	def _load_dbinfo(self):
		data = json.loads(file(self.path + '/dbinfo.json').read())

		# Explicit type coercion for security reasons
		self.level = int(data["level"])
		self.t0 = float(data["t0"])
		self.dt = float(data["dt"])
		self.__nrows = float(data["nrows"])
		self.columns = []
		for (col, dtype) in data["columns"]:
			self.columns.append((str(col), str(dtype)))

	def _store_dbinfo(self):
		data = dict()
		data["level"], data["t0"], data["dt"] = self.level, self.t0, self.dt
		data["nrows"] = self.__nrows
		data["columns"] = self.columns

		f = open(self.path + '/dbinfo.json', 'w')
		f.write(json.dumps(data, indent=4, sort_keys=True))
		f.close()

	def _open_cell(self, id, mode='r', retries=-1, table_type='catalog'):
		""" Open a given cell in read or write mode. If in
		    write mode, the cell will be locked. Do not use this
		    function directly; use cell() instead.

		    Returns:
		    	Tuple of (fp, lock_fn) where lock_fn is the lockfile
		    	or None if mode='r'
		"""
		fn = self._file_for_id(id, table_type)

		if mode == 'r':
			return (tables.openFile(fn), None)
		elif mode == 'w':
			if not os.path.isfile(fn):
				# create directory if needed
				path = fn[:fn.rfind('/')];
				if not os.path.exists(path):
					utils.mkdir_p(path)

				utils.shell('/usr/bin/lockfile -1 -r%d "%s.lock"' % (retries, fn) )

				# intialize the file
				table_type = table_type.split('_')[0]
				if table_type == 'catalog':
					fp  = tables.openFile(fn, mode='w', title='SkysurveyDB')
					fp.createTable('/', 'catalog', np.dtype(self.columns), "Catalog", expectedrows=20*1000*1000)
					fp.createArray('/', 'id_seq', np.ones(1, dtype=np.uint32), 'A sequence for catalog table ID')
				elif table_type == 'xmatch':
					fp  = tables.openFile(fn, mode='w', title='Skysurvey xmatch table')
					fp.createTable('/', 'xmatch', np.dtype([('id1', 'u8'), ('id2', 'u8'), ('dist', 'f4')]), "xmatch", expectedrows=20*1000*1000)
				else:
					raise Exception('Unknown table type!')
			else:
				# open for appending
				utils.shell('/usr/bin/lockfile "%s.lock"' % fn)

				fp = tables.openFile(fn, mode='a')

			return (fp, fn + '.lock')
		else:
			raise Exception("Mode must be one of 'r' or 'w'")

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
		for fn in glob.iglob(prefix + ".catalog.*.h5"):
			if(foot.area() == box.area()):
				foot = None

			# parse out the time, construct cell ID
			t = fn.split('.')[-2]
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
		""" 
		    Parse the column specification and return a dictionary
		    of (table -> [col1 col2 ...]), where the main table will
		    be keyed by an empty string ('').

		    Examples:
		    	'ra dec u g r i z'	-- just those columns
		    	'*'			-- all available columns
		    	'ra dec g r sdss.g sdss.r' -- xmatch w. sdss table

		    Return:
		 	- dtype -- tuple suitable for passing to np.dtype()
			- tables: dictionary(table->columns)
				columns: dict(colname_in_source_table->colname_in_output_table)
		"""

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
	def __init__(self, path, mode='r', columns=None, level=Automatic, t0=Automatic, dt=Automatic):
		self.path = path

		if mode == 'c':
			self.create(columns, level, t0, dt)
		else:
			self._load_dbinfo()

	def create(self, columns, level, t0, dt):
		""" Create a new catalog and store its definition.
		"""
		utils.mkdir_p(self.path)
		if os.path.isfile(self.path + '/dbinfo.json'):
			raise Exception("Creating a new catalog in '%s' would overwrite an existing one." % self.path)
			
		self.columns = columns
		if level != Automatic: self.level = level
		if    t0 != Automatic: self.t0 = t0
		if    dt != Automatic: self.dt = dt

		self._store_dbinfo()

	def insert(self, rows, ra, dec, t = None):
		""" Insert a set of rows into the database. Protects against multiple
		    writers simultaneously inserting into the same file

		    The rows being inserted must NOT contain the index column.
		"""
		ids = self.id_from_pos(ra, dec, t)
		cells = self.id_from_pos(ra, dec, t, self.level)
		unique_cells = list(set(cells))

		ntot = 0
		while unique_cells:
			# Find a cell that is ready to be written to (that isn't locked
			# by another writer)
			for k in xrange(3600):
				try:
					i = k % len(unique_cells)
					cell = unique_cells[i]
					(fp, lockfile)  = self._open_cell(cell, 'w', retries=0, table_type='catalog')
					unique_cells.pop(i)
					break
				except subprocess.CalledProcessError as err:
					#print err
					pass
			else:
				raise Exception('Appear to be stuck on a lock file!')

			t   = fp.root.catalog
			id2 = fp.root.id_seq[0]

			# Extract and store the subset of rows that belong into this cell
			iit = iter(xrange(len(rows) + 1))
			rows2 = [ (ids[i] + id2 + np.uint64(next(iit)),) + rows[i] for i in xrange(len(rows)) if(cells[i] == cell) ]

			t.append(rows2)
			fp.root.id_seq[0] += len(rows2)
			fp.close()
			os.unlink(lockfile)

			##print '[', len(rows2), ']'
			self.__nrows = self.__nrows + len(rows2)
			ntot = ntot + len(rows2)

		if ntot != len(ids):
			print 'ntot=', ntot
			raise Exception('**** Bug detected ****')

		return ids

	def nrows(self):
		return self.__nrows

	def close(self):
		pass

	def __str__(self):
		""" Return some basic (human readable) information about the
		    catalog.
		"""
		i =     'Path:         %s\n' % self.path
		i = i + 'Partitioning: level=%d\n' % (self.level)
		i = i + '(t0, dt):     %f, %f \n' % (self.t0, self.dt)
		i = i + 'Objects:      %d\n' % (self.nrows())
		i = i + '\n'
		s = "%20s %10s\n" % ('Column', 'Type')
		s = s + '-'*(len(s)-1) + '\n'
		for col in self.columns:
			s = s + "%20s %10s\n" % (col[0], col[1])
		return i + s

	def fetch_cell(self, cell_id, where=None, filter=None, filter_args=(), table_type='catalog', include_cached=False, return_id=False):
		""" Load and return all rows from a given cell, possibly
		    filtering them using filter (if given)
		"""
		with self.cell(cell_id, table_type=table_type) as fp:
			table_type = table_type.split('_')[0]
			table = fp.root.__getattr__(table_type)

			if where == None:
				rows = table.read()
			else:
				rows = table.readWhere(where)

		# Reject cached objects, unless requested otherwise
		if not include_cached and len(rows) and "cached" in rows.dtype.names:
			rows = rows[rows["cached"] == 0]

		# Custom user-supplied filter
		if filter != None:
			rows = filter(rows, *filter_args)

		if return_id:
			return rows, np.array(rows['id'])
		else:
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

	def map_reduce(self, mapper, reducer=None, cols=All, foot=All, where=None, testbounds=True, include_cached=False, join_type='outer', mapper_args=(), reducer_args=(), nworkers=None, progress_callback=None):
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
		# parse the column/query specification
		queryspec = self._prep_query(cols)

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
					mapper_args = (mapper, where, self, include_cached, queryspec, join_type, mapper_args),
					progress_callback = progress_callback):

				if type(result) != type(Empty):
					yield result
		else:
			for result in pool.imap_reduce(
					partspecs, _mapper, _reducer,
					mapper_args  = (mapper, where, self, include_cached, queryspec, join_type, mapper_args),
					reducer_args = (reducer, self, reducer_args),
					progress_callback = progress_callback):
				yield result

	@contextmanager
	def cell(self, cell_id, mode='r', retries=-1, table_type='catalog'):
		""" Open and return a pytables object for the given cell.
		    If mode is not 'r', the cell table will be locked
		    for the duration of this context manager, and automatically
		    unlocked upon exit from it.
		"""
		(fp, lockfile) = self._open_cell(cell_id, mode, retries, table_type=table_type)

		yield fp

		fp.close()
		if lockfile != None:
			os.unlink(lockfile)

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

		ntotal = 0
		ncells = 0
		for (cell_id, ncached) in self.map_reduce(_cache_maker_mapper, _cache_maker_reducer, mapper_args=(margin_x, margin_t)):
			ntotal = ntotal + ncached
			ncells = ncells + 1
			print self._file_for_id(cell_id), ": ", ncached, " cached objects"
		print "Total %d cached objects in %d cells" % (ntotal, ncells)

	def compute_summary_stats(self):
		""" Compute frequently used summary statistics and
		    store them into the dbinfo file. This should be called
		    to refresh the stats after insertions.
		"""
		from tasks import compute_counts
		self.__nrows = compute_counts(self)
		self._store_dbinfo()

	def _fetch_xmatches(self, cell_id, ids, cat_to_name):
		"""
			Return a list of crossmatches corresponding to ids
		"""
		table_type = "xmatch_" + cat_to_name

		if len(ids) == 0 or not self.cell_exists(cell_id, table_type=table_type):
			return ([], [])

		rows = self.fetch_cell(cell_id, table_type=table_type)

		# drop all links where id1 is not in ids
		sids = np.sort(ids)
		res = np.searchsorted(sids, rows['id1'])
		res[res == len(sids)] = 0
		ok = sids[res] == rows['id1']
		rows = rows[ok]

		return (rows['id1'], rows['id2'])

	def get_xmatched_catalog(self, cat_to_name):
		# TODO: record the path of the catalog do dbinfo and get it from there
		return Catalog(cat_to_name)

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

def _mapper(partspec, mapper, where, cat, include_cached, queryspec, join_type, mapper_args):
	(cell_id, bounds) = partspec

	# pass on some of the internals to the mapper
	mapper.CELL_ID = cell_id
	mapper.CATALOG = cat
	mapper.BOUNDS = bounds
	mapper.WHERE = where
	mapper.CELL_FN = cat._file_for_id(cell_id)

	# Fetch all rows of the base table
	rows2, id = cat.fetch_cell(cell_id=cell_id, where=where, table_type='catalog', include_cached=include_cached, return_id=True)

	# Reject objects out of bounds
	if bounds != None and len(rows2):
		(x, y) = bhpix.proj_bhealpix(rows2['ra'], rows2['dec'])
		in_ = np.fromiter( (bounds.isInside(px, py) for (px, py) in izip(x, y)), dtype=np.bool)
		rows2 = rows2[in_]
		id = id[in_]

	# Unpack queryspec, create output table, extract only the requested rows
	(dtypespec, colspec) = queryspec
	dtype = np.dtype(dtypespec)
	if rows2.dtype == dtype:
		rows = rows2
	else:
		rows = np.empty(len(rows2), dtype=dtype)
		for (col, rcol) in colspec[''].iteritems():
			rows[rcol] = rows2[col]

	# Perform any JOINs, table by table
	for (table2, cols2) in colspec.iteritems():
		if table2 == '': continue

		# load the xmatch table and instantiate second catalog object
		(m1, m2) = cat._fetch_xmatches(cell_id, id, table2)

		if len(m1) != 0:
			# Load xmatched rows.
			# get the list of cells containing xmatch link targets
			# sort them in descending order by the number of targets
			cat2   = cat.get_xmatched_catalog(table2)
			cells2 = cat2.cell_for_id(m2)
			ucells2 = np.unique1d(cells2)
			count = np.empty_like(ucells2)
			for (i, ucell) in enumerate(ucells2):
				count[i] = sum(cells2 == ucell)
			i = count.argsort()[::-1]; count = count[i]; ucells2 = ucells2[i]

			# Load data from all cells until all cells are exhausted or all
			# indices in m2 are found
			rows2 = None
			mfind = m2
			for cell_id2 in ucells2:
				rows_tmp, id_tmp = cat2.fetch_cell(cell_id2, include_cached=True, return_id=True)

				# Extract the rows that may be matched
				inarr = in_array(id_tmp, mfind)
				if rows2 == None:
					rows2 = rows_tmp[inarr]
					id2   =   id_tmp[inarr]
				else:
					rows2 = np.append(rows2, rows_tmp[inarr])
					id2   = np.append(id2,     id_tmp[inarr])

				# Remove the found indices from the list of indices to look for
				mfound = in_array(mfind, id_tmp)
				i = np.arange(len(mfind)); i = i[mfound == False]
				mfind = mfind[i]
				if len(mfind) == 0:
					break
				print "len(mfind): ", len(mfind)
		else:
			id2 = []

		# Join the two tables (rows and rows2), using (m1, m2) linkage information
		table_join.cell_id = cell_id
		table_join.cat = cat
		(idx1, idx2, isnull) = table_join(id, id2, m1, m2, join_type=join_type)
		nrows = len(idx1)

		if False:
			print "XXX: table2=", table2, "cols2=", cols2
			print "m1: ", len(m1)
			print "cells2: ", len(cells2)
			print "rows:", len(rows)
			print "rows2:", len(rows2)
			print "idx1:", len(idx1)
			print "isnull==0", len(isnull[isnull==0])
			print idx1, idx2, isnull
			print idx1.dtype
			np.savetxt('match.txt', np.transpose((m1, m2)), fmt='%d')
			np.savetxt('id1.txt', id, fmt='%d')
			np.savetxt('id2.txt', id2, fmt='%d')
			exit()

		# Join the old and the new table
		rows = rows[idx1]
		id   = id[idx1]
		for (col, rcol) in cols2.iteritems():
			rows[rcol]         = rows2[col][idx2]
			rows[rcol][isnull] = cat.NULL

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
	# by cell ID

	if len(rows) == 0: return []

	self         = _cache_maker_mapper
	cat          = self.CATALOG
	cell_id = self.CELL_ID

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
	(x, y) = bhpix.proj_bhealpix(rows['ra'], rows['dec'])
	in_ = np.fromiter( (not p.isInside(px, py) for (px, py) in izip(x, y)), dtype=np.bool)
	if margin_t != 0:
		in_ &= np.fromiter( ( 0. < fabs(pt - t) - 0.5*dt < margin_t for pt in rows["t"] ), dtype=np.bool)
	rows = rows[in_]

	res = []
	if len(rows):
		for neighbor in cat.neighboring_cells(cell_id):
			res.append( (neighbor, rows) )

	print "Scanned margins of", self.CELL_FN

	return res

def _cache_maker_reducer(cell_id, newrowblocks):
	# Reduce: the key is the cell ID, the value is
	# a list of objects to be copied there.
	# 1. copy all existing non-cached objects to a temporary table
	# 2. append cached objects
	# 3. remove the original table
	# 4. rename the cached table
	self = _cache_maker_reducer
	cat          = self.CATALOG

	assert cat.is_cell_local(cell_id)

	ncached = 0
	with cat.cell(cell_id, mode='w') as fp:
		#fp.createTable('/', 'catalog_tmp', np.dtype(cat.columns), "Catalog", expectedrows=20*1000*1000)

		rows = fp.root.catalog.read();
		rows = rows[rows["cached"] == 0]

		fp.root.catalog.copy('/', 'catalog_tmp', start=0, stop=0)
		fp.root.catalog_tmp.append(rows)

		for newrows in newrowblocks:
			newrows["cached"] = True
			fp.root.catalog_tmp.append(newrows)
			ncached = ncached + len(newrows)

		fp.flush()
		fp.root.catalog.remove()
		fp.root.catalog_tmp.rename('catalog')

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

