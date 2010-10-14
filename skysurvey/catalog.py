#!/usr/bin/env python

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
import pool2
import os, errno, glob
import json
import fcntl
import Polygon.IO
import utils
import footprint
from utils import astype

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

	def _get_cols(self, cols):
		""" Return a list of columns, given all permissible
		    user-friendly formats that cols can be in.
		    
		    Examples:
		    	'ra dec u g r i z'	-- just those columns
		    	'*'			-- all available columns
		"""
		if cols == '*':		cols = All
		if type(cols) == str:	cols = cols.split()
		return cols

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

	def fetch_cell(self, cell_id, where=None, filter=None, filter_args=(), table_type='catalog', include_cached=False):
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

		return rows

	def fetch(self, cols=All, foot=All, where=None, testbounds=True, include_cached=False, nworkers=None, progress_callback=None, filter=None, filter_args=()):
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

		cols = self._get_cols(cols)
		files = self._get_cells(foot)

		ret = None
		for rows in self.map_reduce(_iterate_mapper, mapper_args=(filter, filter_args), foot=foot, where=where, testbounds=testbounds, include_cached=include_cached, nworkers=nworkers, progress_callback=progress_callback):
			if cols == All:
				cols = rows.dtype.names

			# ensure enough memory has been allocated (and do it
			# intelligently if not)
			if ret == None:
				rcols = [ (col, rows.dtype[col].str) for col in cols ]
				ret = np.empty(len(rows), np.dtype(rcols))
				nret = 0

			while len(ret) < nret + len(rows):
				ret.resize(2*(len(ret)+1))
				#print "Resizing to", len(ret)

			# store
			for col in cols:
				ret[col][nret:nret+len(rows)] = rows[col]
			nret = nret + len(rows)

		ret.resize(nret)
		#print "Resizing to", len(ret)
		return ret

	def iterate(self, cols=All, foot=All, where=None, testbounds=True, return_array=False, include_cached=False, nworkers=None, progress_callback=None, filter=None, filter_args=()):
		""" Yield rows (either on a row-by-row basis if return_array==False
		    or in chunks (numpy structured array)) within a
		    given footprint. Calls 'filter' callable (if given) to filter
		    the returned rows.

		    See the documentation for Catalog.fetch for discussion of
		    'filter' callable.
		"""

		cols = self._get_cols(cols)
		files = self._get_cells(foot)

		for rows in self.map_reduce(_iterate_mapper, mapper_args=(filter, filter_args), foot=foot, where=where, testbounds=testbounds, include_cached=include_cached, nworkers=nworkers, progress_callback=progress_callback):
			if return_array:
				yield rows
			else:
				if cols == All:
					cols = rows.dtype.names

				for row in izip(*[rows[col] for col in cols]):
					yield row

	def map_reduce(self, mapper, reducer=None, foot=All, where=None, testbounds=True, include_cached=False, mapper_args=(), reducer_args=(), nworkers=None, progress_callback=None):
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
					mapper_args = (mapper, where, self, include_cached, mapper_args),
					progress_callback = progress_callback):
				yield result
		else:
			for result in pool.imap_reduce(
					partspecs, _mapper, _reducer,
					mapper_args  = (mapper, where, self, include_cached, mapper_args),
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

def _mapper(partspec, mapper, where, cat, include_cached, mapper_args):
	(cell_id, bounds) = partspec

	# pass on some of the internals to the mapper
	mapper.CELL_ID = cell_id
	mapper.CATALOG = cat
	mapper.BOUNDS = bounds
	mapper.WHERE = where
	mapper.CELL_FN = cat._file_for_id(cell_id)

	# Fetch all rows
	rows = cat.fetch_cell(cell_id=cell_id, where=where, table_type='catalog', include_cached=include_cached)

	# Reject objects out of bounds
	if bounds != None and len(rows):
		(x, y) = bhpix.proj_bhealpix(rows['ra'], rows['dec'])
		in_ = np.fromiter( (bounds.isInside(px, py) for (px, py) in izip(x, y)), dtype=np.bool)
		rows = rows[in_]

	result = mapper(rows, *mapper_args)

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

