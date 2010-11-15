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
from table import Table
from utils import astype, gc_dist, unpack_callable, full_dtype, is_scalar_of_type, isiterable
from intervalset import intervalset
from StringIO import StringIO

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
	level = 6
	t0 = 54335		# default starting epoch (== 2pm HST, Aug 22 2007 (night of GPC1 first light))
	dt = 1			# default temporal resolution (in days)
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
		if t is None:
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
		if dx is None:
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
			prefix = '%s/%s/T%05d/catalog' % (self.path, subpath, t)
		else:
			prefix = '%s/%s/static/catalog' % (self.path, subpath)

		return prefix

	def _tablet_file(self, cell_id, table):
		return "%s.%s.h5" % (self._cell_prefix(cell_id), table)

	def tablet_exists(self, cell_id, table=None):
		if table is None:
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

	## Cell enumeration routines
	def _get_subcells(self, static_cell, table, lev):
		x, y, t, _ = self.unpack_id(static_cell)
		assert t == self.t0	# Must be a static cell

		prefix = self.path + '/' + bhpix.get_path(x, y, lev)
		pattern = "%s/*/*.%s.h5" % (prefix, table)
		
		cells = []
		for fn in glob.iglob(pattern):
			# parse out the time, construct cell ID
			(kind, fname) = fn.split('/')[-2:]
			t = self.t0 if kind == 'static' else float(kind[1:])

			cell_id = self._id_from_xy(x, y, t, self.level)

			cells.append(cell_id)

		return cells

	def _get_cells_recursive(self, outcells, foot, times, tables, pix):
		""" Helper for get_cells(). See documentation of
		    get_cells() for usage
		"""
		# Check for nonzero overlap
		lev  = bhpix.get_pixel_level(pix[0], pix[1])
		dx   = bhpix.pix_size(lev)
		box  = self._cell_bounds_xy(pix[0], pix[1], dx)
		foot = foot & box
		if not foot:
			return


		# Check if the cell directory exists (give up if it doesn't)
		prefix = self.path + '/' + bhpix.get_path(pix[0], pix[1], lev)
		if not os.path.isdir(prefix):
			return

		# Get the cell_ids of existing leaf tablets for the given
		# tables. There can be more than one, if one of the tables
		# has a time dimension
		static_cell = self._id_from_xy(pix[0], pix[1], self.t0, self.level)
		cells = set(( cell for table in tables for cell in self._get_subcells(static_cell, table, lev) ))

		found = False
		if len(cells):
			# remove the static cell if any temporal ones exist. The static
			# cell will be implicitly be taken into account when resolving
			# joins.
			if len(cells) > 1 and static_cell in cells:
				cells.remove(static_cell)

			# Filter on time, add bounds
			xybounds = None if(foot.area() == box.area()) else foot
			for cell_id in cells:
				x, y, t, _ = self.unpack_id(cell_id)

				# Cut on the time component
				tival = intervalset((t, t+self.dt))
				tolap = times & tival
				if len(tolap):
					(l, r) = tolap[-1]				# Get the right-most interval component
					if l == r == t+self.dt:				# Is it a single point?
						tolap = intervalset(*tolap[:-1])	# Since objects in this cell have time in [t, t+dt), remove the t+dt point

				if len(tolap) == 0:					# No overlap between the intervals -- skip this cell
					continue;

				# Return None if the cell is fully contained in the requested interval
				tbounds = None if tival == tolap else tolap

				# Add to output
				if tival == tolap:
					outcells[cell_id][xybounds] = None
				elif xybounds not in outcells[cell_id]:
					outcells[cell_id][xybounds] = tbounds
				elif outcells[cell_id][xybounds] is not None:
					outcells[cell_id][xybounds] |= tbounds

				found = True

		if found:
			return

		# Recursively subdivide the four subpixels
		dx = dx / 2
		for d in np.array([(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]):
			self._get_cells_recursive(outcells, foot & box, times, tables, pix + dx*d)

	def __part_to_xy_t(self, part):
		foot = Polygon.Polygon()
		times = intervalset()
		for v in part:
			if   isinstance(v, Polygon.Polygon):
			  	foot |= v
			elif isinstance(v, intervalset):
				times |= v
			else:
				raise Exception('Incorrect part specification')

		if not foot:		foot =  footprint.ALLSKY
		if len(times) == 0:	times = intervalset((-np.inf, np.inf))

		# Restrict to valid sky
		foot = foot & footprint.ALLSKY

		return (foot, times)

	def get_cells(self, foot=All, return_bounds=False, tables=[]):
		""" Return a list of (cell_id, footprint) tuples completely
		    covering the requested footprint.

		    The footprint can either be a Polyon, a specific 
		    cell_id integer, a (t0, t1) time tuple, or an array
		    of these.

		    Output is a list of (cell_id, xybounds, tbounds) tuples,
		    unless return_bounds=False when the output is just a
		    list of cell_ids.
		"""
		# Handle all-sky requests, and scalars
		if foot == All:
			foots = [ footprint.ALLSKY ]
		elif not isinstance(foot, list):
			foots = [ foot ]
		else:
			foots = foot

		# ensure the primary table is always searched for
		tables = set(tables)
		tables.add(self.primary_table)

		# Now our input is a list that consists of one or more of:
		# a) Polygon instances
		# b) interval instances
		# c) tuples or lists of (Polygon, interval, Polygon, interval, ...)
		# e) cell_ids (integers)
		#
		# Do the following:
		# - extract a) and b) and form a single ([Polygon], interval), add it to the list
		# - append e) to the output list
		# - for each tuple:
		#	form a single ([Polygon], interval)
		# 	call _get_cell_recursive to obtain cell_ids, append them to output

		#print "***INPUT:", foots;

		# cell->foot->times
		cells = defaultdict(dict)
		part0 = []
		parts = []
		for v in foots:
			if   isinstance(v, Polygon.Polygon) or isinstance(v, intervalset):
				part0.append(v)
			elif isinstance(v, tuple) or isinstance(v, list):
				parts.append(v)
			else:
				cells[int(v)][None] = None	# Fetch entire cell (space and time)

		#print "****A1:", cells, part0, parts;
		if part0:
			parts.append(part0)

		#print "****PARTS:", parts

		# Find cells overlapping the requested spacetime
		if len(parts):
			for part in parts:
				foot, times = self.__part_to_xy_t(part)
				#print "HERE:", foot.area(), times;

				# Divide and conquer to find the cells covered by footprint
				self._get_cells_recursive(cells, foot, times, tables, (0., 0.))

		# Reorder cells to be an array of (cell, [(poly, time), (poly, time)] ...) tuples
		cells = dict(( (k, v.items()) for k, v in cells.iteritems() ))

		if False:
			for k, bounds in cells.iteritems():
				print k, ':', str(self.unpack_id(k)),
				for xy, t in bounds:
					print (xy.area() if xy is not None else None, t),
				print ''
			print len(cells)
			exit()

		if not return_bounds:
			return cells.keys()
		else:
			return cells

	def group_cells_by_spatial(self, cell_ids):
		""" Split the array of cell_ids into subarrays,
		    one per each static sky cell it belongs to
		"""
		cell_ids   = np.array(cell_ids)

		x, y, t, _ = self.unpack_id(cell_ids)
		cell_id_xy = self._id_from_xy(x, y, self.t0, self.level)

		ret = {}
		for cell_id in set(cell_id_xy):
			ret[cell_id] = cell_ids[cell_id_xy == cell_id]

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
		self.joined_catalogs = {}
		self.name = name

		if level != Automatic: self.level = level
		if    t0 != Automatic: self.t0 = t0
		if    dt != Automatic: self.dt = dt

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

	def static_if_no_temporal(self, cell_id, table):
		""" Try to locate tablet 'table' in cell_id. If there's
		    no such tablet, return a corresponding static sky
		    cell_id
		"""
		x, y, t, _ = self.unpack_id(cell_id)

		if t == self.t0:
			return cell_id
		
		if self.tablet_exists(cell_id, table):
			##print "Temporal cell found!", self._cell_prefix(cell_id)
			return cell_id

		# return corresponding static-sky cell
		cell_id = self._id_from_xy(x, y, self.t0, self.level)
		#print "Reverting to static sky", self._cell_prefix(cell_id)
		return cell_id

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

	def query_cell(self, cell_id, query='*', include_cached=False):
		""" Execute a query on a local cell.

		    If the cell_id has a temporal component, and there are no
		    tablets in that cell, a static sky cell corresponding
		    to it will be tried.
		"""
		assert self.is_cell_local(cell_id)

		return self.fetch(query, cell_id, include_cached=include_cached, progress_callback=pool2.progress_pass);

	def fetch(self, query='*', foot=All, include_cached=False, testbounds=True, nworkers=None, progress_callback=None, filter=None):
		""" Return a table (numpy structured array) of all rows within a
		    given footprint. Calls 'filter' callable (if given) to filter
		    the returned rows. Returns None if there are no rows to return.

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

		filter, filter_args = unpack_callable(filter)

		ret = None
		for rows in self.map_reduce(query=query, mapper=(_iterate_mapper, filter, filter_args), _pass_empty=True, foot=foot, testbounds=testbounds, include_cached=include_cached, nworkers=nworkers, progress_callback=progress_callback):
			# ensure enough memory has been allocated (and do it
			# intelligently if not)
			if ret is None:
				ret = rows
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

	def iterate(self, query='*', foot=All, include_cached=False, testbounds=True, nworkers=None, progress_callback=None, filter=None, return_blocks=False):
		""" Yield rows (either on a row-by-row basis if return_blocks==False
		    or in chunks (numpy structured array)) within a
		    given footprint. Calls 'filter' callable (if given) to filter
		    the returned rows.

		    See the documentation for Catalog.fetch for discussion of
		    'filter' callable.
		"""

		filter, filter_args = unpack_callable(filter)

		for rows in self.map_reduce(query, (_iterate_mapper, filter, filter_args), foot=foot, testbounds=testbounds, include_cached=include_cached, nworkers=nworkers, progress_callback=progress_callback):
			if return_blocks:
				yield rows
			else:
				for row in rows:
					yield row

	def map_reduce(self, query, mapper, reducer=None, foot=All, testbounds=True, include_cached=False, nworkers=None, progress_callback=None, _pass_empty=False):
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
		# Unpack mapper/reducer args
		mapper,   mapper_args = unpack_callable(mapper)
		reducer, reducer_args = unpack_callable(reducer)

		# Parse the query to obtain a list of catalogs we're going to be JOINing against
		# Then construct a list of join tables that get_cells() will look for
		(select_clause, where_clause, join_clause) = qp.parse(query, TableColsProxy(self))
		tables = set()
		for catname, _ in join_clause:
			if catname == self.name: continue
			v = self.joined_catalogs[catname]
			tables.update((v['table_from'], v['table_to']))

		# slice up the job down to individual cells
		partspecs = self.get_cells(foot, return_bounds=True, tables=tables).items()

		# tell _mapper not to test spacetime boundaries if the user requested so
		if not testbounds:
			partspecs = [ (part_id, None) for (part_id, _) in partspecs ]

		# start and run the workers
		pool = pool2.Pool(nworkers)
		if reducer is None:
			for result in pool.imap_unordered(
					partspecs, _mapper,
					mapper_args = (mapper, self, query, include_cached, mapper_args, _pass_empty),
					progress_callback = progress_callback):

				if type(result) != type(Empty):
					yield result
		else:
			for result in pool.imap_reduce(
					partspecs, _mapper, _reducer,
					mapper_args  = (mapper, self, query, include_cached, mapper_args, _pass_empty),
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

	def neighboring_cells(self, cell_id, include_self=False):
		""" Returns the cell IDs of cells spatially adjacent 
		    to cell_id.
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

		return nhood

	def is_cell_local(self, cell_id):
		""" Returns True if the cell is reachable from the
		    current machine. A placeholder for if/when I decide
		    to make this into a true distributed database.
		"""
		return True

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
		return self._get_schema(self.primary_table)["spatial_keys"]

	def get_primary_key(self):
		# Find out which columns are our spatial keys
		return self._get_schema(self.primary_table)["primary_key"]

	def fetch_joins(self, cell_id, ids, cat_to_name, fetch_tablet=None):
		"""
			Return a list of crossmatches corresponding to ids
		"""
		if fetch_tablet is None:
			fetch_tablet = self.fetch_tablet

		table_from  = self.joined_catalogs[cat_to_name]['table_from']
		table_to    = self.joined_catalogs[cat_to_name]['table_to']
		column_to   = self.joined_catalogs[cat_to_name]['column_to']
		column_from = self.joined_catalogs[cat_to_name]['column_from']

		# This allows joins from static to temporal catalogs where
		# the join table is in a static cell (not implemented yet). 
		# However, it also support an (implemented) case of tripple
		# joins of the form static-static-temporal, where the cell_id
		# will be temporal even when fetching the static-static join
		# table for the two other catalogs.
		cell_id_from = self.static_if_no_temporal(cell_id, table_from)
		cell_id_to   = self.static_if_no_temporal(cell_id, table_to)

		if len(ids) == 0 \
		   or not self.tablet_exists(cell_id_from, table_from) \
		   or not self.tablet_exists(cell_id_to,   table_to):
			return ([], [])

		if table_from == table_to:
			# Both the primary and foreign key maps are in the same table
			#  - this must be the case for many-to-many or one-to-many joins
			#  - it can be the case for one-to-one joins
			rows    = fetch_tablet(self, cell_id_from, table_from)
		else:
			# The primary and foreign key columns are in the same table
			#  - this can only be a one-to-one or many-to-one join
			#  - the number of rows in the two tables must be the same
			colFrom = fetch_tablet(self, cell_id_from, table_from)[column_from]
			rows  = np.empty(len(colFrom), dtype=[(column_from, 'u8'), (column_to, 'u8')])
			rows[column_from] = colFrom
			rows[column_to]   = fetch_tablet(self, cell_id_to, table_to)[column_to]

		# drop all links where id1 is not in ids
		sids = np.sort(ids)
		res = np.searchsorted(sids, rows[column_from])
		res[res == len(sids)] = 0
		ok = sids[res] == rows[column_from]

		return (rows[column_from][ok], rows[column_to][ok])

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

	def get_joined_catalog(self, catname):
		assert catname in self.joined_catalogs
		return Catalog(self.joined_catalogs[catname]['path'])

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
		return self.cat.get_joined_catalog(catname).all_columns()

class iarray(np.ndarray):
	""" Subclass of ndarray allowing per-row indexing.
	
	    Used by ColDict.load_column()
	"""
	def __call__(self, *args):
		"""
		   Apply numpy indexing on a per-row basis. A rough
		   equivalent of:
			
			self[ arange(len(self)) , *args]

		   where any tuples in args will be converted to a
		   corresponding slice, while integers and numpy
		   arrays will be passed in as-given. Any numpy array
		   given as index must be of len(self).

		   Simple example: assuming we have a chip_temp column
		   that was defined with 64f8, to select the temperature
		   of the chip corresponding to the observation, do:
		   
		   	chip_temp(chip_id)

		   Note: A tuple of the form (x,y,z) is will be
		   conveted to [x:y:z] slice. (x,y) converts to [x:y:]
		   and (x,) converts to [x::]
		"""
		# Note: numpy multidimensional indexing is mind numbing...
		# The stuff below works for 1D arrays, I don't guarantee it for higher dimensions.
		#       See: http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
		idx = [ (slice(*arg) if len(arg) > 1 else np.s_[arg[0]:]) if isinstance(arg, tuple) else arg for arg in args ]

		firstidx = np.arange(len(self)) if len(self) else np.s_[:]
		idx = tuple([ firstidx ] + idx)

		return self.__getitem__(idx)

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

class ColDict:
	catalogs = None		# Cache of loaded catalogs and tables
	columns  = None		# Cache of already referenced columns
	cell_id  = None		# cell_id on which we're operating
	
	primary_catalog = None	# the primary catalog
	include_cached = None	# whether we should include the cached data within the cell

	orig_rows= None		# Debugging/sanity checking: dict of catname->number_of_rows that any tablet of this catalog correctly fetched with fetch_tablet() should have

	def __init__(self, query, cat, cell_id, bounds, include_cached):

		self.cell_id = cell_id
		self.columns = {}
		self.primary_catalog = cat.name
		self.include_cached = include_cached

		# parse query
		(select_clause, where_clause, join_clause) = qp.parse(query, TableColsProxy(cat))
		#print (query, select_clause, where_clause, join_clause)
		#exit()

		# Fetch all rows of the base table, including the cached ones (if requested)
		rows2 = cat.fetch_tablet(cell_id=cell_id, table=cat.primary_table, include_cached=include_cached)
		idx2  = np.arange(len(rows2))
		self.orig_rows = { cat.name: len(rows2) }

#		print >> sys.stderr, "CELL:", cell_id
		# Reject objects out of bounds, in space and time (the latter only if applicable)
		have_bounds_t = False
		if len(rows2) and bounds is not None:
			schema = cat._get_schema(cat.primary_table)
			raKey, decKey = schema["spatial_keys"]
			tKey          = schema["temporal_key"] if "temporal_key" in schema else None

			# inbounds[j, i] is true if the object in row j falls within bounds[i]
			inbounds = np.ones((len(rows2), len(bounds)), dtype=np.bool)
			x, y = None, None

			for (i, (bounds_xy, bounds_t)) in enumerate(bounds):
				if bounds_xy is not None:
					if x is None:
#						print "IN", len(rows2)
						(x, y) = bhpix.proj_bhealpix(rows2[raKey], rows2[decKey]) 
#						print "OUT"

#					print "INp", len(rows2)
					inbounds[:, i] &= bounds_xy.isInsideV(x, y)
#					print "OUTp"

				if bounds_t is not None:
					have_bounds_t = True
					if tKey is not None:
#						print "INt", len(rows2)
						inbounds[:, i] &= bounds_t.isInside(rows2[tKey])
#						print "OUTt"

			# Remove all objects that don't fall inside _any_ of the bounds
			in_  = np.any(inbounds, axis=1)

#			print >> sys.stderr, "Primary table filter: (all), (kept by each)", len(inbounds), np.sum(inbounds, axis=0)
#			exit();

			idx2 = idx2[in_]
			inbounds = inbounds[in_]

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
#		self.keys     = rows2[idx2][cat.tables[cat.primary_table]["primary_key"]]
		self.keys     = rows2[cat.tables[cat.primary_table]["primary_key"]][idx2]

		def fetch_cached_tablet(cat, cell_id, table):
			""" A nested function designed to return the tablet
			    from cache (if loaded), or call cat.fetch_tablet
			    otherwise.
			    
			    Used as a parameter to cat.fetch_joins
			"""
			if cat.name in self.catalogs and table in self.catalogs[cat.name]['tables']:
				return self.catalogs[cat.name]['tables'][table]
				
			# TODO: we should cache the resulting tablet, for ColDict use later on
			#       (marginally non-trivial, need to be careful about caching and which
			#       rows to keep and which not to)
			return cat.fetch_tablet(cell_id, table)

		# Load catalog indices to be joined
		for (catname, join_type) in join_clause:
			if cat.name == catname:
				continue

			assert catname not in self.catalogs, "Same catalog, '%s', listed twice in XMATCH clause" % catname;
			assert include_cached == False, "include_cached=True in JOINs is a recipe for disaster. Don't do it"

			# load the xmatch table and instantiate the second catalog object
			(m1, m2) = cat.fetch_joins(cell_id, self.keys, catname, fetch_tablet=fetch_cached_tablet)
			cat2     = cat.get_joined_catalog(catname)
			rows2    = cat2.fetch_tablet(cell_id=cell_id, table=cat2.primary_table, include_cached=True)
			schema   = cat2._get_schema(cat2.primary_table)
			id2      = rows2[schema["primary_key"]]
			self.orig_rows[cat2.name] = len(rows2)

			# Join the tables (jmap and rows2), using (m1, m2) linkage information
			table_join.cell_id = cell_id	# debugging (remove once happy)
			table_join.cat = cat		# debugging (remove once happy)
			(idx1, idx2, isnull) = table_join(self.keys, id2, m1, m2, join_type=join_type)

			# Reject rows that are out of the time interval in this table.
			# We have to do this here as well, to support filtering on time in static_sky->temporal_sky joins
			if len(rows2) and have_bounds_t and "temporal_key" in schema:
				tKey = schema["temporal_key"]
				t    = rows2[tKey][idx2]
				in_  = np.zeros(len(idx2), dtype=bool)

				# This essentially looks for at least one bound specification that contains a given row
				for (i, (_, bounds_t)) in enumerate(bounds):
					if bounds_t is not None:
						in_t = bounds_t.isInside(t)
						in_ |= inbounds[idx1, i] & in_t
					else:
						in_ |= inbounds[idx1, i]

#				print >> sys.stderr, "Time join (all, kept): ", len(in_), in_.sum()
				#if len(in_) !=  in_.sum():
				#	print "T[~in]    =",    t[~in_]
				#	print "idx1[~in] =", idx1[~in_]
				#	print "idx1[in ] =", idx1[in_][0:10]
				#	print "idx1      =", idx1[0:20]
				#	print "idx2      =", idx2[0:20]
				#	assert 0, "Test this codepath"

				if not in_.all():
					# Keep only the rows that were found to be within at least one bound
					idx1   =   idx1[in_]
					idx2   =   idx2[in_]
					isnull = isnull[in_]

			# update the keys and index maps for the already joined catalogs
			self.keys = self.keys[idx1]
			for (catname, v) in self.catalogs.iteritems():
				(idx, isnull_) = v['join']
				v['join']      = idx[idx1], isnull_[idx1]

			#print "XX:", len(self.keys), len(rows2), idx1, idx2, isnull

			# add the newly joined catalog
			rownums  = np.arange(max(len(rows2), 1)) # The max(..,1) is for outer joins where len(rows2) = 0,
								 # but table_join (below) returns idx2=0 (with isnull=True).
								 # Otherwise, line (*) will fail
			self.catalogs[cat2.name] = \
			{
				'cat': 		cat2,
				'join':		(rownums[idx2], isnull),	# (*) -- see above
				'tables':
				{
					cat2.primary_table: rows2
				}
			}
			assert len(self.catalogs[cat2.name]['join'][0]) == len(self.keys)

		# In the cached tables, keep only the rows that remain after the JOIN has been performed
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
		retcols = []
		nrows = None
		global_ = globals()
		for (asname, name) in select_clause:
			col = eval(name, global_, self)

			self[asname] = col
			retcols.append(asname)

			if nrows != None:
				assert nrows == len(col)
			nrows = len(col)

		# evaluate the WHERE clause, to obtain the final filter
		in_    = np.empty(nrows, dtype=bool)
		in_[:] = eval(where_clause, global_, self)

		if nrows == 0:
			# We need to handle this separately, because of multidimensional columns
			self.rows_ = Table( [ (name, self[name]) for name in retcols ] )
		else:
			self.rows_ = Table( [ (name, self[name][in_]) for name in retcols ] )

	def rows(self):
		return self.rows_

	def _filter_joined(self, rows, catname):
		# Join
		cat = self.catalogs[catname]['cat']
		idx, isnull = self.catalogs[catname]['join']
		if len(rows) == 0:
			rows = np.zeros(1, dtype=rows.dtype)
		rows = rows[idx]
		if len(rows):
			for name in rows.dtype.names:
				rows[name][isnull] = cat.NULL
		return rows

	def load_column(self, name, table, catname):
		# Load the column from table 'table' of the catalog 'catname'
		# Also cache the loaded tablet, for future reuse

		cat = self.catalogs[catname]['cat']
		include_cached = self.include_cached if catname == self.primary_catalog else True

		# See if we have already loaded the required tablet
		if table in self.catalogs[catname]['tables']:
			#print table, name, ':', self.catalogs[catname]['tables'][table].dtype.names
			col = self.catalogs[catname]['tables'][table][name]
		else:
			# Load
			rows = cat.fetch_tablet(self.cell_id, table, include_cached=include_cached)
			assert len(rows) == self.orig_rows[catname]

			# Join
			rows = self._filter_joined(rows, catname)

			# Cache
			self.catalogs[catname]['tables'][table] = rows

			# return the requested column (further caching is the responsibility of the caller)
			col = rows[name]

		# Resolve blobs (if blob)
		schema = cat._get_schema(table)
		if 'blobs' in schema and name in schema['blobs']:
			assert col.dtype == np.int64, "Data structure error: blob reference columns must be of int64 type"
			refs = col
			col = cat.fetch_blobs(cell_id=self.cell_id, table=table, column=name, refs=refs, include_cached=include_cached)
			assert refs.shape == col.shape

		# Return the resolved column
		return col.view(iarray)

	def __getitem__(self, name):
		# An already loaded column?
		if name in self.columns:
			return self.columns[name]

		# A yet unloaded column from one of the joined catalogs
		# (including the primary)? Try to find it in tables of
		# joined catalogs. It may be prefixed by catalog name, in
		# which case we force the lookup of that catalog only.
		if name.find('.') == -1:
			# Do this to ensure the primary catalog is the first
			# to get looked up when resolving column names.
			#
			# FIXME: We could achieve the same effect by storing
			# self.catalogs in OrderedDict; alas, this would
			# break Python 2.6 compatibility.
			cats = [ (self.primary_catalog, self.catalogs[self.primary_catalog]) ]
			cats.extend(( v for v in self.catalogs.iteritems() if v[0] != self.primary_catalog ))
			colname = name
		else:
			# Force lookup of a specific catalog
			(catname, colname) = name.split('.')
			cats = [ (catname, self.catalogs[catname]) ]
		for (catname, v) in cats:
			cat = v['cat']
			colname = cat.resolve_alias(colname)
			for (table, schema) in cat.tables.iteritems():
				columns = set(( name for name, _ in schema['columns'] ))
				if colname in columns:
					self[name] = self.load_column(colname, table, catname)
					#print "Loaded column %s.%s.%s for %s (len=%s)" % (catname, table, colname2, name, len(self.columns[name]))
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

def _mapper(partspec, mapper, cat, query, include_cached, mapper_args, _pass_empty=False):
	(cell_id, bounds) = partspec

	# pass on some of the internals to the mapper
	mapper.CELL_ID = cell_id
	mapper.CATALOG = cat
	mapper.BOUNDS = bounds

	# Load, join, select
	rows = ColDict(query, cat, cell_id, bounds, include_cached).rows()

	# Pass on to mapper, unless empty
	if len(rows) != 0 or _pass_empty:
		result = mapper(rows, *mapper_args)
	else:
		# Catalog.map_reduce will not pass this back to the user (or to reduce)
		result = Empty

	return result

###################################################################
## Auxilliary functions implementing Catalog.build_neighbor_cache
## functionallity
def _cache_maker_mapper(rows, margin_x):
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
		for neighbor in cat.neighboring_cells(cell_id):
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
			blobrefs, i, idx = np.unique(rows[bcolname], return_index=True, return_inverse=True)
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
