#!/usr/bin/env python
"""
table_catalog module - TableCatalog implementation

TableCatalog class scans and caches the layout of the <table>/tablets
directory structure into a fast bitmap+list data structure pickled in
<table>/tablet_tree.pkl file. These are used by Table.get_cells() routine to
substantially speed up the cell scan.

On very large tables (e.g., ps1_det), this class accellerates the get_cells()
~ 40x (6 seconds instead of 240).

The main acceleration data structures are a 2D (W x W) numpy array of
indices (stored in bmaps[self._pix.level]), and a 1D ndarray of (mjd, next)
pairs into which the indices in the image refer to (leaves).  Leaves is
(logically) a singly linked list of all temporal cells within the spatial
cell, with the root of the list being pointed to from bmap.  <W> above is
2**lev, where lev is the bhpix level of pixelization of the table.

To look up whether a particular static cell exists and what it
contains do:
        - Compute the (i,j) indices of the cell in bhpix projection,
	  where (i, j) range from [0, W)
	- If bmap[i, j] == 0, the cell is unpopulated
	- If bmap[i, j] != 0, its value is an index, 'offs' into leaves.
	- leaves[offs] is a (mjd, next) tuple, with mjd being the
	  temporal coordinate of the cell.
	- The absolute value of next is the offset to the next entry in
	  leaves that is a temporal cell within this static cell (i.e., offs
	  += abs(leaves[offs]['next']) will position you to the next cell).  Its
	  sign is negative if the cell contain only neighbor
	  caches; otherwise it's positive. This allows us to quickly cull
	  cache-only cells from queries that don't care about it.	  
	  A special value abs(next) = table_catalog.END_MARKER denotes there are no
	  more temporal cells.

	- For further acceleration, there is a series of images of lower
	  resolution (W/2 x W/2), (W/4 x W/4), ... (1 x 1), stored in a
	  dict() named bmaps. These can be used to quickly rule out there
	  are no populated cells in a larger partitioning of space. E.g., if
	  pixel (0, 0) is zero in the (2 x 2) map, it means there are no
	  populated cells with x < 0, y < 0 bhpix coordinates (and the
	  search of that entire subtree can be avoided). TableCatalog
	  makes use of these 'mipmaps' to accelerate get_cells().

TODO:
	- the catalog should be build/maintained whenever the database is
	  modified. Modifications to Table._create_tablet and Table.append
	  should do the trick.
	  	- Note however that there needs to be a synchronization
	  	  mechanism if multiple processes are operating on the same
	  	  table (we probably need a server)
"""
import logging
import cPickle, os, glob
import tables
import pool2
import numpy as np
import bounds as bn
import bhpix
import utils
from pixelization import Pixelization
from interval import intervalset
from collections import defaultdict
from itertools import izip

logger = logging.getLogger('lsd.table_catalog')

END_MARKER=0x7FFFFFFF

def _add_bounds(outcells, cell_id, xybounds, tbounds):
	# cells is a dictionary of cell_id -> dict objects,
	# where each dict object is another dictionary of xybounds -> tbounds
	#
	# This function adds the (xybounds, tbounds) boundaries
	# to the dictionary, for a given cell_id, correctly merging it
	# with any bounds that already exist

	if tbounds is None:
		outcells[cell_id][xybounds] = None		# Cover all times (no temporal bounds)
	elif xybounds not in outcells[cell_id]:			# New entry
		outcells[cell_id][xybounds] = tbounds
	elif outcells[cell_id][xybounds] is not None:		# Merge tbounds with existing entry
		outcells[cell_id][xybounds] |= tbounds

def _get_cells_kernel(ij, lev, cc, bounds):
	i, j = ij
       	cells = defaultdict(dict)
	for bounds_xy, bounds_t in bounds:
		cc._get_cells_recursive(cells, bounds_xy, bounds_t, i, j, lev, bhpix.pix_size(lev))
	yield cells

def _scan_recursive_kernel(xy, lev, cc):
	x, y = xy

	w = bhpix.width(cc._pix.level)
	bmap = np.zeros((w, w), dtype=object)
	cc._scan_recursive(bmap, lev, x, y)

	yield bmap

def iter_siblings(a, at):
	# Iterate through a linked list
	offs = 0
	while offs != END_MARKER:
		at += offs

		row = tuple(a[at])
		yield row

		offs = abs(a['next'][at])

def get_snapshot_path(table_path, snapid):
	if snapid == 0:
		return table_path
	else:
		return os.path.join(table_path, 'snapshots', snapid)

def isnapshots(table_path, return_path=False, first=None, last=None, no_first=False):
	""" Returns an iterable of committed snapshots, in no particular order """
	# Adjust boundaries
	if last is None:
		last = "z"*80	# No snapshot ID should be lexicographically greater than this

	# pre v0.5.0 tables compatibility:
	if first in [0, None]:
		path = os.path.join(table_path, 'tablets')
		if os.path.isdir(path):
			yield (0 if not return_path else (0, table_path))

	# enumerate snapshots
	for path in glob.iglob('%s/snapshots/*/.committed' % (table_path)):
		t = path.split('/')[-2]
		if first <= t and t <= last:
			if no_first and first == t:
				continue
			yield (t if not return_path else (t, os.path.dirname(path)))

class TableCatalog:
	_bmaps = None
	_leaves = None
	_pix = None
	
	#################

	def _get_cells_recursive(self, outcells, bounds_xy, bounds_t, i = 0, j = 0, lev = 0, dx = 2.):
		""" Helper for get_cells(). See documentation of
		    get_cells() for usage
		"""
		w2 = 1 << (lev-1)
		x, y =  (i - w2 + 0.5)*dx, (j - w2 + 0.5)*dx

		# Check for nonzero overlap
		box  = self._pix._cell_bounds_xy(x, y, dx)
		bounds_xy = bounds_xy & box
		if not bounds_xy:
			return

		# Check if the cell directory exists (give up if it doesn't)
		bmap = self._bmaps[lev]
		spatial = bmap[i, j]
		if not spatial:
			return

		# If re reached the bottom of the hierarchy
		if lev == self._pix.level:
			# Get the cell_ids for leaf cells matching pattern
			xybounds = None if(bounds_xy.area() == box.area()) else bounds_xy

			for t, (snapid, storage_cell_id, has_data) in spatial.iteritems():
				if not has_data and not self._include_cached:
					continue

				if t != self._pix.t0:
					# Cut on the time component
					tival = intervalset((t, t+self._pix.dt))
					tolap = bounds_t & tival
					if len(tolap):
						(l, r) = tolap[-1]				# Get the right-most interval component
						if l == r == t+self._pix.dt:				# Is it a single point?
							tolap = intervalset(*tolap[:-1])	# Since objects in this cell have time in [t, t+dt), remove the t+dt point
	
					if len(tolap) == 0:					# No overlap between the intervals -- skip this cell
						continue;
	
					# Return None if the cell is fully contained in the requested interval
					tbounds = None if tival == tolap else tolap
				else:
					# Static cell
					tbounds = bounds_t
					assert len(spatial) == 1, "There can be only one static cell (x,y,t=%s,%s,%s)" % (x, y, t)

				# Add to output
				_add_bounds(outcells, (x, y, t), xybounds, tbounds)
		else:
			# Recursively subdivide the four subpixels
			for (di, dj) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
				i2 = i*2 + di
				j2 = j*2 + dj
				self._get_cells_recursive(outcells, bounds_xy & box, bounds_t, i2, j2, lev+1, dx/2)

	def get_cells(self, bounds=None, return_bounds=False, include_cached=True):
		""" Return a list of (cell_id, bounds) tuples completely
		    covering the requested bounds.

			bounds must be a list of (Polygon, intervalset) tuples.

		    Output is a list of (cell_id, xybounds, tbounds) tuples,
		    unless return_bounds=False when the output is just a
		    list of cell_ids.
		"""
		self._include_cached = include_cached

		# Special case of bounds=None (all sky)
		if bounds == None:
			bounds = [(bn.ALLSKY, intervalset((-np.inf, np.inf)))]

		# Find all existing cells satisfying the bounds
		cells = defaultdict(dict)
		if False:
			# Single-threaded implementation
			for bounds_xy, bounds_t in bounds:
				self._get_cells_recursive(cells, bounds_xy, bounds_t, 0, 0, 1, 1.)
				self._get_cells_recursive(cells, bounds_xy, bounds_t, 0, 1, 1, 1.)
				self._get_cells_recursive(cells, bounds_xy, bounds_t, 1, 0, 1, 1.)
				self._get_cells_recursive(cells, bounds_xy, bounds_t, 1, 1, 1, 1.)
		else:
			# Multi-process implementation (appears to be as good or better than single thread in
			# nearly all cases of interest)
			pool = pool2.Pool()
			lev = min(4, self._pix.level)
			ij = np.indices((2**lev,2**lev)).reshape(2, -1).T # List of i,j coordinates
			for cells_ in pool.imap_unordered(ij, _get_cells_kernel, (lev, self, bounds), progress_callback=pool2.progress_pass):
				for xyt, b in cells_.iteritems():
					for xyb, tb in b.iteritems():
						_add_bounds(cells, xyt, xyb, tb)
			del pool

		if len(cells):
			# Transform (x, y, t) tuples to cell_ids
			xyt = np.array(cells.keys()).transpose()
			cell_ids = self._pix._cell_id_for_xyt(xyt[0], xyt[1], xyt[2])

			# Reorder cells to be a dict of cell: [(poly, time), (poly, time)] entries
			cells = dict(( (cell_id, v.items()) for (cell_id, (k, v)) in izip(cell_ids, cells.iteritems()) ))

		if not return_bounds:
			return cells.keys()
		else:
			return cells

	def get_cells_in_snapshot(self, snapid, cgroup, include_cached=True):
		""" Return a list of cells that are physically stored in snapshot snapid """
		xyt = []
		for (x, y), spatial in self.spatial_index[cgroup][self._pix.level].iteritems():
			for t, (snapid_, storage_cell_id, has_data) in spatial.iteritems():
				if snapid_ == snapid:
					xyt.append((x, y, t))

		xyt = np.array(xyt).transpose()
		return self._pix._cell_id_for_xyt(xyt[0], xyt[1], xyt[2])

	def tablet_location(self, cell_id, cgroup):
		# A replacement for snapshot_of_cell
		x, y, t = self._pix._xyt_from_cell_id(cell_id)
		w2 = self.width / 2
		i, j = (x // dx + w2, y // dx + w2)
		try:
			snapid, storage_cell_id, has_data = self.spatial_index[cgroup][(x, y)][t]
			return snapid, storage_cell_id
		except KeyError:
			raise LookupError()

	def writable_tablet_location(self, cell_id, cgroup, snapid, table_path):
		# Find/update writable tablet location

		# See if we already have the location in catalog
		try:
			return self.index[(cell_id, cgroup)]
		except KeyError:
			pass

		# See if the tablet exists in previous snapshot
		try:
			storage_snapid, storage_cell_id = self.tablet_location(cell_id, cgroup)
			self.index[(cell_id, cgroup)] = snapid, storage_cell_id
			return snapid, storage_cell_id
		except LookupError:
			pass

		# An unpopulated cell -- find where to place it
		...

		base = get_snapshot_path(table_path, snapid)

	#################

	def _get_temporal_siblings(self, path):
		""" Given a path to a cell, get all sibling temporal cells (including the static
		    sky cell) that exist in it.
		"""

		pattern = "%s/*/%s.*.h5" % (path, self.__name)	# The first asterisk is either 'Txxxx' or 'static'. The second is for cgroup

		for fn in glob.iglob(pattern):
			# parse out the time, construct cell ID
			(kind, fname) = fn.split('/')[-2:]
			t = self._pix.t0 if kind == 'static' else float(kind[1:])

			# parse out the cgroup
			_, cgroup, _ = fname.split('.')
			assert cgroup in self.__cgroups

			yield t, cgroup, fn

        def _has_data(self, node):
		# check if there are any non-cached data in here
		try:
			return len(node.main.table) > 0
		except tables.exceptions.NoSuchNodeError:
			return False

	def _scan_recursive(self, bmap, lev=0, x=0., y=0.):
		# Check if the cell directory exists in any allowable snapshot
		# Snapshots are ordered newest-to-oldest, so we can avoid touching the
		# siblings of older snapshots if they exists in newer ones
		dx = bhpix.pix_size(lev)

		# .../snapshots/XXXXXX/tablets/-0.5+0.5/..../
		snapid = self._snapid

		path = '%s/tablets/%s' % (get_snapshot_path(table_path, snapid),  self._pix._path_to_cell_base_xy(x, y, lev))

		if not os.path.isdir(path): # Dead end
			return

		if lev != self._pix.level:
			# Recursively subdivide the four subpixels
			dx = dx / 2
			for d in np.array([(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]):
				(x2, y2) = (x, y) + dx*d
				self._scan_recursive(bmap, lev+1, x2, y2)
		else:
			# Leaf
			w2 = bhpix.width(lev) // 2
			i, j = (x // dx + w2, y // dx + w2)

			# Collect the data we have, and add them to the set of cells that exist
			for tcell, cgroup, fn in self._get_temporal_siblings(path):
				storage_cell_id = self._pix._cell_id_for_xyt(x, y, tcell)
				with tables.openFile(fn) as fp:
					# check if this is a v0.5.x file
					if 'main' in fp.root:
						has_data = self._has_data(fp.root)
						self._add_tablet(bmap, x, y, tcell, w2, cgroup, storage_cell_id, has_data)
					else:
						# Enumerate all cells that are stored in this file
						for (name, node) in fp.root._v_children.iteritems():
							if name[0] != 'c':
								continue

							cell_id = np.uint64(name[1:])
							has_data = self._has_data(node)

							xx, yy, tt = self._pix._xyt_from_cell_id(cell_id)
							self._add_tablet(bmap, xx, yy, tt, w2, cgroup, storage_cell_id, has_data)

	# bmaps: bmaps[cgroup][(i, j)][tcell] -> (snapid, storage_cell_id)
	def _add_tablet(self, bmaps, x, y, tcell, w2, cgroup, snapid, storage_cell_id, has_data):
		# Get the map for the given cgroup
		try:
			bmap = bmaps[cgroup]
		except KeyError:
			bmap = bmaps[cgroup] = defaultdict(dict)

		# Find the spatial cell
		i, j = (x // dx + w2, y // dx + w2)
		spatial = bmap[(i, j)]

		# Store the location information to the
		# apropriate temporal cell
		spatial[tcell] = (snapid, storage_cell_id, has_data)

	def update(self, table_path, cgroups, snapid):
		self.__cgroups = cgroups

		# Find what we already have loaded
		prevsnap = self.snapid
		assert prevsnap <= snapid, "Cannot update a catalog to an older snapshot"

		# Update the snapshot ID
		self.snapid = snapid

		# Get existing top-level maps from self.spatial_index
		bmaps = { cgroup: bmaps[self._pix.level] for cgroup, bmaps in self.spatial_index }

		# Recursively scan, in parallel
		lev = min(4, self._pix.level)
		dx = bhpix.pix_size(lev)
		i, j = np.indices((2**lev,2**lev)).reshape(2, -1) # List of i,j coordinates
		w2 = 1 << (lev-1)
		x, y =  (i - w2 + 0.5)*dx, (j - w2 + 0.5)*dx

		pool = pool2.Pool()
		for bmaps2 in pool.imap_unordered(zip(x, y), _scan_recursive_kernel, (lev, self)):
			# Update the existing map with new data
			# NOTE: We don't yet support deletion !
			for cgroup, bmap2 in bmaps2.iteritems():
				try:
					bmap = bmaps[cgroup]
				except KeyError:
					bmap = bmaps[cgroup] = defaultdict(dict)

				for key, spatial in bmap2:
					bmap[key].update(spatial)
		del pool

		# Recompute mipmaps
		self.spatial_index = { cgroup: self._compute_mipmaps(b) for cgroup, b in bmaps }

	def _compute_mipmaps(self, bmap):
		# True/False bitmap: is there's data in a spatial cell?
		bmapTF = np.zeros((self.width, self.width), dtype=np.bool)
		for key, spatial:
			bmapTF[key] = len(spatial) != 0

		# Create mip-maps
		bmaps = {self._pix.level: bmapTF}
		for lev in xrange(self._pix.level-1, -1, -1):
			w = bhpix.width(lev)
			m0 = np.zeros((w, w), dtype=np.bool)
			m1 = bmaps[lev+1]
			for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
				m0[:,:] |= m1[i::2, j::2]
			bmaps[lev] = m0

		# Final mipmap is the dict with data
		bmaps[self._pix.level] = bmap

		return bmaps

	def save(self, fn):
		dir = os.path.dirname(os.path.normpath(fn))
		if dir != '':
			utils.mkdir_p(dir)
		cPickle.dump((self.spatial_index, self._pix), file(fn, mode='w'), -1)

	def load(self, fn):
		data = cPickle.load(file(fn))
		if len(data) == 3:
			# v0.5.x compatibility
			self._bmaps, self._leaves, self._pix = data

			# TODO: write code transforming this to new data structures
			...


	def __init__(self, fn=None, pix=None):
		if fn is not None:
			assert pix is None
			self.load(fn)
		else:
			assert pix is not None
			assert fn is None
			self._pix = pix
			self.spatial_index = {}

		self.width = bhpix.width(self._pix.level)

	def __eq__(self, b):
		# Compare Pixelization objects
		if self._pix != b._pix:
			return False
		
		# Compare low-resolution (boolean) mipmaps
		for i in xrange(self._pix.level):
			if np.all(self._bmaps[i] != b._bmaps[i]):
				return False

		# Compare the overall layout of the base bitmap
		bmap1 = self._bmaps[self._pix.level]
		bmap2 = b._bmaps[b._pix.level]
		if not np.all( (bmap1 > 1) == (bmap2 > 1) ):
			return False

		# Compare the temporal siblings in each bitmap
		for offs1, offs2 in izip(bmap1[bmap1 > 1], bmap2[bmap2 > 1]):
			list1 = sorted((mjd, snap_id, cell_id) for (mjd, snap_id, cell_id, _) in iter_siblings(self._leaves, offs1))
			list2 = sorted((mjd, snap_id, cell_id) for (mjd, snap_id, cell_id, _) in iter_siblings(b._leaves,    offs2))
			if list1 != list2:
				return False

		# Compare _leaves, all columns but 'next'
		if self._leaves.dtype != b._leaves.dtype:
			return False
		names = [ name for name in self._leaves.dtype.names if name != 'next' ]
		s1 = np.sort(self._leaves, order=names)
		s2 = np.sort(b._leaves,    order=names)
		for name in names:
			if not np.all(s1[name] == s2[name]):
				return False

		# Compare the signs of the 'next' column (== has_data)
		if not np.all((s1['next'] > 0) == (s2['next'] > 0)):
			return False

		# The two objects are identical
		return True

	def __ne__(self, b):
		return not (self == b)

def check_table_catalog(table_path, pattern, snapid=None):
	""" Verify the consistency of the table catalog
	
	    Load the table catalog given by snapid (or the lates one,
	    if none is given), and compare it to a catalog reconstructed
	    from scratch.
	"""

	# Load existing catalog
	snapshots = dict(isnapshots(table_path, return_path=True))
	if snapid is None:
		snapid = max(snapshots.keys())
	fn = os.path.join(snapshots[snapid], 'catalog.pkl')
	cc1 = TableCatalog(fn=fn)

	# Construct one from scratch
	cc2 = TableCatalog(pix=cc1._pix)
	cc2.update(table_path, pattern, snapid)

	assert cc1 == cc2

if __name__ == '__main__':
	tpath = '/n/pan/mjuric/lsd_test5/ps1_det'
	#check_table_catalog(tpath, 'ps1_det.astrometry.h5'); exit()
	if False:
		import sys
		pix = Pixelization(6, 54335, 1)
		cc = TableCatalog(pix=pix)
		if True:
			snaps = sorted(isnapshots(tpath))
			for i, snapid in enumerate(snaps):
				print >>sys.stderr, snapid,
				if i+1 == len(snaps) or (i % 2 == 0):
					cc.update(tpath, 'ps1_det.astrometry.h5', snapid)
				else:
					print >>sys.stderr, "skipping."
		else:
			cc.load('catalog.pkl')
		cc.update(tpath, 'ps1_det.astrometry.h5', '20110617094228.164285')
		###cc.update(tpath, 'ps1_det.astrometry.h5', '20110617084754.101965')
		###cc.update(tpath, 'ps1_det.astrometry.h5', '20110617094228.164285')
		cc.save('catalog.pkl')
		#cc2 = TableCatalog(fn='/n/pan/mjuric/lsd_test5/ps1_det/snapshots/20110617084754.101965/catalog.pkl')
		cc2 = TableCatalog(fn='/n/pan/mjuric/lsd_test5/ps1_det/snapshots/20110617094228.164285/catalog.pkl')
		assert cc == cc2
		print "OK"
	else:
		cc1 = TableCatalog(fn='catalog.pkl')
		#cc2 = TableCatalog(fn='/n/pan/mjuric/lsd_test5/ps1_det/snapshots/20110617084754.101965/catalog.pkl')
		cc2 = TableCatalog(fn='/n/pan/mjuric/lsd_test5/ps1_det/snapshots/20110617094228.164285/catalog.pkl')
		assert cc1 == cc2
		print "OK"
