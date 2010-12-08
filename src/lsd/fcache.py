#!/usr/bin/env python
"""
fcache module - TabletTreeCache implementation

TabletTreeCache class scans and caches the layout of the <table>/tablets
directory structure into a fast bitmap+list data structure pickled in
<table>/tablet_tree.pkl file. These are used by Table.get_cells() routine to
substantially speed up the cell scan.

On very large tables (e.g., ps1_det), this class accellerates the get_cells()
~ 40x (6 seconds instead of 240).

The main acceleration data structures are a 2D (W x W) numpy array of
indices (bmap), and a 1D ndarray of (mjd, next) pairs into which the indices
in the image refer to (leaves). Leaves is (logically) a singly linked list
of all temporal cells within the spatial cell, with the root of the list
being pointed to from bmap. <W> above is 2**lev, where lev is the bhpix
level of pixelization of the table.

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
	  A special value abs(next) = fcache.END_MARKER denotes there are no
	  more temporal cells.

	- For further acceleration, there is a series of images of lower
	  resolution (W/2 x W/2), (W/4 x W/4), ... (1 x 1), stored in a
	  dict() named bmaps. These can be used to quickly rule out there
	  are no populated cells in a larger partitioning of space. E.g., if
	  pixel (0, 0) is zero in the (2 x 2) map, it means there are no
	  populated cells with x < 0, y < 0 bhpix coordinates (and the
	  search of that entire subtree can be avoided). TabletTreeCache
	  makes use of these 'mipmaps' to accelerate get_cells().

TODO:
	- this cache should be build/maintained whenever the database is
	  modified. Modifications to Table._create_tablet and Table.append
	  should do the trick.
	- instead of pickling, we should mmap the bmaps and leaves
	  structures. It will make them easier to update
"""
import logging
import cPickle, os, glob
import tables
import pool2
import numpy as np
import bounds as bn
import bhpix
from pixelization import Pixelization
from interval import intervalset
from collections import defaultdict
from itertools import izip

END_MARKER=0x7FFFFFFF

def _get_cells_kernel(ij, lev, cc, bounds_xy, bounds_t):
	i, j = ij
	cells = defaultdict(dict)
	cc._get_cells_recursive(cells, bounds_xy, bounds_t, i, j, lev, bhpix.pix_size(lev))
	yield cells

def _scan_recursive_kernel(xy, lev, cc):
	x, y = xy
	cc._reset()
	cc._scan_recursive(lev, x, y)
	yield cc._bmap, cc._leaves

class TabletTreeCache:
	_bmap = None

	_bmaps = None
	_leaves = None
	
	_tablet_path = None
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
		offs = bmap[i, j]
		if not offs:
			return

		# If re reached the bottom of the hierarchy
		if offs > 1:
			# Get the cell_ids for leaf cells matching pattern
			xybounds = None if(bounds_xy.area() == box.area()) else bounds_xy
			next = 0
			while next != END_MARKER:
				(t, next) = self._leaves[offs]
				has_data = next > 0
				next = abs(next)
				offs += next

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
					assert next == END_MARKER, "There can be only one static cell (x,y,t=%s,%s,%s)" % (x, y, t)

				# Compute cell ID
				cell_id = (x, y, t)

				# Add to output
				if tbounds is None:
					outcells[cell_id][xybounds] = None
				elif xybounds not in outcells[cell_id]:
					outcells[cell_id][xybounds] = tbounds
				elif outcells[cell_id][xybounds] is not None:
					outcells[cell_id][xybounds] |= tbounds
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
			for bounds_xy, bounds_t in bounds:
				for cells_ in pool.imap_unordered(ij, _get_cells_kernel, (lev, self, bounds_xy, bounds_t), progress_callback=pool2.progress_pass):
					cells.update(cells_)
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

	#################

	def _get_temporal_siblings(self, path, pattern):
		""" Given a cell_id, get all sibling temporal cells (including the static
		    sky cell) that exist in it.
		"""
		pattern = "%s/*/%s" % (path, pattern)

		for fn in glob.iglob(pattern):
			# parse out the time, construct cell ID
			(kind, _) = fn.split('/')[-2:]
			t = self._pix.t0 if kind == 'static' else float(kind[1:])

			yield t, fn

	def _scan_recursive(self, lev = 0, x = 0., y = 0.):
		dx = bhpix.pix_size(lev)

		# Check if the cell directory exists (give up if it doesn't)
		path = self._tablet_path + '/' + self._pix._path_to_cell_base_xy(x, y, lev)
		if not os.path.isdir(path):
			return

		# If we reached the bottom of the hierarchy
		siblings = list(self._get_temporal_siblings(path, self._pattern))
		if len(siblings):
			offs0 = len(self._leaves)
			for i, (t, fn) in enumerate(siblings):
				# check if there are any non-cached data in here
				try:
					with tables.openFile(fn) as fp:
						has_data = len(fp.root.main.table) > 0
				except tables.exceptions.NoSuchNodeError:
					has_data = 0

				# Construct the "next" pointer
				next = 1 if i+1 != len(siblings) else END_MARKER

				# Convention: next is positive if there's data, negative if caches-only
				if not has_data:
					next *= -1

				# Add to leaves
				self._leaves.append((t, next))

			if offs0 != len(self._leaves):
				assert self._bmap.shape[0] == bhpix.width(lev), "Update this code if multi-resolution cells need to be supported"
				w2 = bhpix.width(lev)/2
				i, j = (x // dx + w2, y // dx + w2)
				self._bmap[i, j] = offs0
				##print i, j, offs0, len(self._leaves)-offs0
		else:
			# Recursively subdivide the four subpixels
			dx = dx / 2
			for d in np.array([(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]):
				(x2, y2) = (x, y) + dx*d
				self._scan_recursive(lev+1, x2, y2)

	def _reset(self):
		w = bhpix.width(self._pix.level)

		self._leaves = [(np.inf, END_MARKER)]*2		# A floating point list of (mjd, next_delta) tuples. next_delta is positive if the cell has non-cached data.
		self._bmap = np.zeros((w, w), dtype=np.int32)	# A map with indices to self._leaves for static cells that have an entry, zero otherwise

	def scan_table(self, pix, tablet_path, pattern):
		self._pix = pix
		self._tablet_path = tablet_path
		self._pattern = pattern

		self._reset()

		# Populate _bmap
		if False:
			self._scan_recursive()
		else:
			pool = pool2.Pool()
			lev = min(4, self._pix.level)
			dx = bhpix.pix_size(lev)
			i, j = np.indices((2**lev,2**lev)).reshape(2, -1) # List of i,j coordinates
			w2 = 1 << (lev-1)
			x, y =  (i - w2 + 0.5)*dx, (j - w2 + 0.5)*dx

			# When running in single-threaded mode, these get overwritten.
			bmap = self._bmap
			leaves = self._leaves
			self._reset()
			for b, l in pool.imap_unordered(zip(x, y), _scan_recursive_kernel, (lev, self)):
				# Adjust offsets
				b[b > 0] += len(leaves) - 2
				leaves.extend(l[2:])
				assert np.all(bmap[b != 0] == 0)
				bmap |= b
			del pool
			# Store the number of elements in the array to 'next' field of the first array element
			leaves[0] = (np.inf, len(leaves))
			self._bmap = bmap
			self._leaves = leaves

		# Convert _leaves to numpy array
		npl = np.empty(len(self._leaves), dtype=[('mjd', 'f4'), ('next', 'i4')])
		for (i, (offs, next)) in enumerate(self._leaves):
			npl[i] = (offs, next)
		self._leaves = npl

		# Recompute mipmaps
		self._compute_mipmaps()

		#nstatic = np.sum(self._bmap != 0)
		#logging.debug("%s cells in %s static cells" % (len(self._leaves)-2, nstatic))
		#logging.debug("%s cells with data, the rest contain caches only." % (sum(npl['next'] > 0)-2))
		#print self._bmaps[0]
		#print self._bmaps[1]
		#print self._bmaps[2]
		#print self._bmaps[3]
		#print self._bmaps[4].astype(int)

		return self._bmaps, self._leaves

	def _compute_mipmaps(self):
		# Create mip-maps
		self._bmaps = {self._pix.level: self._bmap}	# Mip-maps of self._bmap with True if the cell has data in its leaves, and False otherwise
		for lev in xrange(self._pix.level-1, -1, -1):
			w = bhpix.width(lev)
			m0 = np.zeros((w, w), dtype=np.int32)
			m1 = self._bmaps[lev+1]
			for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
				m0[:,:] |= m1[i::2, j::2]
			m0 = m0 != 0

			self._bmaps[lev] = m0

	def create_cache(self, pix, tablet_path, pattern, fn):
		self.scan_table(pix, tablet_path, pattern)
		cPickle.dump((self._bmaps, self._leaves, self._pix), file(fn, mode='w'))
		return self

	def __init__(self, fn = None):
		if fn is None:
			return

		self._bmaps, self._leaves, self._pix = cPickle.load(file(fn))
		self._bmap = self._bmaps[self._pix.level]

if __name__ == '__main__':
	if 1:
		pix = Pixelization(7, 54335, 1)
		#cc = TabletTreeCache().create_cache(pix, 'db/ps1_det/tablets', 'ps1_det.astrometry.h5', 'fcache.pkl')
		cc = TabletTreeCache().create_cache(pix, 'db/sdss/tablets', 'sdss.main.h5', 'fcache.pkl')
	else:
		cc = TabletTreeCache('fcache.pkl')
		cells = cc.get_cells(include_cached=False)
		print "%s cells." % len(cells)
		print cells[:10]
