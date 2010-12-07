#!/usr/bin/env python

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

class CellPathCache:
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
			while True:
				(t, has_data) = self._leaves[offs]
				offs += 1
				if t == np.inf:
					break

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
					assert self._leaves[offs][0] == 1, "There can be only one static cell (x,y,t=%s,%s,%s)" % (x, y, t)

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

	def get_cells(self, bounds=None, return_bounds=False, include_cached=False):
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
			t = self.t0 if kind == 'static' else float(kind[1:])

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
			for t, fn in siblings:
				# TODO: check if there are any non-cached data in here
				try:
					with tables.openFile(fn) as fp:
						has_data = len(fp.root.main.table) > 0
				except tables.exceptions.NoSuchNodeError:
					has_data = 0
#				has_data = 1

				# Add to leaves
				self._leaves.append((t, has_data))

			if offs0 != len(self._leaves):
				assert self._bmap.shape[0] == bhpix.width(lev), "Update this code if multi-resolution cells need to be supported"
				w2 = bhpix.width(lev)/2
				i, j = (x // dx + w2, y // dx + w2)
				self._bmap[i, j] = offs0
				# Add end marker
				self._leaves.append((np.inf, False))
				##print i, j, offs0, len(self._leaves)-offs0
		else:
			# Recursively subdivide the four subpixels
			dx = dx / 2
			for d in np.array([(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]):
				(x2, y2) = (x, y) + dx*d
				self._scan_recursive(lev+1, x2, y2)

	def _reset(self):
		w = bhpix.width(self._pix.level)

		self._leaves = [(0, False)]*2			# A floating point list of (mjd, has_data) tuples. has_data is True if the cell has non-cached data.
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
			self._bmap = bmap
			self._leaves = leaves

		# Convert _leaves to numpy array
		npl = np.empty(len(self._leaves), dtype=[('mjd', 'f4'), ('has_data', 'bool')])
		for (i, (offs, has_data)) in enumerate(self._leaves):
			npl[i] = (offs, has_data)
		self._leaves = npl

		# Recompute mipmaps
		self._compute_mipmaps()

		nstatic = np.sum(self._bmap != 0)
		print "%s cells in %s static cells" % (len(self._leaves)-nstatic-2, nstatic)
		print "%s cells with data, the rest contain caches only." % (sum(npl[npl['mjd'] < np.inf]['has_data']))
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
		cPickle.dump((self._bmaps, self._leaves, self._pix), file('fcache.pkl', mode='w'))
		return self

	def __init__(self, fn = None):
		if fn is None:
			return

		self._bmaps, self._leaves, self._pix = cPickle.load(file(fn))
		self._bmap = self._bmaps[self._pix.level]

if __name__ == '__main__':
	if 1:
		pix = Pixelization(7, 54335, 1)
		cc = CellPathCache().create_cache(pix, 'pdb/ps1_det/tablets', 'ps1_det.astrometry.h5', 'fcache.pkl')
	else:
		cc = CellPathCache('fcache.pkl')
		cells = cc.get_cells()
		print "%s cells." % len(cells)
		print cells[:10]
