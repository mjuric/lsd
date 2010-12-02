#!/usr/bin/env python
import bhpix
import numpy as np
import Polygon.Shapes
from intervalset import intervalset
from numpy import fabs
import bounds as bn
import glob, os
from collections import defaultdict

class Pixelization(object):
	""" Pixelization class - manages the generation and interpretation
	    of object and cell IDs, as well as the directory structure
	    representing paths to individual cells.
	    
	    Clients should always use this class to obtain paths to cells,
	    and convert between lon/lat and object and cell_ids.
	"""
	# Space/time pixelization (filled in constructor)
	level = None
	t0    = None
	dt    = None

	# Number of bits reserved for position and time coordinate
	# in a 32-bit cell identifier
	xybits = 10
	tbits  = 32 - 2*xybits

	### Low-level functions that care about the bits in IDs

	def __init__(self, level, t0, dt):
		self.level = level
		self.t0 = t0
		self.dt = dt

		# Compute object_id -> cell_id mask (used by cell_for_id())
		xymask = 2**(self.xybits - self.level)-1	# 0b111
		mask = (~(
					xymask << (self.xybits+self.tbits) | 
					xymask << (self.tbits)
				) << 32) | 0xFFFFFFFF
		self.id2cell_mask = mask
#		print bin(xymask)
#		print mask
#		print bin(mask)
#		print bin(np.uint64(mask))
#		exit()

		# Temporal cell -> static cell_id mask (used by static_cell_for_cell())
		mask = (2**self.tbits-1)<< 32
		self.tmask = mask
		self.t2static_mask = ~mask			# Zeros out the temporal bits when &-ed

		self.mask_x32 = (2**self.xybits-1) << (self.xybits + self.tbits)
		self.mask_y32 = (2**self.xybits-1) << (self.tbits)
		self.mask_t32 = (2**self.tbits-1)

#		cell_id = self._cell_id_for_xyt(-0.328125, +0.234375, 55249)
#		(x, y, t) = self._xyt_from_cell_id(cell_id)
#		print cell_id
#		print x, y, t
#		exit()

	def _id_from_xyti(self, x, y, t=None, i = 0):
		""" Returns the pixel ID, given x, y, t and object
		    index i. If i == 0xFFFFFFFF, we return the
		    corresponding cell ID.
		"""
		if t is None:
			t = np.array(self.t0)
		ct = np.array((t - self.t0) / self.dt, np.uint64)

#		# Round to closest pixel
#		# TODO: This shouldn't strictly be necessary
#		(x, y) = bhpix.xy_center(x, y, self.xybits)

		# construct the 32bit ID prefix from the above
		# Prefix format: 10bit x + 10bit y + 12bit time
		npixhalf = 2**(self.xybits-1)
		ix   = np.array((1 + x) * npixhalf)
		iy   = np.array((1 + y) * npixhalf)
		ix   = ix.astype(np.uint64)
		iy   = iy.astype(np.uint64)
		id   = ix << (self.tbits + self.xybits)
		id  |= iy << (self.tbits)
		id  |= ct & (2**self.tbits - 1)
		id <<= 32
		id  |= i & 0xFFFFFFFF

		if np.any(i == 0xFFFFFFFF):
			# If we're computing cell IDs, make sure to wipe out the
			# sub-pixel bits for the given level (otherwise we'd get
			# multiple cell_ids for the same cell, depending on where
			# in the cell the x,y were)
			id &= self.id2cell_mask

		# TODO: Transformation verification, remove when debugged
		if np.any(i == 0xFFFFFFFF):
			ux, uy, ut = self._xyt_from_cell_id(id)
			ui = 0xFFFFFFFF
			(cx, cy) = bhpix.xy_center(x, y, self.level)
		else:
			ux, uy, ut, ui = self._xyti_from_id(id)
			(cx, cy) = bhpix.xy_center(x, y, self.xybits)
		ct = ct * self.dt + self.t0
		ci = i
		if np.any(cx != ux) or np.any(cy != uy) or np.any(ct != ut) or np.any(ci != ui):
			print cc, "==", cu, ct, "==", ut
			raise Exception("**** Bug detected ****")			

		# NOTE: Test tranformation correctness (comment this out for production code)
		#(ux, uy, ut, ui) = self._xyti_from_id(id)
		#cc = bhpix.xy_center(x, y, self.xybits)
		#cu = bhpix.xy_center(ux, uy, self.xybits)
		#ct = ct * self.dt + self.t0
		#if np.any(cc[0] != cu[0]) or np.any(cc[1] != cu[1]) or np.any(ct != ut) or np.any(ui != 0):
		#	print cc, "==", cu, ct, "==", ut
		#	raise Exception("**** Bug detected ****")			

		return id

	def _xyti_from_id(self, id):
		# Unpack x, y, t, i from ID. If i == 0, assume this
		# is a cell_id (and not object ID), and round the
		# x, y to our pixelization level.

		id = np.array(id, np.uint64)
		ci = id & 0xFFFFFFFF

		id >>= 32
		ix = (id & self.mask_x32) >> (self.xybits + self.tbits)
		iy = (id & self.mask_y32) >> (self.tbits)
		it = (id & self.mask_t32)
		npixhalf = 2.**(self.xybits-1)
		cx = (ix + 0.5) / npixhalf - 1.
		cy = (iy + 0.5) / npixhalf - 1.
		ct = np.array(it, float) * self.dt + self.t0

		# Special handling for cell IDs
		cellids = ci == 0xFFFFFFFF
		if np.any(cellids):
			assert np.all(cellids)
			(cx, cy) = bhpix.xy_center(cx, cy, self.level)

		return (cx, cy, ct, ci)

	def _xyt_from_cell_id(self, cell_id):
		assert np.all(self.is_cell_id(cell_id))
		return self._xyti_from_id(cell_id)[:3]

	def _cell_id_for_xyt(self, x, y, t=None):
		return self._id_from_xyti(x, y, t, 0xFFFFFFFF)

	def cell_for_id(self, id):
		""" Return a cell id corresponding to a given object ID """
		cell_id    = (id & np.uint64(self.id2cell_mask)) | 0xFFFFFFFF
		assert np.all(self.is_cell_id(cell_id))

		# TODO: Debugging (remove when happy)
		x, y, t, _ = self._xyti_from_id(id)
		cell_id2    = self._cell_id_for_xyt(x, y, t)
		assert np.all(cell_id2 == cell_id), 'cell_id2=%s cell_id=%s %s x=%s y=%s t=%s' % (cell_id2, cell_id, bin(cell_id), x, y, t)

		return cell_id

	def id_for_cell_i(self, cell_id, i):
		# Returns an ID given a valid cell_id and the 32-bit
		# object ID part
		assert np.all(self.is_cell_id(cell_id))		# Must be cell_id
		assert not np.any(i & 0xFFFFFFFF00000000)	# Must be 32bit
		return (cell_id & 0xFFFFFFFF00000000) | i

	def is_cell_id(self, id):
		# Return True if id is a cell_id
		return id | 0x00000000FFFFFFFF == id

	def static_cell_for_cell(self, cell_id):
		""" Return a static-sky cell corresponding to a given cell_id """
		assert np.all(self.is_cell_id(cell_id))
		# Mask out the time
		return cell_id & self.t2static_mask

	def is_temporal_cell(self, cell_id):
		""" Return True if the cell_id points to a temporal cell """
		assert np.all(self.is_cell_id(cell_id))
		return cell_id & self.tmask

	def str_id(self, id):
		""" Binary pretty-print an id """
		s = bin(id)
		s = '0'*64 + s[2:]
		idbits = 64 - (2*self.xybits+self.tbits)
		ret = []
		nover = self.xybits - self.level
		for l, k in [(0, idbits), (0, self.tbits), (nover, self.xybits), (nover, self.xybits)]:
			c = s[-k:]
			s = s[:-k]
			if l:
				c = c[:-l] + '[' + c[-l:] + ']'
			ret.insert(0, c)
		return ' '.join(ret)

	#############

	def obj_id_from_pos(self, ra, dec, t=None, i=0):
		# Return an ID for an object with pos (ra, dec, t)
		(x, y) = bhpix.proj_bhealpix(ra, dec)
		return self._id_from_xyti(x, y, t, i)

	def cell_id_for_pos(self, ra, dec, t=None):
		# Return cell_id where (ra, dec, t) can be found
		(x, y) = bhpix.proj_bhealpix(ra, dec)
		return self._cell_id_for_xyt(x, y, t)

	def _cell_bounds_xy(self, x, y, dx = None):
		""" Return a bounding polygon for a given
		    cell center x,y
		"""
		if dx is None:
			lev  = bhpix.get_pixel_level(x, y)
			dx   = bhpix.pix_size(lev)
			##dx = bhpix.pix_size(self.level)

		bounds = Polygon.Shapes.Rectangle(dx)
		bounds.shift(x - 0.5*dx, y - 0.5*dx);

		if fabs(fabs(x) - fabs(y)) == 0.5:
			# If it's a "halfpixel", return a triangle
			# by clipping agains the sky
			bounds &= bn.ALLSKY

		return bounds

	def cell_bounds(self, cell_id):
		"""
			Return the bounding polygon and time intervalset
			for a given cell.
		"""
		x, y, t, _ = self._xyti_from_id(cell_id)
		assert _ == 0xFFFFFFFF

		bounds = self._cell_bounds_xy(x, y)
		return (bounds, intervalset((t, t+self.dt)))

	def _path_to_cell_base_xy(self, x, y, level = None):
		# The only place in the code where we map from
		# (x, y) to cell base path
		if level is None:
			level  = bhpix.get_pixel_level(x, y)

		path = [];
		for lev in xrange(1, level+1):
			(cx, cy) = bhpix.xy_center(x, y, lev)
			subpath = "%+g%+g" % (cx, cy)
			path.append(subpath)

		return '/'.join(path)

	def path_to_cell(self, cell_id, return_base=False):
		# Return the full path to a given cell
		x, y, t = self._xyt_from_cell_id(cell_id)
		path = self._path_to_cell_base_xy(x, y, self.level)

		if not return_base:
			if t >= self.t0 + self.dt:
				path = '%s/T%05.0f' % (path, t)
			else:
				path = '%s/static' % path

		return path

	def _get_temporal_siblings(self, x, y, path, pattern):
		""" Given a cell_id, get all sibling temporal cells (including the static
		    sky cell) that exist in it.
		"""
		pattern = "%s/*/%s" % (path, pattern)

		for fn in glob.iglob(pattern):
			# parse out the time, construct cell ID
			(kind, _) = fn.split('/')[-2:]
			t = self.t0 if kind == 'static' else float(kind[1:])

			cell_id = self._cell_id_for_xyt(x, y, t)
			yield cell_id

	def _is_valid_cell_xy(self, x, y):
		return bhpix.get_pixel_level(x, y) == self.level

	def _get_cells_recursive(self, outcells, data_path, bounds_xy, bounds_t, pattern, x = 0., y = 0.):
		""" Helper for get_cells(). See documentation of
		    get_cells() for usage
		"""
		lev  = bhpix.get_pixel_level(x, y)
		dx   = bhpix.pix_size(lev)

		# Check for nonzero overlap
		box  = self._cell_bounds_xy(x, y, dx)
		bounds_xy = bounds_xy & box
		if not bounds_xy:
			return

		# Check if the cell directory exists (give up if it doesn't)
		path = data_path + '/' + self._path_to_cell_base_xy(x, y, lev)
		if not os.path.isdir(path):
			return

		# If re reached the bottom of the hierarchy
		if self._is_valid_cell_xy(x, y):
			# Get the cell_ids for leaf cells matching pattern
			xybounds = None if(bounds_xy.area() == box.area()) else bounds_xy
			sibling_cells = list(self._get_temporal_siblings(x, y, path, pattern))
			for cell_id in sibling_cells:
				# Filter on time, add bounds
				x, y, t = self._xyt_from_cell_id(cell_id)

				if t != self.t0:
					# Cut on the time component
					tival = intervalset((t, t+self.dt))
					tolap = bounds_t & tival
					if len(tolap):
						(l, r) = tolap[-1]				# Get the right-most interval component
						if l == r == t+self.dt:				# Is it a single point?
							tolap = intervalset(*tolap[:-1])	# Since objects in this cell have time in [t, t+dt), remove the t+dt point
	
					if len(tolap) == 0:					# No overlap between the intervals -- skip this cell
						continue;
	
					# Return None if the cell is fully contained in the requested interval
					tbounds = None if tival == tolap else tolap
				else:
					# Static cell
					tbounds = bounds_t
					assert len(sibling_cells) == 1, "There can be only one static cell (cell_id=%s)" % cell_id

				# Add to output
				if tbounds is None:
					outcells[cell_id][xybounds] = None
				elif xybounds not in outcells[cell_id]:
					outcells[cell_id][xybounds] = tbounds
				elif outcells[cell_id][xybounds] is not None:
					outcells[cell_id][xybounds] |= tbounds
		else:
			# Recursively subdivide the four subpixels
			dx = dx / 2
			for d in np.array([(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]):
				(x2, y2) = (x, y) + dx*d
				self._get_cells_recursive(outcells, data_path, bounds_xy & box, bounds_t, pattern, x2, y2)

	def get_cells(self, data_path, pattern, bounds, return_bounds=False):
		""" Return a list of (cell_id, bounds) tuples completely
		    covering the requested bounds.

			bounds must be a list of (Polygon, intervalset) tuples.

		    Output is a list of (cell_id, xybounds, tbounds) tuples,
		    unless return_bounds=False when the output is just a
		    list of cell_ids.
		 """

		# Special case of bounds=None (all sky)
		if bounds == None:
			bounds = [(bn.ALLSKY, intervalset((-np.inf, np.inf)))]

		# Find all existing cells satisfying the bounds
		cells = defaultdict(dict)
		for bounds_xy, bounds_t in bounds:
			self._get_cells_recursive(cells, data_path, bounds_xy, bounds_t, pattern)

		# Reorder cells to be a dict of cell: [(poly, time), (poly, time)] entries
		cells = dict(( (k, v.items()) for k, v in cells.iteritems() ))

		if False:
			# TODO: Debugging, remove when happy
			for k, bounds in cells.iteritems():
				print k, ':', str(self._xyti_from_id(k)),
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
		cell_ids   = np.array(cell_ids, copy=False)
		cell_id_xy = self.static_cell_for_cell(cell_ids)

		ret = {}
		for cell_id in set(cell_id_xy):
			ret[cell_id] = cell_ids[cell_id_xy == cell_id]

		return ret

	def neighboring_cells(self, cell_id, include_self=False):
		""" Returns the cell IDs of cells spatially adjacent 
		    to cell_id.
		"""
		x, y, t = self._xyt_from_cell_id(cell_id)

		ncells = bhpix.neighbors(x, y, self.level, include_self)
		for (cx, cy) in ncells:
			if fabs(fabs(cx) - fabs(cy)) > 0.5:
				print "PROBLEM: ", x, y, cx, cy
				print ncells

		nhood = [ self._cell_id_for_xyt(x, y, t) for (x, y) in ncells ]

		# TODO: Remove once we're confident it works
		rrr = set([ self._xyt_from_cell_id(cid)[:2] for cid in nhood ])
		assert rrr == ncells

		return nhood

