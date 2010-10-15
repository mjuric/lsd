"""
	Common tasks needed when dealing with survey datasets.
"""

import pool2
import numpy as np
from itertools import izip
import bhpix
import catalog

###################################################################
## Sky-coverage computation
def _coverage_mapper(rows, dx = 1., filter=None, filter_args=()):
	self = _coverage_mapper

	if filter is not None:
		rows = filter(rows, *filter_args)

	if len(rows) == 0:
		return (None, None, None, None)

	i = (rows['ra'] / dx).astype(int)
	j = ((90 - rows['dec']) / dx).astype(int)

	(imin, imax, jmin, jmax) = (i.min(), i.max(), j.min(), j.max())
	w = imax - imin + 1
	h = jmax - jmin + 1
	sky = np.zeros((w, h))

	i -= imin; j -= jmin
	for (ii, jj) in izip(i, j):
		sky[ii, jj] += 1

	return (sky, imin, jmin, self.CELL_FN)

def compute_coverage(cat, dx = 0.5, include_cached=False, where=None, filter=None, filter_args=(), foot=catalog.All):
	""" compute_coverage - produce a sky map of coverage, using
	    a filter function if given. The output is a starcount
	    array in (ra, dec) binned to <dx> resolution.
	"""
	width  = int(round(360/dx))
	height = int(round(180/dx))

	sky = np.zeros((width, height))

	for (patch, imin, jmin, fn) in cat.map_reduce(_coverage_mapper, mapper_args=(dx, filter, filter_args), where=where, include_cached=include_cached, foot=foot):
		if patch is None:
			continue
		sky[imin:imin + patch.shape[0], jmin:jmin + patch.shape[1]] += patch

	return sky
###################################################################

###################################################################
## Count the number of objects in the entire catalog
def ls_mapper(rows):
	# return the number of rows in this chunk, keyed by the filename
	self = ls_mapper
	return (self.CELL_FN, len(rows))

def compute_counts(cat, include_cached=False):
	ntotal = 0
	for (file, nobjects) in cat.map_reduce(ls_mapper, include_cached=include_cached):
		ntotal = ntotal + nobjects
	return ntotal
###################################################################

###################################################################
## Cross-match two catalogs

def gnomonic(lon, lat, clon, clat):
	from numpy import sin, cos

	phi  = np.radians(lat)
	l    = np.radians(lon)
	phi1 = np.radians(clat)
	l0   = np.radians(clon)

	cosc = sin(phi1)*sin(phi) + cos(phi1)*cos(phi)*cos(l-l0)
	x = cos(phi)*sin(l-l0) / cosc
	y = (cos(phi1)*sin(phi) - sin(phi1)*cos(phi)*cos(l-l0)) / cosc

	return (np.degrees(x), np.degrees(y))

def gc_dist(lon1, lat1, lon2, lat2):
	from numpy import sin, cos, arcsin, sqrt

	lon1 = np.radians(lon1); lat1 = np.radians(lat1)
	lon2 = np.radians(lon2); lat2 = np.radians(lat2)

	return np.degrees(2*arcsin(sqrt( (sin((lat1-lat2)*0.5))**2 + cos(lat1)*cos(lat2)*(sin((lon1-lon2)*0.5))**2 )));

def _xmatch_mapper(rows, cat_to, xmatch_radius, cat_to_name):
	from scikits.ann import kdtree

	# locate cell center
	self = _xmatch_mapper
	cell_id = self.CELL_ID
	cat     = self.CATALOG
	(bounds, tbounds)  = cat.cell_bounds(cell_id)
	(clon, clat) = bhpix.deproj_bhealpix(*bounds.center())

	# Fetch data and project to tangent plane around the center
	# of the cell. We assume the cell is small enough for the
	# distortions not to matter
	if cat_to.cell_exists(cell_id):
		rows2 = cat_to.fetch_cell(cell_id=cell_id, include_cached=True)
		id1, ra1, dec1 = rows['id'], rows['ra'], rows['dec']
		id2, ra2, dec2 = rows2['id'], rows2['ra'], rows2['dec']
		xy1 = np.column_stack(gnomonic(ra1, dec1, clon, clat))
		xy2 = np.column_stack(gnomonic(ra2, dec2, clon, clat))

		# Construct kD-tree and find the nearest to each object
		# in this cell
		tree = kdtree(xy2)
		match_idx, match_d2 = tree.knn(xy1, 1)
		match_idx = match_idx[:,0]		# First neighbor only
		del tree

		# Create the index table array
		xmatch = np.empty(len(match_idx), dtype=[('id1', 'u8'), ('id2', 'u8'), ('dist', 'f4')])
		xmatch['id1'] = id1
		xmatch['id2'] = id2[match_idx]
		xmatch['dist'] = gc_dist(ra1, dec1, ra2[match_idx], dec2[match_idx])
		# Remove the matches beyond xmatch_radius
		xmatch = xmatch[xmatch['dist'] < xmatch_radius]

		if len(xmatch) != 0:
			# Store the xmatch table
			with cat.cell(cell_id, mode='w', table_type='xmatch_' + cat_to_name) as fp:
				if fp.root.xmatch.nrows != 0:
					fp.root.xmatch.removeRows(0, fp.root.xmatch.nrows)
				fp.root.xmatch.append(xmatch)

		return len(id1), len(id2), len(xmatch)
	else:
		return len(rows), 0, 0

def xmatch(cat_from, cat_to, cat_to_name, radius=1./3600.):
	""" Cross-match objects from cat_to with cat_from catalog and
	    store the result into a cross-match table in cat_from.

	    Mapper:
	    	- load all objects in a cell (incl. neighbors), make an ANN tree
	    	- stream through all objects in cat_to, find matches
	    	- store the output into an index table
	"""
	for (nfrom, nto, nmatch) in cat_from.map_reduce(_xmatch_mapper, mapper_args=(cat_to, radius, cat_to_name)):
		if nfrom != 0 and nto != 0:
			print "  ===>  %6d xmatch %6d -> %6d (%5.2f)" % (nfrom, nto, nmatch, 100. * nmatch / nfrom)

	# TODO: add metadata do dbinfo about this xmatch

###################################################################
