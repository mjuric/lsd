"""
	Common tasks needed when dealing with survey datasets.
"""

import pool2
import numpy as np
from itertools import izip
import bhpix
import catalog
from utils import as_columns, gnomonic, gc_dist

###################################################################
## Sky-coverage computation
def _coverage_mapper(rows, dx = 1., filter=None, filter_args=()):
	self = _coverage_mapper

	if filter is not None:
		rows = filter(rows, *filter_args)

	if len(rows) == 0:
		return (None, None, None, None)

	lon, lat = as_columns(rows, 2)

	i = (lon / dx).astype(int)
	j = ((90 - lat) / dx).astype(int)

	(imin, imax, jmin, jmax) = (i.min(), i.max(), j.min(), j.max())
	w = imax - imin + 1
	h = jmax - jmin + 1
	i -= imin; j -= jmin

	if False:
		# Binning (method #1, straightforward but slow)
		sky = np.zeros((w, h))
		for (ii, jj) in izip(i, j):
			sky[ii, jj] += 1
	else:
		# Binning (method #2, fast)
		sky2 = np.zeros(w*h)
		idx = np.bincount(j + i*h)
		sky2[0:len(idx)] = idx
		sky = sky2.reshape((w, h))

	#assert not (sky-sky2).any()

	return (sky, imin, jmin)

def compute_coverage(cat, dx = 0.5, include_cached=False, filter=None, filter_args=(), foot=catalog.All, query='ra, dec'):
	""" compute_coverage - produce a sky map of coverage, using
	    a filter function if given. The output is a starcount
	    array in (ra, dec) binned to <dx> resolution.
	"""
	width  = int(round(360/dx))
	height = int(round(180/dx))

	sky = np.zeros((width, height))

	for (patch, imin, jmin) in cat.map_reduce(_coverage_mapper, mapper_args=(dx, filter, filter_args), query=query, include_cached=include_cached, foot=foot):
		if patch is None:
			continue
		sky[imin:imin + patch.shape[0], jmin:jmin + patch.shape[1]] += patch

	print "Objects:", sky.sum()
	return sky
###################################################################

###################################################################
## Count the number of objects in the entire catalog
def ls_mapper(rows):
	# return the number of rows in this chunk, keyed by the filename
	self = ls_mapper
	return (self.CELL_ID, len(rows))

def compute_counts(cat, include_cached=False):
	ntotal = 0
	for (cell_id, nobjects) in cat.map_reduce(ls_mapper, query='id', include_cached=include_cached):
		ntotal = ntotal + nobjects
	return ntotal
###################################################################

###################################################################
## Cross-match two catalogs

def _xmatch_mapper(rows, cat_to, xmatch_radius, xmatch_table):
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
	if not cat_to.tablet_exists(cell_id):
		return len(rows), 0, 0

	(id1, ra1, dec1) = as_columns(rows)

	rows2 = cat_to.fetch_cell(cell_id, include_cached=True)
	raKey2, decKey2 = cat_to.get_spatial_keys()
	idKey2          = cat_to.get_primary_key()
	id2, ra2, dec2 = rows2[idKey2], rows2[raKey2], rows2[decKey2]
	xy1 = np.column_stack(gnomonic(ra1, dec1, clon, clat))
	xy2 = np.column_stack(gnomonic(ra2, dec2, clon, clat))

	# Construct kD-tree and find the nearest to each object
	# in this cell
	tree = kdtree(xy2)
	match_idx, match_d2 = tree.knn(xy1, 1)
	del tree
	match_idx = match_idx[:,0]		# First neighbor only

	# Create the index table array
	xmatch = np.empty(len(match_idx), dtype=np.dtype(cat._get_schema(xmatch_table)['columns']))
	xmatch['id1'] = id1
	xmatch['id2'] = id2[match_idx]
	xmatch['d1']  = gc_dist(ra1, dec1, ra2[match_idx], dec2[match_idx])
	# Remove the matches beyond xmatch_radius
	xmatch = xmatch[xmatch['d1'] < xmatch_radius]

	if len(xmatch) != 0:
		# Store the xmatch table
		cat._drop_tablet(cell_id, xmatch_table)
		cat._append_tablet(cell_id, xmatch_table, xmatch)

		return len(id1), len(id2), len(xmatch)
	else:
		return len(rows), 0, 0

def xmatch(cat_from, cat_to, radius=1./3600.):
	""" Cross-match objects from cat_to with cat_from catalog and
	    store the result into a cross-match table in cat_from.

	    Mapper:
	    	- load all objects in a cell (incl. neighbors), make an ANN tree
	    	- stream through all objects in cat_to, find matches
	    	- store the output into an index table
	"""
	# Create the x-match table
	xmatch_table = 'xmatch_' + cat_to.name
	cat_from.create_table(xmatch_table, { 'columns': [('id1', 'u8'), ('id2', 'u8'), ('d1', 'f4')] }, ignore_if_exists=True, hidden=True)

	raKey, decKey = cat_from.get_spatial_keys()
	idKey         = cat_from.get_primary_key()
	query = "%s, %s, %s" % (idKey, raKey, decKey)

	#for (nfrom, nto, nmatch) in cat_from.map_reduce(_xmatch_mapper, query=query, mapper_args=(cat_to, radius, xmatch_table)):
	#	if nfrom != 0 and nto != 0:
	#		print "  ===>  %6d xmatch %6d -> %6d matches (%5.2f%%)" % (nfrom, nto, nmatch, 100. * nmatch / nfrom)

	# Add metadata about this xmatch to dbinfo
	cat_from.add_xmatched_catalog(cat_to, xmatch_table)

###################################################################
