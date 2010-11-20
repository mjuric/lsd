"""
	Common tasks needed when dealing with survey datasets.
"""

import pool2
import numpy as np
from itertools import izip
import bhpix
import catalog
from utils import as_columns, gnomonic, gc_dist, unpack_callable

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

def compute_coverage(cat, dx = 0.5, include_cached=False, filter=None, foot=catalog.All, query='ra, dec'):
	""" compute_coverage - produce a sky map of coverage, using
	    a filter function if given. The output is a starcount
	    array in (ra, dec) binned to <dx> resolution.
	"""
	filter, filter_args = unpack_callable(filter)

	width  = int(round(360/dx))
	height = int(round(180/dx))

	sky = np.zeros((width, height))

	for (patch, imin, jmin) in cat.map_reduce(query, (_coverage_mapper, dx, filter, filter_args), foot=foot, include_cached=include_cached):
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
	primary_key = cat._get_schema(cat.primary_table)['primary_key']
	for (_, nobjects) in cat.map_reduce(primary_key, ls_mapper, include_cached=include_cached):
		ntotal = ntotal + nobjects
	return ntotal
###################################################################

###################################################################
## Cross-match two catalogs

def _xmatch_mapper(rows, cat_to, radius, join_table):
	"""
	    Mapper:
	    	- given all objects in a cell, make an ANN tree
	    	- load all objects in cat_to (including neighbors), make an ANN tree, find matches
	    	- store the output into an index table
	"""
	from scikits.ann import kdtree

	# locate cell center
	self     = _xmatch_mapper
	cell_id  = self.CELL_ID
	cat      = self.CATALOG

	bounds, _    = cat.cell_bounds(cell_id)
	(clon, clat) = bhpix.deproj_bhealpix(*bounds.center())

	# Fetch data and project to tangent plane around the center
	# of the cell. We assume the cell is small enough for the
	# distortions not to matter
	if not cat_to.tablet_exists(cell_id):
		return len(rows), 0, 0

	(id1, ra1, dec1) = rows.as_columns()
	(id2, ra2, dec2) = cat_to.query_cell(cell_id, '_ID, _LON, _LAT', include_cached=True).as_columns()
	xy1 = np.column_stack(gnomonic(ra1, dec1, clon, clat))
	xy2 = np.column_stack(gnomonic(ra2, dec2, clon, clat))

	# Construct kD-tree to find an object in cat_to that is nearest
	# to an object in cat_from, for every object in cat_from
	tree = kdtree(xy2)
	match_idx, match_d2 = tree.knn(xy1, 1)
	del tree
	match_idx = match_idx[:,0]		# Consider first neighbor only

	# Create the index table array
	join = np.empty(len(match_idx), dtype=cat._get_schema(join_table)['columns'])
	join['id1'] = id1
	join['id2'] = id2[match_idx]
	join['d1']  = gc_dist(ra1, dec1, ra2[match_idx], dec2[match_idx])

	# Remove matches beyond the xmatch radius
	join = join[join['d1'] < radius]

	if len(join) != 0:
		# Store the join table
		cat._drop_tablet(cell_id, join_table)
		cat._append_tablet(cell_id, join_table, join)

		return len(id1), len(id2), len(join)
	else:
		return len(rows), 0, 0

def xmatch(cat_from, cat_to, radius=1./3600.):
	""" Cross-match objects from cat_to with cat_from catalog and
	    store the result into a cross-match table in cat_from.

	    Typical usage:
	    	xmatch(ps1_obj, sdss_obj)

	   Note:
	        - No attempt is being made to force the xmatch result to be a
	          one-to-one map. In particular, more than one object from cat_from
	          may be mapped to a same object in cat_to
	"""
	# Create the x-match table and setup the JOIN relationship
	join_table = 'join:' + cat_to.name
	cat_from.create_table(join_table, { 'columns': [('id1', 'u8'), ('id2', 'u8'), ('d1', 'f4')] }, ignore_if_exists=True, hidden=True)

	ntot = 0
	for (nfrom, nto, nmatch) in cat_from.map_reduce("_ID, _LON, _LAT", (_xmatch_mapper, cat_to, radius, join_table), progress_callback=pool2.progress_pass):
		ntot += nmatch
		if nfrom != 0 and nto != 0:
			pctfrom = 100. * nmatch / nfrom
			pctto   = 100. * nmatch / nto
			print "  ===> %7d xmatch %7d -> %7d matched (%6.2f%%, %6.2f%%)" % (nfrom, nto, nmatch, pctfrom, pctto)
			if cat_from.name == cat_to.name:	# debugging: sanity check when xmatching to self
				assert nfrom == nmatch

	cat_from.define_join(cat_to, join_table, join_table, 'id1', 'id2')
	print "Matched a total of %d sources." % (ntot)

###################################################################
