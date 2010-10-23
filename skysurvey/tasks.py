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
	for (cell_id, nobjects) in cat.map_reduce('id', ls_mapper, include_cached=include_cached):
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

	#(id1, ra1, dec1, cat_id1, ext_id1, obj_id1, file_id1) = as_columns(rows)
	(id1, ra1, dec1) = as_columns(rows)

	rows2 = cat_to.fetch_cell(cell_id, include_cached=True)
	raKey2, decKey2 = cat_to.get_spatial_keys()
	idKey2          = cat_to.get_primary_key()
	id2, ra2, dec2 = rows2[idKey2], rows2[raKey2], rows2[decKey2]
	xy1 = np.column_stack(gnomonic(ra1, dec1, clon, clat))
	xy2 = np.column_stack(gnomonic(ra2, dec2, clon, clat))
	
	#cat_id2, ext_id2, obj_id2, file_id2 = rows2['cat_id'], rows2['ext_id'], rows2['obj_id'], rows2['file_id']

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
	matched = xmatch['d1'] < xmatch_radius
	xmatch = xmatch[matched]

#	if len(rows) and cat.name == cat_to.name:	# debugging: self-match
#		# This test revealed duplicate objects in DVO files. Example:
#		# In /raid14/panstarrs/dvo-201008/s2230/6485.06.cpt, objid = 1823 and 1824 are the same object (have the same ra/dec)
#		assert len(rows) == len(xmatch)
#		#if not (xmatch['id1'] == xmatch['id2']).all():
#		#	ra1 = ra1[matched]
#		#	ra2 = ra2[match_idx][matched]
#		#	dec1 = dec1[matched]
#		#	dec2 = dec2[match_idx][matched]
#		#	cat_id1 = cat_id1[matched]
#		#	cat_id2 = cat_id2[match_idx][matched]
#		#	ext_id1 = ext_id1[matched]
#		#	ext_id2 = ext_id2[match_idx][matched]
#		#	obj_id1 = obj_id1[matched]
#		#	obj_id2 = obj_id2[match_idx][matched]
#		#	file_id1 = file_id1[matched]
#		#	file_id2 = file_id2[match_idx][matched]
#		#	diff = xmatch['id1'] != xmatch['id2']
#		#	print xmatch['id1'][diff]
#		#	print xmatch['id2'][diff]
#		#	print xmatch['d1'][diff]
#		#	print ra1[diff]
#		#	print ra2[diff]
#		#	print dec1[diff]
#		#	print dec2[diff]
#		#	print cat_id1[diff]
#		#	print cat_id2[diff]
#		#	print ext_id1[diff]
#		#	print ext_id2[diff]
#		#	print obj_id1[diff]
#		#	print obj_id2[diff]
#		#	print file_id1[diff]
#		#	print file_id2[diff]
#		#	print cat._tablet_file(cell_id, 'astrometry')
#		assert (xmatch['id1'] == xmatch['id2']).all()
#		assert (np.abs(xmatch['d1']) < 1.e-14).all()

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
	#query = "%s, %s, %s, cat_id, ext_id, obj_id, file_id" % (idKey, raKey, decKey)
	query = "%s, %s, %s" % (idKey, raKey, decKey)

	ntot = 0
	for (nfrom, nto, nmatch) in cat_from.map_reduce(query, (_xmatch_mapper, cat_to, radius, xmatch_table)):
		ntot += nmatch
		if nfrom != 0 and nto != 0:
			print "  ===>  %6d xmatch %6d -> %6d matches (%5.2f%%)" % (nfrom, nto, nmatch, 100. * nmatch / nfrom)
			if cat_from.name == cat_to.name:	# debugging: sanity check when xmatching to self
				assert nfrom == nmatch

	print "Matched a total of %d sources." % (ntot)

	# Add metadata about this xmatch to dbinfo
	cat_from.add_xmatched_catalog(cat_to, xmatch_table)

###################################################################
