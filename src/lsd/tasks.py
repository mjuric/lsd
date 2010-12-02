#!/usr/bin/env python
"""
	Common tasks needed when dealing with survey datasets.
"""

import pool2
import numpy as np
from itertools import izip
import bhpix
from utils import as_columns, gnomonic, gc_dist, unpack_callable
from colgroup import ColGroup
from join_ops import IntoWriter

###################################################################
## Sky-coverage computation
def _coverage_mapper(qresult, dx, filter):
	filter, filter_args = unpack_callable(filter)

	for rows in qresult:
		if filter is not None:
			rows = filter(rows, *filter_args)

		lon, lat = list(rows.as_columns())[:2]
	
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

		yield (sky, imin, jmin)

def compute_coverage(db, query, dx = 0.5, bounds=None, include_cached=False, filter=None):
	""" compute_coverage - produce a sky map of coverage, using
	    a filter function if given. The output is a starcount
	    array in (ra, dec) binned to <dx> resolution.
	"""
	width  = int(round(360/dx))
	height = int(round(180/dx))

	sky = np.zeros((width, height))

	for (patch, imin, jmin) in db.query(query).execute([(_coverage_mapper, dx, filter)], bounds=bounds, include_cached=include_cached):
		sky[imin:imin + patch.shape[0], jmin:jmin + patch.shape[1]] += patch

	print "Objects:", sky.sum()
	return sky
###################################################################

###################################################################
## Count the number of objects in the entire catalog
def ls_mapper(qresult):
	# return the number of rows in this chunk, keyed by the filename
	for rows in qresult:
		yield rows.cell_id, len(rows)

def compute_counts(db, cat, include_cached=False):
	ntotal = 0
	for (_, nobjects) in db.query("_ID FROM '%s'" % cat).execute([ls_mapper], include_cached=include_cached):
		ntotal = ntotal + nobjects
	return ntotal
###################################################################

###################################################################
## Cross-match two catalogs

def _xmatch_mapper(qresult, cat_to_dir, radius, xm_catdir):
	"""
	    Mapper:
	    	- given all objects in a cell, make an ANN tree
	    	- load all objects in cat_to (including neighbors), make an ANN tree, find matches
	    	- store the output into an index table
	"""
	from scikits.ann import kdtree

	db     = qresult.db
	pix    = qresult.pix
	cat_to = db.catalog(cat_to_dir)
	cat_xm = db.catalog(xm_catdir)

	for rows in qresult:
		cell_id  = rows.cell_id

		join = ColGroup()

		(id1, ra1, dec1) = rows.as_columns()
		(id2, ra2, dec2) = db.query('_ID, _LON, _LAT FROM %s' % cat_to_dir).fetch_cell(cell_id, include_cached=True).as_columns()

		if len(id2) != 0:
			# Project to tangent plane around the center of the cell. We
			# assume the cell is small enough for the distortions not to
			# matter and Euclidian distances apply
			bounds, _    = pix.cell_bounds(cell_id)
			(clon, clat) = bhpix.deproj_bhealpix(*bounds.center())
			xy1 = np.column_stack(gnomonic(ra1, dec1, clon, clat))
			xy2 = np.column_stack(gnomonic(ra2, dec2, clon, clat))

			# Construct kD-tree to find an object in cat_to that is nearest
			# to an object in cat_from, for every object in cat_from
			tree = kdtree(xy2)
			match_idx, match_d2 = tree.knn(xy1, 1)
			del tree
			match_idx = match_idx[:,0]		# Consider first neighbor only

			# Create the index table array
			join['_M1']   = id1
			join['_M2']   = id2[match_idx]
			join['_DIST'] = gc_dist(ra1, dec1, ra2[match_idx], dec2[match_idx])
			join['_LON']  = ra2[match_idx]
			join['_LAT']  = dec2[match_idx]

			# Remove matches beyond the xmatch radius
			join = join[join['_DIST'] < radius]

		if len(join):
			# compute the cell_id part of the join table's
			# IDs. While this is unimportant now (as we could
			# just set all of them equal to cell_id part of
			# cell_id), if we ever decide to change the
			# pixelation of the catalog later on, this will
			# allow us to correctly repixelize the join table as
			# well.
			#x, y, t, _  = pix._xyti_from_id(join['_M1'])	# ... but at the spatial location given by the object catalog.
			#join['_ID'] = pix._id_from_xyti(x, y, t, 0)     # This will make the new IDs have zeros in the object part (so Table.append will autogen them)
			join['_ID'] = pix.cell_for_id(join['_M1'])

			# TODO: Debugging, remove when happy
			cid = np.unique(pix.cell_for_id(join['_ID']))
			assert len(cid) == 1, len(cid)
			assert cid[0] == cell_id, '%s %s' % (cid[0], cell_id)
			####

			cat_xm.append(join)

			yield len(id1), len(id2), len(join)
		else:
			yield len(rows), 0, 0

xm_cat_def = \
{
	'filters': { 'complevel': 1, 'complib': 'zlib', 'fletcher32': True }, # Enable compression and checksumming
	'schema': {
		'main': {
			'columns': [
				('xm_id',		'u8', '', 'Unique ID of this row'),
				('m1',			'u8', '', 'ID in catalog one'),
				('m2',			'u8', '', 'ID in catalog two'),
				('dist',		'f4', '', 'Distance (in degrees)'),
				# Not strictly necessary, but useful for creation of neighbor cache (needed for joins agains M2 table)
				# TODO: split this off into a separate column group
				('ra',			'f8', '', 'Position from catalog two'),
				('dec',			'f8', '', 'Position from catalog two'),
			],
			'primary_key': 'xm_id',
			'spatial_keys': ('ra', 'dec'),
		}
	},
	'aliases': {
		'_M1': 'm1',
		'_M2': 'm2',
		'_DIST': 'dist'
	}
}

def xmatch(db, cat_from_dir, cat_to_dir, radius=1./3600.):
	""" Cross-match objects from cat_to with cat_from catalog and
	    store the result into a cross-match table in cat_from.

	    Typical usage:
	    	xmatch(db, ps1_obj, sdss_obj)

	   Note:
	        - No attempt is being made to force the xmatch result to be a
	          one-to-one map. In particular, more than one object from cat_from
	          may be mapped to a same object in cat_to
	"""
	# Create the x-match table and setup the JOIN relationship

	xm_catdir = '_%s_to_%s' % (cat_from_dir, cat_to_dir)

	if not db.catalog_exists(xm_catdir):
		# Create the new linkage table if needed
		xm_cat  = db.create_catalog(xm_catdir, xm_cat_def)

		if cat_from_dir != cat_to_dir: # If it's a self-join (useful only for debugging), setup no JOIN relations
			# Set up a one-to-X join relationship between the two catalogs (join obj_cat:obj_id->det_cat:det_id)
			db.define_default_join(cat_from_dir, cat_to_dir,
				type = 'indirect',
				m1   = (xm_catdir, "m1"),
				m2   = (xm_catdir, "m2")
			)

			# Set up a join between the indirection table and cat_from (mostly for debugging)
			db.define_default_join(cat_from_dir, xm_catdir,
				type = 'indirect',
				m1   = (xm_catdir, "m1"),
				m2   = (xm_catdir, "xm_id")
			)

			# Set up a join between the indirection table and cat_to (mostly for debugging)
			db.define_default_join(cat_to_dir, xm_catdir,
				type = 'indirect',
				m1   = (xm_catdir, "m2"),
				m2   = (xm_catdir, "xm_id")
			)

	ntot = 0
	for (nfrom, nto, nmatch) in db.query("_ID, _LON, _LAT from '%s'" % cat_from_dir).execute(
					[ (_xmatch_mapper, cat_to_dir, radius, xm_catdir) ],
					progress_callback=pool2.progress_pass):
		ntot += nmatch
		if nfrom != 0 and nto != 0:
			pctfrom = 100. * nmatch / nfrom
			pctto   = 100. * nmatch / nto
			print "  ===> %7d xmatch %7d -> %7d matched (%6.2f%%, %6.2f%%)" % (nfrom, nto, nmatch, pctfrom, pctto)
			if cat_from_dir == cat_to_dir:	# debugging: sanity check when xmatching to self
				assert nfrom == nmatch

	print "Matched a total of %d sources." % (ntot)

###################################################################

def _accumulator(qresult, key, val, oval):
	from collections import defaultdict
#	yield 0, [1]
#	return

	static_cell = None

	accum = defaultdict(list)
	for rows in qresult:
		kdtype = rows[key].dtype
		vdtype = rows[val].dtype
		cell_id = rows.cell_id

		if static_cell == None:
			static_cell = qresult.pix.static_cell_for_cell(cell_id)
		else:
			assert qresult.pix.static_cell_for_cell(cell_id) == static_cell

		for k, v in izip(rows[key], rows[val]):
			accum[k].append(v)

	if accum:
		# Convert to jagged object array
		keys = np.fromiter(accum.iterkeys(), dtype=kdtype, count=len(accum))
		vals = np.empty(len(accum), dtype=object)
		for (i, v) in enumerate(accum.itervalues()):
			vals[i] = np.array(v, dtype=vdtype)
#			vals[i] = v

		# Return it as a ColGroup(), keyed to this static cell_id
		yield static_cell, ColGroup([(key, keys), (oval, vals)])

def _accumulate_and_write(qresult, qwriter, key, val, oval):
	for static_cell, rows in _accumulator(qresult, key, val, oval):
		result = qwriter.write(static_cell, rows)
		yield result
#		yield 0, [1, 2]

if __name__ == '__main__':
	from join_ops import DB
	ntot = 0
	db = DB('db2')
	writer = IntoWriter(db, "magbase WHERE obj_id |= obj_id")
	for band in 'grizy':
#	for band in 'g':
		nband = 0
		q = db.query("obj_id, ap_mag, filterid FROM obj, det WHERE filterid == '%s.0000'" % band)
		for static_cell, rows in q.execute([(_accumulate_and_write, writer, 'obj_id', 'ap_mag', band)], group_by_static_cell=True):
			nband += len(rows)
##		q = db.query("obj_id, ap_mag, filterid FROM obj, det WHERE filterid == 'y.0000' INTO magbase")
##		for static_cell, rows in q.execute([(_accumulator, 'obj_id', 'ap_mag', 'ap_mag')], group_by_static_cell=True):
##			nband += len(rows)
		ntot += nband
		print "%s objects in band %s" % (nband, band)
	db.compute_summary_stats('magbase')
	print "%s insertions for %s objects." % (ntot, db.catalog('magbase').nrows())
#	for static_cell, rows in q.execute([(_accumulator, 'obj_id', 'ap_mag', 'ap_mag')], group_by_static_cell=True):
#		print static_cell, type(rows), len(rows)
#		(key, val) = rows.as_columns()
#		for k, v in izip(key, val):
#			print k, v

###################################################################
