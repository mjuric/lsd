#!/usr/bin/env python
"""
	Common tasks needed when dealing with survey datasets.
"""

import pool2
import numpy as np
from itertools import izip
import bhpix
import sys
from utils import as_columns, gnomonic, gc_dist, unpack_callable
from colgroup import ColGroup
from join_ops import IntoWriter, DB

###################################################################
## Sky-coverage computation
def _coverage_mapper(qresult, dx, filter):
	filter, filter_args = unpack_callable(filter)

	for rows in qresult:
		assert len(rows)
		if filter is not None:
			rows = filter(rows, *filter_args)

		lon, lat = list(rows.as_columns())[:2]

		# Work around PS1 bugs:
		tofix = (lon < 0) | (lon >= 360)
		if np.any(tofix):
			#print "Fixing RIGHT ASCENSION in cell ", rows.info.cell_id
			lon[tofix] = np.fmod(np.fmod(lon[tofix], 360.) + 360., 360.)
		tofix = (lat < -90) | (lat > 90)
		if np.any(tofix):
			print "Fixing DECLINATION in cell ", rows.info.cell_id
			lat[lat < -90] = -90
			lat[lat > 90]  = 90

		i = (lon / dx).astype(int)
		j = ((90 - lat) / dx).astype(int)

		assert len(lon)
		assert len(lat)
		assert len(i)
		assert len(j)

		(imin, imax, jmin, jmax) = (i.min(), i.max(), j.min(), j.max())
		w = imax - imin + 1
		h = jmax - jmin + 1
		i -= imin; j -= jmin
		if w <= 0 or h <= 0 or w > 10800 or h > 5400:
			print w, h
			print rows.info.cell_id
			exit()
	
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
	width  = int(np.ceil(360/dx))
	height = int(np.ceil(180/dx)+1)	# The +1 is because the poles are included

	sky = np.zeros((width, height))

	for (patch, imin, jmin) in db.query(query).execute([(_coverage_mapper, dx, filter)], bounds=bounds, include_cached=include_cached):
		#print patch.shape, imin, jmin, sky.shape
		sky[imin:imin + patch.shape[0], jmin:jmin + patch.shape[1]] += patch

	print "Objects:", sky.sum()
	return sky
###################################################################

###################################################################
## Count the number of objects in the table
def ls_mapper(cell_id, db, tabname):
	# return the number of rows in this chunk, keyed by the filename
	try:
		with db.table(tabname).lock_cell(cell_id) as cell:
			with cell.open() as fp:
				n = fp.root.main.table.nrows
	except LookupError:
		# This can occur when counting from cells in previous snapshots,
		# and the cell in question was not populated there
		return

	yield n

def compute_counts_aux(db, tabname, cells, progress_callback=None):
	ntotal = 0

	pool = pool2.Pool()
	for nobjects in pool.map_reduce_chain(cells, [(ls_mapper, db, tabname)], progress_callback=progress_callback):
		ntotal += nobjects

	return ntotal

def compute_counts(db, tabname, force=False):
	if not force:
		# Use the saved count from previous snapshot, and
		# just add the rows added by this one
		assert db.in_transaction()

		# Load the count that was valid before this transaction
		db2 = DB(':'.join(db.path))
		try:
			nrows = db2.table(tabname).nrows()
			new = False
		except IOError:
			# This is a new table
			nrows = 0
			new = True

		# Fetch all cells modified by this transaction
		cells = db.table(tabname).get_cells_in_snapshot(db.snapid)

		if not new:
			# Count up rows before modification
			nrows_old = compute_counts_aux(db2, tabname, cells, progress_callback=pool2.progress_pct_nnl)
		else:
			nrows_old = 0

		# Count up rows in modified cells
		nrows_new = compute_counts_aux(db,  tabname, cells)

		# The new row count
		nrows += nrows_new - nrows_old
	else:
		# Directly count everything
		cells = db.table(tabname).get_cells()
		nrows = compute_counts_aux(db, tabname, cells)

	return nrows

###################################################################

###################################################################
## Default neighbor cache building hook
def commit_hook__build_neighbor_cache(db, table):
	db.build_neighbor_cache(table.name, snapid=table.snapid)
	print >> sys.stderr, "[%s] Updating tablet catalog:" % (table.name),
	table.rebuild_catalog()
###################################################################

###################################################################
## Cross-match two tables

def _xmatch_mapper(qresult, tabname_to, radius, tabname_xm, n_neighbors):
	"""
	    Mapper:
	    	- given all objects in a cell, make an ANN tree
	    	- load all objects in tabname_to (including neighbors), make an ANN tree, find matches
	    	- store the output into an index table
	"""
	from scipy.spatial import cKDTree as kdtree

	db       = qresult.db
	pix      = qresult.pix
	table_xm = db.table(tabname_xm)

	for rows in qresult:
		cell_id  = rows.info.cell_id

		join = ColGroup(dtype=[('_M1', 'u8'), ('_M2', 'u8'), ('_DIST', 'f4'), ('_NR', 'u1'), ('_LON', 'f8'), ('_LAT', 'f8')])

		(id1, ra1, dec1) = rows.as_columns()
		(id2, ra2, dec2) = db.query('_ID, _LON, _LAT FROM %s' % tabname_to).fetch_cell(cell_id, include_cached=True).as_columns()

		if len(id2) != 0:
			# Project to tangent plane around the center of the cell. We
			# assume the cell is small enough for the distortions not to
			# matter and Euclidian distances apply
			bounds, _    = pix.cell_bounds(cell_id)
			(clon, clat) = bhpix.deproj_bhealpix(*bounds.center())
			xy1 = np.column_stack(gnomonic(ra1, dec1, clon, clat))
			xy2 = np.column_stack(gnomonic(ra2, dec2, clon, clat))

			# Construct kD-tree to find an object in table_to that is nearest
			# to an object in table_from, for every object in table_from
			tree = kdtree(xy2)
			k=min(self.n, len(xy2))
			match_d, match_idxs = tree.query(xy1, k=k)
			del tree
			if (k == 1):
				match_idxs = match_idxs.reshape((match_idxs.shape[0],1))

			# Create the index table array
			join.resize(match_idxs.size)
			for k in xrange(match_idxs.shape[1]):
				match_idx = match_idxs[:,k]
				join['_M1'][k::match_idxs.shape[1]]   = id1
				join['_M2'][k::match_idxs.shape[1]]   = id2[match_idx]
				join['_DIST'][k::match_idxs.shape[1]] = gc_dist(ra1, dec1, ra2[match_idx], dec2[match_idx])
				join['_LON'][k::match_idxs.shape[1]]  = ra2[match_idx]
				join['_LAT'][k::match_idxs.shape[1]]  = dec2[match_idx]
				join['_NR'][k::match_idxs.shape[1]]   = k

			# Remove matches beyond the xmatch radius
			join = join[join['_DIST'] < radius]

		if len(join):
			# compute the cell_id part of the join table's
			# IDs. While this is unimportant now (as we could
			# just set all of them equal to cell_id part of
			# cell_id), if we ever decide to change the
			# pixelation of the table later on, this will
			# allow us to correctly repixelize the join table as
			# well.
			#x, y, t, _  = pix._xyti_from_id(join['_M1'])	# ... but at the spatial location given by the object table.
			#join['_ID'] = pix._id_from_xyti(x, y, t, 0)     # This will make the new IDs have zeros in the object part (so Table.append will autogen them)
			
			# TODO: Allow the stuff above (in Table.append)
			join['_ID'] = pix.cell_for_id(join['_M1'])

			# TODO: Debugging, remove when happy
			cid = np.unique(pix.cell_for_id(join['_ID']))
			assert len(cid) == 1, len(cid)
			assert cid[0] == cell_id, '%s %s' % (cid[0], cell_id)
			####

			table_xm.append(join)

			yield len(id1), len(id2), len(join)
		else:
			yield len(rows), 0, 0

xm_table_def = \
{
	'filters': { 'complevel': 5, 'complib': 'blosc', 'fletcher32': False }, # Enable compression
	'schema': {
		'main': {
			'columns': [
				('xm_id',		'u8', '', 'Unique ID of this row'),
				('m1',			'u8', '', 'ID in table one'),
				('m2',			'u8', '', 'ID in table two'),
				('dist',		'f4', '', 'Distance (in degrees)'),
				('nr',			'u1', '', 'Neighbor rank (0 == first neighbor, 1 == second neighbor, ...)'),
				# Not strictly necessary, but useful for creation of neighbor cache (needed for joins agains M2 table)
				# TODO: split this off into a separate column group
				('ra',			'f8', '', 'Position from table two'),
				('dec',			'f8', '', 'Position from table two'),
			],
			'primary_key': 'xm_id',
			'spatial_keys': ('ra', 'dec'),
		}
	},
	'aliases': {
		'_M1': 'm1',
		'_M2': 'm2',
		'_DIST': 'dist',
		'_NR': 'nr'
	}
}

def xmatch(db, tabname_from, tabname_to, radius, neighbors):
	""" Cross-match objects from tabname_to with tabname_from table and
	    store the result into a cross-match table in tabname_from.

	    Typical usage:
	    	xmatch(db, ps1_obj, sdss_obj)

	   Note:
	   	- The maximum radius is in _degrees_ (!!)
	        - No attempt is being made to force the xmatch result to be a
	          one-to-one map. In particular, more than one object from tabname_from
	          may be mapped to a same object in tabname_to
	"""
	tabname_xm = '_%s_to_%s' % (tabname_from, tabname_to)

	if not db.table_exists(tabname_xm):
		# Create the new linkage table if needed
		table_xm  = db.create_table(tabname_xm, xm_table_def)

		if tabname_from != tabname_to: # If it's a self-join (useful only for debugging), setup no JOIN relations
			# Set up a one-to-X join relationship between the two tables (join obj_table:obj_id->det_table:det_id)
			db.define_default_join(tabname_from, tabname_to,
				type = 'indirect',
				m1   = (tabname_xm, "m1"),
				m2   = (tabname_xm, "m2")
			)

			# Set up a join between the indirection table and tabname_from (mostly for debugging)
			db.define_default_join(tabname_from, tabname_xm,
				type = 'indirect',
				m1   = (tabname_xm, "m1"),
				m2   = (tabname_xm, "xm_id")
			)

			# Set up a join between the indirection table and tabname_to (mostly for debugging)
			db.define_default_join(tabname_to, tabname_xm,
				type = 'indirect',
				m1   = (tabname_xm, "m2"),
				m2   = (tabname_xm, "xm_id")
			)

	ntot = 0
	for (nfrom, nto, nmatch) in db.query("_ID, _LON, _LAT from '%s'" % tabname_from).execute(
					[ (_xmatch_mapper, tabname_to, radius, tabname_xm, neighbors) ],
					progress_callback=pool2.progress_pass):
		ntot += nmatch
		if nfrom != 0 and nto != 0:
			pctfrom = 100. * nmatch / nfrom
			pctto   = 100. * nmatch / nto
			print "  ===> %7d xmatch %7d -> %7d matched (%6.2f%%, %6.2f%%)" % (nfrom, nto, nmatch, pctfrom, pctto)
			if tabname_from == tabname_to:	# debugging: sanity check when xmatching to self
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
		cell_id = rows.info.cell_id

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
	print "%s insertions for %s objects." % (ntot, db.table('magbase').nrows())
#	for static_cell, rows in q.execute([(_accumulator, 'obj_id', 'ap_mag', 'ap_mag')], group_by_static_cell=True):
#		print static_cell, type(rows), len(rows)
#		(key, val) = rows.as_columns()
#		for k, v in izip(key, val):
#			print k, v

###################################################################
