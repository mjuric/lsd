import catalog
import pyfits
import pool2
import time
import numpy as np
from slalib import sla_eqgal
from itertools import imap, izip
import copy
import itertools as it

# ra, dec, g, r, i, exposure.mjd_obs, chip.T XMATCH chip, exposure
# ra, dec, g, r, i, hdr.ctype1 XMATCH exposure

# Table defs for exposure catalog
exp_cat_def = \
{
	#
	#	 LSD column name      Type    FITS column      Description
	#
	'main': {
		'columns': [
			('exp_id',		'u8', 	'',		'LSD primary key for this exposure'   ),
			('filterid',		'a6',	'filterid',	'Filter used (instrument name)'       ),
			('equinox',		'f4',	'EQUINOX' ,	'Celestial coordinate system'	      ),
			('ra',			'f8',	'RA'	  ,	'Right Ascension of boresight'	      ),
			('dec',			'f8',	'DEC'	  ,	'Declination of boresight'	      ),
			('l',			'f8',	''	  ,	'Galactic lon of boresight'	      ),
			('b',			'f8',	''	  ,	'Galactic lat of boresight'	      ),
			('mjd_obs', 		'f8',	'MJD-OBS' ,	'Time of observation'		      ),
			('exptime', 		'f4',	'EXPTIME' ,	'Exposure time (sec)'		      ),
			('airmass',		'f4',	'AIRMASS' ,	'Observation airmass'		      ),
			('cached',		'bool',	''	  ,	'Set to True if this is a row cached from a different cell'),
		],
		'primary_key' : 'exp_id',
		'temporal_key': 'mjd_obs',
		'spatial_keys': ('ra', 'dec'),
		"cached_flag" : "cached"
	},
	'obsdet': {
		'columns': [
			('m2z',			'f4',	'M2Z'	  ,	'Telescope focus'		      ),
			('obstype',		'a20',	'OBSTYPE' ,	'Type of observation'		      ),
			('object',		'a20',	'OBJECT'  ,	'Object of observation'		      ),
			('expreq',		'f4',	'EXPREQ'  ,	'Exposure time (sec)'		      ),
			('dettem',		'f4',	'DETTEM'  ,	'Temperature of focal plane'	      ),
			('filename',		'a20',	'FILENAME', 	'FPA observation identifier'	      ),
			('nghosts',		'u4',	'NGHOSTS' ,	'total expected ghosts'	              ),
		],
	},
	'meta': {
		'columns': [
			('notfound_keys',	'i8',	''	 ,	'BLOB list listing header keys that were not found in source .smf file'),
			('notfound_cols',	'64i8',	''	 ,	'BLOB list listing columns that were not found in the source .smf file'),
		],
		'blobs': [ 'notfound_keys', 'notfound_cols' ]
	},
	'zeropoints_ifa': {
		'columns': [
			('mag_zp',		'f4',	'MAG_ZP'  ,	'Magnitude zero point'		      ),
			('zpt_obs',		'f4', 	'ZPT_OBS' ,	'measured zero point'		      ),
			('zpt_ref',		'f4', 	'ZPT_REF' ,	'reference zero point'		      ),
			('zpt_err',		'f4', 	'ZPT_ERR' ,	'error on zero point'		      ),
			('zpt_off',		'f4', 	'ZPT_OFF' ,	'zero point offset'		      ),
		],
	},
	'pointing': {
		'columns': [
			('posangle',		'f4',	'POSANGLE',	'Position angle of instrument'	      ),
			('rot',			'f4',	'ROT'	  ,	'Rotator angle of instrument'	      ),
			('alt',			'f8', 	'ALT'	  ,	'Altitude of boresight'	              ),
			('az',			'f8', 	'AZ'	  ,	'Azimuth of boresight'		      ),
			('ast_r0',		'f8', 	'AST_R0'  ,	'boresite offset in RA (TP units)'    ),
			('ast_d0',		'f8', 	'AST_D0'  ,	'boresite offset in DEC (TP units)'   ),
			('ast_t0',		'f8', 	'AST_T0'  ,	'boresite angle (degrees)'	      ),
			('ast_s0',		'f8', 	'AST_S0'  ,	'boresite scale correction'	      ),
			('ast_rs',		'f8', 	'AST_RS'  ,	'boresite scatter in RA (TP units)'   ),
			('ast_ds',		'f8', 	'AST_DS'  ,	'boresite scatter in DEC (TP units)'  ),
			('dt_astr',		'f8', 	'DT_ASTR' ,	'elapsed psastro time'		      ),
		],
	},
	'header': {
		'columns': [
			('hdr',			'i8',	'',		'Primary FITS header of .smf file'    ), # Note -- this will be a blob
			('smf_fn',		'a40',  '',		'Filename of the input smf file'      ),
		],
		'blobs': [ 'hdr' ]
	},
	'chips': {
		'columns': [
			('chip_hdr',		'64i8',	'',		'XY??.hdr FITS headers, one per chip' )	# Note -- this will be a blob
		],
		'blobs': [ 'chip_hdr' ]
	}
}

det_cat_def = \
{
	#
	#	 LSD column name      Type    FITS column      Description
	#
	'astrometry': {
		'columns': [
			('det_id',		'u8', '',		'Unique LSD ID of this detection'),
			('exp_id',		'u8', '',		'Exposure ID, joined to the image catalog'),
			('chip_id',		'u1', '',		'Index of the OTA where this object was detected (integer, 0-63)'),
			('ra',			'f8', 'ra_psf',		''),
			('dec',			'f8', 'dec_psf',	''),
			('mjd_obs', 		'f8', '',		'Time of observation'), # Copied from image header, for convenience
			('l',			'f8', '',		''),
			('b',			'f8', '',		''),
			('flags',		'u4', 'flags',		''),
			('flags2',		'u4', 'flags2',		''),
			('n_frames',		'i2', 'n_frames',	''),
			('cached',		'bool', '',		''),
		],
		'primary_key': 'det_id',
		'temporal_key': 'mjd_obs',
		'spatial_keys': ('ra', 'dec'),
		"cached_flag": "cached"
	},
	'detxy': {
		'columns': [
			('ipp_idet',		'i4', 'ipp_idet',	''),
			('x_psf',		'f4', 'x_psf',		''),
			('y_psf',		'f4', 'y_psf',		''),
			('x_psf_sig',		'f4', 'x_psf_sig',	''),
			('y_psf_sig',		'f4', 'y_psf_sig',	''),
			('posangle',		'f4', 'posangle',	''),
			('pltscale',		'f4', 'pltscale',	''),
		],
	},
	'photometry': {
		'columns': [
			('filterid',		'a6', '',			'Filter ID, read from image header (here for convenience)'),
			('psf_inst_mag',	'f4', 'psf_inst_mag',		''),
			('psf_inst_mag_sig',	'f4', 'psf_inst_mag_sig',	''),
			('psf_inst_flux',	'f4', 'psf_inst_flux'	,	''),
			('psf_inst_flux_sig',	'f4', 'psf_inst_flux_sig',	''),
			('ap_mag',		'f4', 'ap_mag',			''),
			('ap_mag_raw',		'f4', 'ap_mag_raw',		''),
			('ap_mag_radius',	'f4', 'ap_mag_radius',		''),
			('peak_flux_as_mag',	'f4', 'peak_flux_as_mag',	''),
			('cal_psf_mag',		'f4', 'cal_psf_mag',		''),
			('cal_psf_mag_sig',	'f4', 'cal_psf_mag_sig',	''),
		],
	},
	'quality': {
		'columns': [
			('sky',			'f4', 'sky',			''),
			('sky_sigma',		'f4', 'sky_sigma',		''),
			('psf_chisq',		'f4', 'psf_chisq',		''),
			('cr_nsigma',		'f4', 'cr_nsigma',		''),
			('ext_nsigma',		'f4', 'ext_nsigma',		''),
			('psf_major',		'f4', 'psf_major',		''),
			('psf_minor',		'f4', 'psf_minor',		''),
			('psf_theta',		'f4', 'psf_theta',		''),
			('psf_qf',		'f4', 'psf_qf',			''),
			('psf_qf_perfect',	'f4', 'psf_qf_perfect',		''),
			('psf_ndof',		'u4', 'psf_ndof',		''),
			('psf_npix',		'u4', 'psf_npix',		''),
		],
	},
	'moments': {
		'columns': [
			('moments_xx',		'f4', 'moments_xx',		''),
			('moments_xy',		'f4', 'moments_xy',		''),
			('moments_yy',		'f4', 'moments_yy',		''),
			('moments_m3c',		'f4', 'moments_m3c',		''),
			('moments_m3s',		'f4', 'moments_m3s',		''),
			('moments_m4c',		'f4', 'moments_m4c',		''),
			('moments_m4s',		'f4', 'moments_m4s',		''),
			('moments_r1',		'f4', 'moments_r1',		''),
			('moments_rh',		'f4', 'moments_rh',		''),
		],
	},
	'kron': {
		'columns': [
			('kron_flux',		'f4', 'kron_flux',		''),
			('kron_flux_err',	'f4', 'kron_flux_err',		''),
			('kron_flux_inner',	'f4', 'kron_flux_inner',	''),
			('kron_flux_outer',	'f4', 'kron_flux_outer',	''),
		],
	}
}

def to_dtype(cols):
	return list(( (name, dtype) for (name, dtype, _) in cols ))

def create_catalog(catdir, name, catdef):
	"""
		Creates the catalog given the extended schema description.

	"""
	cat = catalog.Catalog(catdir, name=name, mode='c')

	for tname, schema in catdef.iteritems():
		schema = copy.deepcopy(schema)
		schema['columns'] = [ (name, type) for (name, type, _, _) in schema['columns'] ]
		cat.create_table(tname, schema)

	return cat

def gen_cat2fits(catdef):
	""" Returns a mapping from catalog columns to
	    FITS columns:
		
		cat2fits[catcol] -> fitscol
	"""
	cat2fits = {}

	for tname, schema in catdef.iteritems():
		cat2fits.update( dict(( (colname, fitscol) for colname, _, fitscol, _ in schema['columns'] if fitscol != '' )) )

	return cat2fits

def gen_cat2type(catdef):
	""" Returns a mapping from column name to dtype
	"""
	
	cat2type = {}
	for tname, schema in catdef.iteritems():
		cat2type.update( dict(( (colname, dtype) for colname, dtype, _, _ in schema['columns'] )) )
		
	return cat2type

def import_from_smf(det_catdir, exp_catdir, smf_files, create=False):
	""" Import a PS1 catalog from DVO

	    Note: Assumes underlying shared storage for all catalog
	          cells (i.e., any worker is able to write to any cell).
	"""
	if create:
		# Create the new database
		det_cat = create_catalog(det_catdir, 'ps1_det', det_cat_def)
		exp_cat = create_catalog(exp_catdir, 'ps1_exp', exp_cat_def)

		# Set up a one-to-X join relationship between the two catalogs (join det_cat:exp_id->exp_cat:exp_id)
		det_cat.define_join(exp_cat, 'astrometry', 'astrometry', 'det_id', 'exp_id')
	else:
		det_cat = catalog.Catalog(det_catdir)
		exp_cat = catalog.Catalog(exp_catdir)

	det_c2f = gen_cat2fits(det_cat_def)
	exp_c2f = gen_cat2fits(exp_cat_def)
	
	t0 = time.time()
	at = 0; ntot = 0
	pool = pool2.Pool()
	for (file, nloaded, nin) in pool.imap_unordered(smf_files, import_from_smf_aux, (det_cat, exp_cat, det_c2f, exp_c2f), progress_callback=pool2.progress_pass):
	#for (file, nloaded, nin) in imap(lambda file: import_from_smf_aux(file, cat), smf_files):
		at = at + 1
		ntot = ntot + nloaded
		t1 = time.time()
		time_pass = (t1 - t0) / 60
		time_tot = time_pass / at * len(smf_files)
		print('  ===> Imported %s [%d/%d, %5.2f%%] +%-6d %9d (%.0f/%.0f min.)' % (file, at, len(smf_files), 100 * float(at) / len(smf_files), nloaded, ntot, time_pass, time_tot))

def add_lb(cols):
	(ra, dec) = cols['ra'], cols['dec']
	l = np.empty_like(ra)
	b = np.empty_like(dec)
	for i in xrange(len(ra)):
		(l[i], b[i]) = np.degrees(sla_eqgal(*np.radians((ra[i], dec[i]))))
	cols['l']      = l
	cols['b']      = b

def load_columns_from_data(dat, c2f, c2t):
	#cols = dict(( (name, dat.field(fitsname))   for (name, fitsname) in c2f.iteritems() if fitsname != ''))
	cols = dict()
	nullkeys = []
	for (name, fitsname) in c2f.iteritems():
		if fitsname == '': pass
		try:
			cols[name] = dat.field(fitsname)
		except KeyError:
			cols[name] = np.zeros(len(dat), dtype=c2t[name])
			nullkeys.append(name)
			if(cols[name].dtype.kind == 'f'):
				cols[name][:] = np.nan
			#print "Column %s does not exist." % (fitsname,)
	return cols, nullkeys

def load_columns_from_header(hdr, c2f, c2t):
	#cols = dict(( (name, np.array([hdr[fitsname]]))   for (name, fitsname) in c2f.iteritems() if fitsname != ''))
	cols = dict()
	nullkeys = []
	for (name, fitsname) in c2f.iteritems():
		if fitsname == '': pass
		try:
			cols[name] = np.array([ hdr[fitsname] ])
		except KeyError:
			cols[name] = np.zeros(1, dtype=c2t[name])
			nullkeys.append(name)
			if(cols[name].dtype.kind == 'f'):
				cols[name][:] = np.nan
			#print "Key %s does not exist in header." % (fitsname,)
	return cols, nullkeys

def all_chips():
	""" Iterate over all valid chip IDs.
	
	    Returns chip_id = y*8+x, and 'XY??' string (with
	     ?? replaced by corresponding x, y)
	"""
	for (x, y) in it.product(xrange(8), xrange(8)):
		if (x == 0 or x == 7) and (y == 0 or y == 7):		# No chips on the edges
			continue

		chip_xy = 'XY%1d%1d' % (x, y)
		chip_id = x + y*8

		yield chip_id, chip_xy

def import_from_smf_aux(file, det_cat, exp_cat, det_c2f, exp_c2f):
	det_c2t = gen_cat2type(det_cat_def)
	exp_c2t = gen_cat2type(exp_cat_def)

	# Load the .smf file
	hdus = pyfits.open(file)

	# Do the exposure table entry first; we'll need the
	# image ID to fill in the detections table
	imhdr = hdus[0].header
	exp_cols, exp_nullcols = load_columns_from_header(imhdr, exp_c2f, exp_c2t)
	add_lb(exp_cols)

	# Record the list of keywords not found in the header
	# and prepare the lists of columns not found in the header
	exp_cols['notfound_keys'] = np.empty(1, dtype=object)
	if len(exp_nullcols):
		exp_cols['notfound_keys'][0] = exp_nullcols
	exp_cols['notfound_cols']	= np.empty(1, dtype='64O')
	exp_cols['notfound_cols'][:]	= None

	# Add the filename of the .smf we drew this from
	fn = '/'.join(file.split('/')[-2:])
	exp_cols['smf_fn'] = np.empty(1, dtype='a40')
	exp_cols['smf_fn'][:] = fn

	# Add the main image, and all chip headers
	exp_cols['hdr']		= np.array([str(imhdr)], dtype=object)
	exp_cols['chip_hdr']	= np.empty(1, dtype='64O')
	exp_cols['chip_hdr'][:] = None
	nrows = 0
	for (idx, (chip_id, chip_xy)) in enumerate(all_chips()):
		# Workaround because .smf files have a bug, listing the EXTNAME keyword
		# twice for each XY??.hdr HDU
		try:
			i = hdus.index_of(chip_xy + '.psf') - 1
		except KeyError:
			pass
			##print "Warning: Extension %s.psf not found in file %s." % (chip_xy, file)
		else:
			hdr = hdus[i].header
			exp_cols['chip_hdr'][0][chip_id] = str(hdr)

			dat = hdus[chip_xy + '.psf'].data
			if dat is not None:
				nrows = nrows + len(dat)

	### Detections
	filterid = imhdr['FILTERID']
	mjd_obs  = imhdr['MJD-OBS']
	det_cols_all = None
	at = 0
	for (idx, (chip_id, chip_xy)) in enumerate(all_chips()):

		try:	
			dat = hdus[chip_xy + '.psf'].data
			if dat is None:
				continue
		except KeyError:
			continue

		# Slurp all columns from FITS to 
		det_cols, nullcols  = load_columns_from_data(dat, det_c2f, det_c2t)

		# Record any columns that were not found
		if len(nullcols):
			exp_cols['notfound_cols'][0][chip_id] = nullcols

		# Add computed columns
		add_lb(det_cols)
		det_cols['filterid']    = np.empty(len(dat), dtype='a6')
		det_cols['mjd_obs']     = np.empty(len(dat), dtype='f8')
		det_cols['exp_id']      = np.empty(len(dat), dtype='u8')
		det_cols['chip_id']     = np.empty(len(dat), dtype='u1')
		det_cols['filterid'][:] = filterid
		det_cols['mjd_obs'][:]  = mjd_obs
		det_cols['chip_id'][:]  = chip_id

		# Create output array, if needed
		if det_cols_all is None:
			det_cols_all = dict(( (col, np.empty(nrows, dtype=det_c2t[col])) for col in det_cols ))

		# Store to output array
		for col in det_cols_all:
			det_cols_all[col][at:at+len(det_cols[col])] = det_cols[col]
		at = at + len(det_cols['exp_id'])
	assert at == nrows

	(exp_id,) = exp_cat.append(exp_cols)
	det_cols_all['exp_id'][:] = exp_id

	ids = det_cat.append(det_cols_all)

	return (file, len(ids), len(ids))

#########

# image cache creator kernels
# 1) stream through entire detection catalog, recording exp_ids of images
#    from other cells. Key these by their cell_id, our cell_id, and return.
# 2) stream through the image catalog, collecting rows of images that are
#    referenced out of their cells. Return the rows keyed by destination
#    cells.
# 3) store the copies into the cache of each cell.

def make_image_cache(det_cat_path, exp_cat_path):
	# Entry point for creation of image cache
	det_cat = catalog.Catalog(det_cat_path)
	exp_cat = catalog.Catalog(exp_cat_path)

	# Fetch all non-empty cells with detections. This will be the
	# list over which the first kernel will map.
	det_cells = det_cat.get_cells()

	pool = pool2.Pool()
	for _ in pool.map_reduce_chain(det_cells,
					      [
						(_exp_id_gather,  det_cat, exp_cat),
						(_exp_id_load,    exp_cat),
						(_exp_store_rows, exp_cat)
					      ]):
		pass;

def _exp_id_gather(det_cell, det_cat, exp_cat):
	#-- This kernel is called once for each cell in det_cat.

	# Fetch all exp_ids referenced from this cell
	(exp_id, ) = det_cat.query_cell(det_cell, 'exp_id').as_columns()
	exp_id     = np.unique(exp_id)
	exp_cells  = exp_cat.cell_for_id(exp_id)

	# keep only those that aren't local
	outside    = exp_cells != det_cell
	exp_id     =    exp_id[outside]
	exp_cells  = exp_cells[outside]

	# pack them in (exp_cell, (det_cell, exposures)) tuples
	ret = [];
	for exp_cell in set(exp_cells):
		# List of exposure IDs from exp_cell, that need
		# to be cached in det_cell
		exps = exp_id[exp_cells == exp_cell]

		ret.append( (exp_cell, (det_cell, exps)) )

	return ret

def _exp_id_load(kv, exp_cat):
	#-- This kernel is called once for each exp_cat cell referenced from det_cat
	#
	# kv = (exp_cell, detexps) with
	#    detexps = [ (det_cell1, exps1), (det_cell, exps2), ... ]
	# where exps are the list of exp_ids to load, and det_cell
	# are the cells that should get a copy of each of those
	# exposure IDs.
	#

	exp_cell, detexps = kv

	# Load exp_id and sort it
	(exp_id,) = exp_cat.query_cell(exp_cell, 'exp_id').as_columns()
	sorted_idx = np.argsort(exp_id)
	exp_id = exp_id[sorted_idx]

	# Find all rows that appear in any of the detexps lists
	in_ = np.zeros(len(exp_id), dtype=bool)
	idx_list = {}
	for (det_cell, exps) in detexps:
		idx = np.searchsorted(exp_id, exps)
		assert (exp_id[idx] == exps).all()		# It's a bug if there are exp_id duplicates or unmatched exp_ids

		in_[idx] = True
		idx_list[det_cell] = sorted_idx[idx]

	# return in_ to original (pre-sorting) ordering
	in_[sorted_idx] = in_.copy()
	idxin = np.arange(len(in_))[in_]			# Indexes of tablet rows that will be loaded

	# Load full rows
	allrows = catalog.load_full_rows(exp_cat, exp_cell, in_)

	# Dispatch the right rows to right det_cells
	ret = []
	for (det_cell, idx_) in idx_list.iteritems():
		idx = np.searchsorted(idxin, idx_)		# idxall are indices in the full table. Reduce to only those appearing after culling by in_
		assert (idxin[idx] == idx_).all()

		rows = catalog.extract_full_rows_subset(allrows, idx)
		ret.append( (det_cell, rows) )

	#print exp_cell, " -> ", [ (det_cell, len(rows[exp_cat.primary_table]['rows'])) for (det_cell, rows) in ret ]

	# Returns a list of (dec_cell, fullrows) tuples
	return ret

def _exp_store_rows(kv, exp_cat):
	# Store all rows to the correct cells.
	cell_id, nborblocks = kv

	ncached = catalog.write_neighbor_cache(exp_cat, cell_id, nborblocks);

	# Return the number of new rows cached into this cell
	return (cell_id, ncached)

###############
# object catalog creator

def make_object_catalog(obj_catdir, det_catdir, create):
	# Entry point for creation of image cache
	det_cat = catalog.Catalog(det_catdir)

	if create:
		# Create the new object database
		obj_cat = create_catalog(obj_catdir, 'ps1_det', obj_cat_def)
	else:
		obj_cat = catalog.Catalog(obj_catdir)

	# Set up join relationship and table between the object and detections catalog
	xmatch_table = 'xmatch_' + det_cat.name
	cat_from.create_table(xmatch_table, { 'columns': [('obj_id', 'u8'), ('det_id', 'u8'), ('d1', 'f4')] }, ignore_if_exists=True, hidden=True)
	cat_from.define_join(cat_to, xmatch_table, xmatch_table, 'obj_id', 'det_id')

	# Fetch all non-empty cells with detections. This will be the
	# list over which the first kernel will map.
	det_cells = det_cat.get_cells()

	pool = pool2.Pool()
	for _ in pool.map_reduce_chain(det_cells,
					      [
						(_det_coord_gather, det_cat, obj_cat),
						(_obj_match,        obj_cat),
					      ]):
		pass;

# Single-pass detections->objects mapper.
##def _xmatch_mapper(rows, cat_to, xmatch_radius, join_table):
def _obj_det_match(obj_cell, obj_cat, det_cat, radius, join_table):
	# This kernel assumes:
	#   a) det_cat and obj_cat have equal partitioning (equally sized/enumerated spatial cells)
	#   b) det_cat cells have neighbor caches
	#   c) temporal det_cat cells within this spatial cell are stored local to this process

	from scikits.ann import kdtree

	# get a list of detection cells corresponding to this static sky cell
	det_cells = det_cat.get_t_cells(obj_cell)
	assert len(det_cells)

	# locate cell center
	(bounds, tbounds)  = obj_cat.cell_bounds(cell_id)
	(clon, clat) = bhpix.deproj_bhealpix(*bounds.center())

	# Fetch data and project to tangent plane around the center
	# of the cell. We assume the cell is small enough for the
	# distortions not to matter

	# fetch existing catalog data, convert to gnomonic
	raKey, decKey = obj_cat.get_spatial_keys()
	idKey         = obj_cat.get_primary_key()
	objects = obj_cat.query_cell(det_cell, '%s, %s, %s' % (idKey, raKey, decKey), include_cached=True)
	(id1, ra1, dec1) = objects.as_columns()
	xy1 = np.column_stack(gnomonic(ra1, dec1, clon, clat))

	# prep detections query (we'll execute it repeatedly)
	raKey2, decKey2 = det_cat.get_spatial_keys()
	idKey2          = det_cat.get_primary_key()
	det_query = '%s, %s, %s' % (idKey2, raKey2, decKey2)

	# prep join table
	jdtype = np.dtype(cat._get_schema(join_table)['columns'])
	objKey, detKey, distKey = jdtype.names[0:3]			# Extract column names
	joins = []
	newobj = []

	# Loop and xmatch
	nobj = 0; njoins = 0;
	for det_cell in det_cells:
		# fetch detections, convert to gnomonic
		rows2 = det_cat.query_cell(cell_id, det_query, include_cached=True)
		id2, ra2, dec2 = rows2.as_columns()
		xy2 = np.column_stack(gnomonic(ra2, dec2, clon, clat))

		# Construct kD-tree and find the nearest to each object
		# in this cell
		tree = kdtree(xy1)
		match_idx, match_d2 = tree.knn(xy2, 1)
		del tree
		match_idx = match_idx[:,0]		# First neighbor only

		dist = gc_dist(ra1[match_idx], dec1[match_idx], ra2, dec2)
		matched   = dist < radius
		unmatched = matched == False

		# Extract matches
		rows = np.empty(len(match_idx), dtype=jdtype)
		rows[objKey]  = match_idx[matched]
		rows[detKey]  = id2[matched]
		rows[distKey] = dist[matched]
		joins.append(matches[matched])
		njoins += len(joins[-1])

		# Promote unmatched detections to objects
		newobj.append((ra2[unmatched], dec2[unmatched]))
		nobj += len(newobj[-1][0])

		# Append the gnomonic coordinates of new objects
		xy1 = np.resize(xy1, xy2[unmatched])

	# Coalesce newobj and joins arrays into single tables
	jrows = np.empty(njoins, dtype=jdtype)
	at = 0
	for rows in joins:
		jrows[at:at+len(rows)] = rows
		at += len(rows)
	joins = jrows

	ra  = np.empty(nobj, dtype=ra1.dtype)
	dec = np.empty_like(ra)
	at = 0
	for (ra_, dec_) in newobj:
		ra [at:at+len(ra_)]  = ra_
		dec[at:at+len(dec_)] = dec_
		at += len(ra_)
	newobj = Table(cols=[(raKey, ra), (decKey, dec)])

	# Drop new objects that fall past cell boundaries. These will
	# be picked up by processing in their parent cells.
	(x, y) = bhpix.proj_bhealpix(ra, dec)
	in_ = np.fromiter( (not p.isInside(px, py) for (px, py) in izip(x, y)), dtype=np.bool, count=len(x))
	idx = np.arange(len(in_))[in_]
	joins  = joins[ np.in1d(joins[objKey], idx) ]
	newobj = newobj[ in_ ]

	# Append new objects to the object catalog, thus getting the obj_ids.
	newids = obj_cat.append(newobj)

	# Change the index to obj_id in join table
	ids = np.append(id1, newids)
	joins[objKey] = ids[joins[objKey]]

	# Append the join table
	if len(joins) != 0:
		# Store the xmatch table
		#obj_cat._drop_tablet(cell_id, join_table)
		obj_cat._append_tablet(cell_id, join_table, joins)

		return len(id1), len(id2), len(joins)
	else:
		return len(rows), 0, 0
