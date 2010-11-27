#!/usr/bin/env python

import pyfits
import pool2
import time
import numpy as np
from slalib import sla_eqgal
from itertools import imap, izip
import copy
import itertools as it
import bhpix
from utils import gnomonic, gc_dist
import table
import sys, os

# ra, dec, g, r, i, exposure.mjd_obs, chip.T XMATCH chip, exposure
# ra, dec, g, r, i, hdr.ctype1 XMATCH exposure

# Table defs for exposure catalog
exp_cat_def = \
{
	'fgroups': {
		'hdr': {
			'filter': ('bzip2', {'compresslevel': 9})
		}
	},
	'filters': { 'complevel': 1, 'complib': 'zlib', 'fletcher32': True }, # Enable compression and checksumming
	'schema': {
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
			'exposure_key': 'exp_id',
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
			'blobs': {
				'notfound_keys': {},
				'notfound_cols': {}
			},
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
			'blobs': { 'hdr': {} }
		},
		'chips': {
			'columns': [
				('chip_hdr',		'64i8',	'',		'XY??.hdr FITS headers, one per chip' )	# Note -- this will be a blob
			],
			'blobs': { 'chip_hdr': {} }
		}
	}
}

det_cat_def = \
{
	'filters': { 'complevel': 1, 'complib': 'zlib', 'fletcher32': True }, # Enable compression and checksumming
	'schema': {
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
			'exposure_key': 'exp_id',
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
}

obj_cat_def = \
{
	'filters': { 'complevel': 1, 'complib': 'zlib', 'fletcher32': True }, # Enable compression and checksumming
	'schema': {
		#
		#	 LSD column name      Type    FITS column      Description
		#
		'astrometry': {
			'columns': [
				('obj_id',		'u8', '',		'Unique LSD ID of this object'),
				('ra',			'f8', 'ra_psf',		''),
				('dec',			'f8', 'dec_psf',	''),
				('cached',		'bool', '',		''),
			],
			'primary_key': 'obj_id',
			'spatial_keys': ('ra', 'dec'),
			"cached_flag": "cached"
		}
	}
}

o2d_cat_def = \
{
	'filters': { 'complevel': 1, 'complib': 'zlib', 'fletcher32': True }, # Enable compression and checksumming
	'schema': {
		'main': {
			'columns': [
				('o2d_id',		'u8', '', 'Unique ID of this row'),
				('obj_id',		'u8', '', 'Object ID'),
				('det_id',		'u8', '', 'Detection ID'),
				('dist',		'f4', '', 'Distance (in degrees)'),
				('ra',			'f8', 'ra_psf',		''),
				('dec',			'f8', 'dec_psf',	''),
			],
			'primary_key': 'o2d_id',
			'spatial_keys': ('ra', 'dec'),
		}
	},
	'aliases': {
		'_M1': 'obj_id',
		'_M2': 'det_id',
		'_DIST': 'dist'
	}
}

def gen_cat2fits(catdef):
	""" Returns a mapping from catalog columns to
	    FITS columns:
		
		cat2fits[catcol] -> fitscol
	"""
	cat2fits = {}

	for tname, schema in catdef['schema'].iteritems():
		cat2fits.update( dict(( (colname, fitscol) for colname, _, fitscol, _ in schema['columns'] if fitscol != '' )) )

	return cat2fits

def gen_cat2type(catdef):
	""" Returns a mapping from column name to dtype
	"""
	
	cat2type = {}
	for tname, schema in catdef['schema'].iteritems():
		cat2type.update( dict(( (colname, dtype) for colname, dtype, _, _ in schema['columns'] )) )
		
	return cat2type

def import_from_smf(db, det_catname, exp_catname, smf_files, create=False):
	""" Import a PS1 catalog from DVO

	    Note: Assumes underlying shared storage for all catalog
	          cells (i.e., any worker is able to write to any cell).
	"""
	if create:
		# Create the new database
		det_cat  = db.create_catalog(det_catname, det_cat_def)
		exp_cat  = db.create_catalog(exp_catname, exp_cat_def)

		# Set up a one-to-X join relationship between the two catalogs (join det_cat:exp_id->exp_cat:exp_id)
		db.define_default_join(det_catname, exp_catname,
			type = 'indirect',
			m1   = (det_catname, "det_id"),
			m2   = (det_catname, "exp_id")
			)
	else:
		det_cat = db.catalog(det_catname)
		exp_cat = db.catalog(exp_catname)

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
	del pool

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
	fn = '.'.join(file.split('/')[-1].split('.')[:-1])
	exp_cols['smf_fn'] = np.empty(1, dtype='a40')
	exp_cols['smf_fn'][:] = fn

	# Store the primary and all chip headers in external linked files
	uri = 'lsd:%s:hdr:%s/primary.txt' % (exp_cat.name, fn)
	with exp_cat.open_uri(uri, mode='w', clobber=False) as f:
		f.write(str(imhdr) + '\n')
		exp_cols['hdr']	= np.array([uri], dtype=object)

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
			
			uri = 'lsd:%s:hdr:%s/%s.txt' % (exp_cat.name, fn, chip_xy)
			with exp_cat.open_uri(uri, mode='w', clobber=False) as f:
				f.write(str(hdr) + '\n')
				exp_cols['chip_hdr'][0][chip_id] = uri

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

	yield (file, len(ids), len(ids))

#########

# image cache creator kernels
# 1) stream through entire detection catalog, recording exp_ids of images
#    from other cells. Key these by their cell_id, our cell_id, and return.
# 2) stream through the image catalog, collecting rows of images that are
#    referenced out of their cells. Return the rows keyed by destination
#    cells.
# 3) store the copies into the cache of each cell.

def make_image_cache(db, det_cat_path, exp_cat_path):
	# Entry point for creation of image cache

	query = "_EXP FROM '%s'" % (det_cat_path)
	for _ in db.query(query).execute([
					_exp_id_gather,
					(_exp_id_load,    db, exp_cat_path),
					(_exp_store_rows, db, exp_cat_path)
				      ],
				      include_cached=True):
		pass;

def _exp_id_gather(qresult):
	# Fetch all exp_ids referenced from this cell
	for rows in qresult:
		cell_id = rows.cell_id

		exp_id     = np.unique(rows['_EXP'])
		exp_cells  = qresult.pix.cell_for_id(exp_id)

		# keep only those that aren't local
		outside    = exp_cells != cell_id
		exp_id     =    exp_id[outside]
		exp_cells  = exp_cells[outside]

		# return (exp_cell, (det_cell, exposures)) tuples
		for exp_cell in set(exp_cells):
			# List of exposure IDs from exp_cell, that need
			# to be cached in det_cell
			exps = exp_id[exp_cells == exp_cell]

			yield (exp_cell, (cell_id, exps))

def _exp_id_load(kv, db, exp_cat_path):
	#-- This kernel is called once for each exp_cat cell referenced from det_cat
	#
	# kv = (exp_cell, detexps) with
	#    detexps = [ (det_cell1, exps1), (det_cell, exps2), ... ]
	# where exps are the list of exp_ids to load, and det_cell
	# are the cells that should get a copy of each of those
	# exposure IDs.
	#
	exp_cell, detexps = kv

	# Load the entire cell
	rows = db.query("_ID, * FROM '%s'" % exp_cat_path).fetch_cell(exp_cell)
	if rows is None:
		return
	exp_id = rows['_ID']
	rows.drop_column('_ID')

	# Dispatch the right rows to their destination cells
	for (cache_cell, exps) in detexps:
		in_ = np.in1d(exp_id, exps, assume_unique=True)
		ret = rows[in_]

		# Yield row-by-row, to help the content-deduplication
		# mechanisms of the framework to do a better job
		for i in xrange(len(ret)):
			yield cache_cell, ret[i:i+1]
		#yield cache_cell, ret

	#print exp_cell, " -> ", [ (cache_cell, len(rows[exp_cat.primary_table]['rows'])) for (det_cellt, rows) in ret ]

def _store_rows(cell_id, exp_cat, rowlist):
	# Helper for _exp_store_rows
	if len(rowlist) == 0:
		return 0

	n = 0
	for rows in rowlist: n += len(rows)

	arows = rowlist.pop(0)
	at = len(arows)
	arows.resize(n)

	for rows in rowlist:
		arows[at:at+len(rows)] = rows
		at += len(rows)
	assert at == n, '%d %d' % (at, n)

	exp_cat.append(arows, cell_id=cell_id, group='cached')
	return len(arows)

def _exp_store_rows(kv, db, exp_cat_path):
	# Cache all rows to be cached in this cell
	cell_id, rowblocks = kv

	# Delete existing neighbors
	exp_cat = db.catalog(exp_cat_path)
	exp_cat.drop_row_group(cell_id, 'cached')

	# Accumulate a few blocks, then add them to cache (appending
	# bit-by-bit is expensive because of the all the fsyncs() invloved)
	ncached = 0
	rowlist = []
	chk = 0
	for rows in rowblocks:
		chk += len(rows)
		rowlist.append(rows)

		if len(rowlist) >= 50:
			ncached += _store_rows(cell_id, exp_cat, rowlist)
			rowlist = [ ]
	ncached += _store_rows(cell_id, exp_cat, rowlist)
	assert chk == ncached

	# Return the number of new rows cached into this cell
	yield cell_id, ncached

###############
# object catalog creator

def make_object_catalog(db, obj_catdir, det_catdir, radius=1./3600., create=True):
	# Entry point for creation of image cache

	# For debugging -- a simple check to see if matching works is to rerun
	# the match across a just matched catalog. In this case, we expect
	# all detections to be matched to existing objects, and no new ones added.
	_rematching = int(os.getenv('REMATCHING', False))
	if _rematching:
		create = False

	det_cat = db.catalog(det_catdir)

	o2d_catdir = '_%s_to_%s' % (obj_catdir, det_catdir)

	if create:
		# Create the new object database
		obj_cat = db.create_catalog(obj_catdir, obj_cat_def)
		o2d_cat = db.create_catalog(o2d_catdir, o2d_cat_def)

		# Set up a one-to-X join relationship between the two catalogs (join obj_cat:obj_id->det_cat:det_id)
		db.define_default_join(obj_catdir, det_catdir,
			type = 'indirect',
			m1   = (o2d_catdir, "obj_id"),
			m2   = (o2d_catdir, "det_id")
			)
		# Set up a join between the indirection table and detections catalog (det_cat:det_id->o2d_cat:o2d_id)
		db.define_default_join(det_catdir, o2d_catdir,
			type = 'indirect',
			m1   = (o2d_catdir, "det_id"),
			m2   = (o2d_catdir, "o2d_id")
			)
		# Set up a join between the indirection table and detections catalog (det_cat:det_id->o2d_cat:o2d_id)
		db.define_default_join(obj_catdir, o2d_catdir,
			type = 'indirect',
			m1   = (o2d_catdir, "obj_id"),
			m2   = (o2d_catdir, "o2d_id")
			)
	else:
		obj_cat = db.catalog(obj_catdir)
		o2d_cat = db.catalog(o2d_catdir)

	# Fetch all non-empty cells with detections. Group them by the same spatial
	# cell ID. This will be the list over which the first kernel will map.
	det_cells = det_cat.get_cells()
	det_cells_grouped = det_cat.pix.group_cells_by_spatial(det_cells).items()

	t0 = time.time()
	pool = pool2.Pool()
	ntot = 0
	ntotobj = 0
	at = 0
	for (nexp, nobj, ndet, nnew, nmatch, ndetnc) in pool.map_reduce_chain(det_cells_grouped,
					      [
						(_obj_det_match, db, obj_catdir, det_catdir, o2d_catdir, radius, _rematching),
					      ],
					      progress_callback=pool2.progress_pass):
		at += 1
		if nexp is None:
			continue

		t1 = time.time()
		time_pass = (t1 - t0) / 60
		time_tot = time_pass / at * len(det_cells)

		ntot += nmatch
		ntotobj += nnew
		nobjnew = nobj + nnew
		pctnew   = 100. * nnew / nobjnew  if nobjnew else 0.
		pctmatch = 100. * nmatch / ndetnc if ndetnc else 0.
		print "  match %7d det to %7d obj (%3d exps): %7d new (%6.2f%%), %7d matched (%6.2f%%)  [%.0f/%.0f min.]" % (ndet, nobj, nexp, nnew, pctnew, nmatch, pctmatch, time_pass, time_tot)

	print >> sys.stderr, "Building neighbor cache for static sky: ",
	db.build_neighbor_cache(obj_catdir)
	print >> sys.stderr, "Building neighbor cache for indirection table: ",
	db.build_neighbor_cache(o2d_catdir)

	# Compute summary stats for object catalog
	print >> sys.stderr, "Computing summary statistics for static sky: ",
	db.compute_summary_stats(obj_catdir)
	print >> sys.stderr, "Computing summary statistics for indirection table: ",
	db.compute_summary_stats(o2d_catdir)

	print "Matched a total of %d sources." % (ntot)
	print "Total of %d objects added." % (ntotobj)
	print "Rows in the object catalog: %d." % (obj_cat.nrows())
	print "Rows in the detection catalog: %d." % (det_cat.nrows())
	assert not _rematching or det_cat.nrows() == ntot

def reserve_space(arr, minsize):
	l = len(arr)
	if l == 0: l = l + 1
	while l < minsize:
		l = 2*l
	arr.resize(l, refcheck=False)

# Single-pass detections->objects mapper.
def _obj_det_match(cells, db, obj_catdir, det_catdir, o2d_catdir, radius, _rematching=False):
	"""
	This kernel assumes:
	   a) det_cat and obj_cat have equal partitioning (equally
	      sized/enumerated spatial cells)
	   b) both det_cat and obj_cat have up-to-date neighbor caches
	   c) temporal det_cat cells within this spatial cell are stored
	      local to this process (relevant for shared-nothing setups)
	   d) exposures don't stretch across temporal cells

	Algorithm:
	   - fetch all existing static sky objects, including the cached ones (*)
	   - project them to tangent plane around the center of the cell
	     (we assume the cell is small enough for the distortions not to matter)
	   - construct a kD tree in (x, y) tangent space
	   - for each temporal cell, in sorted order (++):
	   	1.) Fetch the detections, including the cached ones (+)
	   	2.) Project to tangent plane

	   	3.) for each exposure, in sorted order (++):
		    a.) Match agains the kD tree of objects
		    b.) Add those that didn't match to the list of objects 

		4.) For newly added objects: store to disk only those that
		    fall within this cell (the others will be matched and
		    stored in their parent cells)

		5.) For matched detections: Drop detections matched to cached
		    objects (these will be matched and stored in the objects'
		    parent cell). Store the rest.


	   (+) It is allowed (and necessary to allow) for a cached detection
		    to be matched against an object within our cell.  This
		    correctly matches cases when the object is right inside
		    the cell boundary, but the detection is just to the
		    outside.

	   (++) Having cells and detections sorted ensures that objects in overlapping
	        (cached) regions are seen by kernels in different cells in the same
	        order, thus resulting in the same matches. Note: this may fail in
	        extremely crowded region, but as of now it's not clear how big of
	        a problem (if any!) will this pose.

	   (*) Cached objects must be loaded and matched against to guard against
	       the case where an object is just outside the edge, while a detection
	       is just inside. If the cached object was not loaded, the detection
	       would not match and be proclamed to be a new object. However, in the
	       cached object's parent cell, the detection would match to the object
	       and be stored there as well.
	       
	       The algorithm above ensures that such a detection will matched to
	       the cached object in this cell (and be dropped in step 5), preventing
	       it from being promoted into a new object.

	   TODO: The above algorithm ensures no detection is assigned to more than
	   	one object. It also ensures that each detection links to an object.
	   	Implement a consistency check to verify that.
	"""

	from scikits.ann import kdtree

	# Input is a tuple of obj_cell, and det_cells falling under that obj_cell
	obj_cell, det_cells = cells
	det_cells.sort()
	assert len(det_cells)

	# Fetch the frequently used bits
	obj_cat = db.catalog(obj_catdir)
	det_cat = db.catalog(det_catdir)
	o2d_cat = db.catalog(o2d_catdir)
	pix = obj_cat.pix

	# locate cell center (for gnomonic projection)
	(bounds, tbounds)  = pix.cell_bounds(obj_cell)
	(clon, clat) = bhpix.deproj_bhealpix(*bounds.center())

	# fetch existing static sky, convert to gnomonic
	objs  = db.query('_ID, _LON, _LAT FROM %s' % obj_catdir).fetch_cell(obj_cell, include_cached=True)
	xyobj = np.column_stack(gnomonic(objs['_LON'], objs['_LAT'], clon, clat))
	nobj  = len(objs)	# Total number of static sky objects
	tree  = None

	# for sanity checks/debugging (see below)
	expseen = set()

	## TODO: Debugging, remove when happy
	assert (np.unique(sorted(det_cells)) == sorted(det_cells)).all()
	##print "Det cells: ", det_cells

	# Loop, xmatch, and store
	det_query = db.query('_ID, _LON, _LAT, _EXP, _CACHED FROM %s' % det_catdir)
	for det_cell in sorted(det_cells):
		# fetch detections in this cell, convert to gnomonic coordinates
		detections = det_query.fetch_cell(det_cell, include_cached=True)
		_, ra2, dec2, exposures, cached = detections.as_columns()
		detections.add_column('xy', np.column_stack(gnomonic(ra2, dec2, clon, clat)))

		# if there are no preexisting static sky objects, and all detections in this cell are cached,
		# there's no way we'll get a match that will be kept in the end. Just continue to the
		# next one if this is the case.
		cachedonly = len(objs) == 0 and cached.all()
		if cachedonly:
#			print "Skipping cached-only", len(cached)
			yield (None, None, None, None, None, None) # Yield just to have the progress counter properly incremented
			continue;

		# prep join table
		join  = table.Table(dtype=o2d_cat.dtype_for(['_ID', '_M1', '_M2', '_DIST', '_LON', '_LAT']))
		njoin = 0;
		nobj0 = nobj;

		##print "Cell", det_cell, " - Unique exposures: ", set(exposures)

		# Process detections exposure-by-exposure, as detections from
		# different exposures within a same temporal cell are allowed
		# to belong to the same object
		uexposures = set(exposures)
		for exposure in sorted(uexposures):
			# Sanity check: a consistent catalog cannot have two
			# exposures stretching over more than one cell
			assert exposure not in expseen
			expseen.add(exposure);

			# Extract objects belonging to this exposure only
			detections2 = detections[exposures == exposure]
			id2, ra2, dec2, _, _, xydet = detections2.as_columns()
			ndet = len(xydet)

			if len(xyobj) != 0:
				# Construct kD-tree and find the object nearest to each
				# detection from this cell
				if tree is None or nobj_old != len(xyobj):
					del tree
					nobj_old = len(xyobj)
					tree = kdtree(xyobj)
				match_idx, match_d2 = tree.knn(xydet, 1)
				match_idx = match_idx[:,0]		# First neighbor only

				# Compute accurate distances, and select detections not matched to existing objects
				dist       = gc_dist(objs['_LON'][match_idx], objs['_LAT'][match_idx], ra2, dec2)
				unmatched  = dist >= radius
			else:
				# All detections will become new objects (and therefore, dist=0)
				dist       = np.zeros(ndet, dtype='f4')
				unmatched  = np.ones(ndet, dtype=bool)
				match_idx  = np.empty(ndet, dtype='i4')

#			x, y, t = pix._xyt_from_cell_id(det_cell)
#			print "det_cell %s, MJD %s, Exposure %s  ==  %d detections, %d objects, %d matched, %d unmatched" % (det_cell, t, exposure, len(detections2), nobj, len(unmatched)-unmatched.sum(), unmatched.sum())

			# Promote unmatched detections to new objects
			_, newra, newdec, _, _, newxy = detections2[unmatched].as_columns()
			nunmatched = unmatched.sum()
			reserve_space(objs, nobj+nunmatched)
			objs['_LON'][nobj:nobj+nunmatched] = newra
			objs['_LAT'][nobj:nobj+nunmatched] = newdec
			dist[unmatched]                    = 0.
			match_idx[unmatched] = np.arange(nobj, nobj+nunmatched, dtype='i4')	# Set the indices of unmatched detections to newly created objects

			# Join objects to their detections
			reserve_space(join, njoin+ndet)
			join['_M1'][njoin:njoin+ndet]   = match_idx
			join['_M2'][njoin:njoin+ndet]   =       id2
			join['_DIST'][njoin:njoin+ndet] =      dist
			# TODO: For debugging; remove when happy
			join['_LON'][njoin:njoin+ndet]  =       ra2
			join['_LAT'][njoin:njoin+ndet]  =      dec2
			njoin += ndet

			# Prep for next loop
			nobj  += nunmatched
			xyobj  = np.append(xyobj, newxy, axis=0)

			# TODO: Debugging: Final consistency check (remove when happy with the code)
			dist = gc_dist( objs['_LON'][  join['_M1'][njoin-ndet:njoin]  ],
					objs['_LAT'][  join['_M1'][njoin-ndet:njoin]  ], ra2, dec2)
			assert (dist < radius).all()

		# Truncate output tables to their actual number of elements
		objs = objs[0:nobj]
		join = join[0:njoin]
		assert len(objs) >= nobj0

		# Find the objects that fall outside of cell boundaries. These will
		# be processed and stored by their parent cells. Also leave out the objects
		# that are already stored in the database
		(x, y) = bhpix.proj_bhealpix(objs['_LON'], objs['_LAT'])
		in_    = bounds.isInsideV(x, y)
		innew  = in_.copy();
		innew[:nobj0] = False											# New objects in cell selector

		ids = objs['_ID']
		nobjadded = innew.sum()
		if nobjadded:
			# Append the new objects to the object catalog, obtaining their IDs.
			assert not _rematching, 'cell_id=%s, nnew=%s\n%s' % (det_cell, nobjadded, objs[innew])
			ids[innew] = obj_cat.append(objs[('_LON', '_LAT')][innew])

		# Set the indices of objects not in this cell to zero (== a value
		# no valid object in the database can have). Therefore, all
		# out-of-bounds links will have _M1 == 0 (#1), and will be removed
		# by the np1d call (#2)
		ids[~in_] = 0

		# 1) Change the relative index to true obj_id in the join table
		join['_M1'] = ids[join['_M1']]

		# 2) Keep only the joins to objects inside the cell
		join = join[ np.in1d(join['_M1'], ids[in_]) ]

		# Append to the join table, in *dec_cell* of obj_cat (!important!)
		if len(join) != 0:
			# compute the cell_id part of the join table's
			# IDs.While this is unimportant now (as we could
			# just set all of them equal to cell_id part of
			# cell_id), if we ever decide to change the
			# pixelation of the catalog later on, this will
			# allow us to correctly split up the join table as
			# well.
			_, _, t    = pix._xyt_from_cell_id(det_cell)	# This row points to a detection in the temporal cell ...
			x, y, _, _ = pix._xyti_from_id(join['_M1'])	# ... but at the spatial location given by the object catalog.
			join['_ID'][:] = pix._id_from_xyti(x, y, t, 0) # This will make the new IDs have zeros in the object part (so Catalog.append will autogen them)

			o2d_cat.append(join)

		assert not cachedonly or (nobjadded == 0 and len(join) == 0)

		# return: Number of exposures, number of objects before processing this cell, number of detections processed (incl. cached),
		#         number of newly added objects, number of detections xmatched, number of detection processed that weren't cached
		# Note: some of the xmatches may be to newly added objects (e.g., if there are two 
		#       overlapping exposures within a cell; first one will add new objects, second one will match agains them)
		yield (len(uexposures), nobj0, len(detections), nobjadded, len(join), (cached == False).sum())

if __name__ == '__main__':
	make_object_catalog('ps1_obj', 'ps1_det')
