#!/usr/bin/env python

import pyfits
import pool2
import time
import numpy as np
from pyslalib.slalib import sla_eqgal
import itertools as it
import bhpix
from utils import gnomonic, gc_dist
from colgroup import ColGroup
import colgroup
import sys, os
import logging
from interval import intervalset
import bounds
import locking

logger = logging.getLogger("lsd.smf")

# Table defs for exposure table
exp_table_def = \
{
	'fgroups': {
		'hdr': {
			'filter': ('bzip2', {'compresslevel': 9})
		}
	},
	'filters': { 'complevel': 5, 'complib': 'blosc', 'fletcher32': False }, # Enable compression
	'schema': {
		#
		#	 LSD column name      Type    FITS column      Description
		#
		'main': {
			'columns': [
				('exp_id',		'u8', 	'',		'LSD primary key for this exposure'   ),
				('filterid',		'a6',	'filterid',	'Filter used (instrument name)'       ),
				('survey',		'a4',   '',		'PS1 sub-survey (3pi, mdf, sas, ...)' ),
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

det_table_def = \
{
	'filters': { 'complevel': 5, 'complib': 'blosc', 'fletcher32': False }, # Enable compression
	'schema': {
		#
		#	 LSD column name      Type    FITS column      Description
		#
		'astrometry': {
			'columns': [
				('det_id',		'u8', '',		'Unique LSD ID of this detection'),
				('exp_id',		'u8', '',		'Exposure ID, joined to the image table'),
				('chip_id',		'u1', '',		'Index of the OTA where this object was detected (integer, 0-63)'),
				('survey',		'a4', '',		'PS1 sub-survey (3pi, mdf, sas, ...)' ),
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

obj_table_def = \
{
	'filters': { 'complevel': 5, 'complib': 'blosc', 'fletcher32': False }, # Enable compression
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

o2d_table_def = \
{
	'filters': { 'complevel': 5, 'complib': 'blosc', 'fletcher32': False }, # Enable compression
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

def gen_tab2fits(tabdef):
	""" Returns a mapping from table columns to
	    FITS columns:
		
		tab2fits[tabcol] -> fitscol
	"""
	tab2fits = {}

	for tname, schema in tabdef['schema'].iteritems():
		tab2fits.update( dict(( (colname, fitscol) for colname, _, fitscol, _ in schema['columns'] if fitscol != '' )) )

	return tab2fits

def gen_tab2type(tabdef):
	""" Returns a mapping from column name to dtype
	"""
	
	tab2type = {}
	for tname, schema in tabdef['schema'].iteritems():
		tab2type.update( dict(( (colname, dtype) for colname, dtype, _, _ in schema['columns'] )) )
		
	return tab2type

def store_smf_list(db, exp_tabname, new_exps):
	""" Store a human-readable list of loaded SMF files and their exp_ids,
	    for fast lookup by ps1-load and lsd-make-object-catalog.

	    Note: For forward-compatibility, readers of this file should assume it has an
	    unspecified number of columns. I.e., don't do (a, b) = line.split(), but
	    do (a, b) = line.split()[:2]
	"""
	uri = 'lsd:%s:cache:all_exposures.txt' % (exp_tabname)
	
	# Try to load from cache, query otherwise
	try:
		with db.open_uri(uri) as f:
			old_exps = np.loadtxt(f, dtype=[('_EXP', 'u8'),('smf_fn', 'a40')])
	except IOError:
		old_exps = db.query("select _EXP, smf_fn from %s" % exp_tabname).fetch(progress_callback=pool2.progress_pass)

	exps = colgroup.fromiter([old_exps, new_exps], blocks=True) if len(old_exps) else new_exps

	# Store to cache
	with db.open_uri(uri, mode='w') as f:
		for exp_id, exp_fn in exps:
			f.write("%d %s\n" % (exp_id, exp_fn))

def import_from_smf(db, det_tabname, exp_tabname, smf_files, survey, create=False):
	""" Import a PS1 table from DVO

	    Note: Assumes underlying shared storage for all table
	          cells (i.e., any worker is able to write to any cell).
	"""
	with locking.lock(db.path[0] + "/.__smf-import-lock.lock"):
		if not db.table_exists(det_tabname) and create:
			# Set up commit hooks
			exp_table_def['commit_hooks'] = [ ('Updating neighbors', 1, 'lsd.smf', 'make_image_cache', [det_tabname]) ]

			# Create new tables
			det_table  = db.create_table(det_tabname, det_table_def)
			exp_table  = db.create_table(exp_tabname, exp_table_def)

			# Set up a one-to-X join relationship between the two tables (join det_table:exp_id->exp_table:exp_id)
			db.define_default_join(det_tabname, exp_tabname,
				type = 'indirect',
				m1   = (det_tabname, "det_id"),
				m2   = (det_tabname, "exp_id"),
				_overwrite=create
				)
		else:
			det_table = db.table(det_tabname)
			exp_table = db.table(exp_tabname)

	det_c2f = gen_tab2fits(det_table_def)
	exp_c2f = gen_tab2fits(exp_table_def)

	t0 = time.time()
	at = 0; ntot = 0
	pool = pool2.Pool()
	smf_fns = []
	exp_ids = []
	for (file, exp_id, smf_fn, nloaded, nin) in pool.imap_unordered(smf_files, import_from_smf_aux, (det_table, exp_table, det_c2f, exp_c2f, survey), progress_callback=pool2.progress_pass):
		smf_fns.append(smf_fn)
		exp_ids.append(exp_id)
		at = at + 1
		ntot = ntot + nloaded
		t1 = time.time()
		time_pass = (t1 - t0) / 60
		time_tot = time_pass / at * len(smf_files)
		print >>sys.stderr, '  ===> Imported %s [%d/%d, %5.2f%%] +%-6d %9d (%.0f/%.0f min.)' % (file, at, len(smf_files), 100 * float(at) / len(smf_files), nloaded, ntot, time_pass, time_tot)
	del pool

	ret = colgroup.ColGroup()
	ret._EXP   = np.array(exp_ids, dtype=np.uint64)
	ret.smf_fn = np.array(smf_fns, dtype='a40')
	return ret

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

def fix_ps1_coord_bugs(cols, file, hduname):
	# FIXME: Work around PS1 bugs
	ra, dec = cols['ra'], cols['dec']

	if np.any(ra < 0):
		ra[ra < 0] = np.fmod(np.fmod(ra[ra < 0], 360) + 360, 360)

	if np.any(ra >= 360):
		ra[ra >= 360] = np.fmod(np.fmod(ra[ra >= 360], 360) + 360, 360)

	if np.any(np.abs(dec) > 90):
		logger.warning("Encountered %d instances of dec > +/- 90 in file %s, %s HDU. Truncating to +/-90." % (sum(np.abs(dec) > 90), file, hduname))
		dec[dec > 90] = 90
		dec[dec < -90] = -90

	cols['ra'], cols['dec'] = ra, dec

	# Remove any NaN rows
	if np.isnan(cols['ra'].sum()):
		logger.warning("Encountered %d instances of ra == NaN in file %s, %s HDU. Removing those rows." % (sum(np.isnan(ra)), file, hduname))
		keep = ~np.isnan(cols['ra'])
		for name in cols: cols[name] = cols[name][keep]

	if np.isnan(cols['dec'].sum()):
		logger.warning("Encountered %d instances of dec == NaN in file %s, %s HDU. Removing those rows." % (sum(np.isnan(dec)), file, hduname))
		keep = ~np.isnan(cols['dec'])
		for name in cols: cols[name] = cols[name][keep]

def import_from_smf_aux(file, det_table, exp_table, det_c2f, exp_c2f, survey):
	det_c2t = gen_tab2type(det_table_def)
	exp_c2t = gen_tab2type(exp_table_def)

	# Load the .smf file
	hdus = pyfits.open(file)

	# Do the exposure table entry first; we'll need the
	# image ID to fill in the detections table
	imhdr = hdus[0].header
	exp_cols, exp_nullcols = load_columns_from_header(imhdr, exp_c2f, exp_c2t)
	add_lb(exp_cols)
	fix_ps1_coord_bugs(exp_cols, file, 'primary')

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
	uri = 'lsd:%s:hdr:%s/primary.txt' % (exp_table.name, fn)
	with exp_table.open_uri(uri, mode='w') as f:
		f.write(str(imhdr) + '\n')
		exp_cols['hdr']	= np.array([uri], dtype=object)

	exp_cols['chip_hdr']	= np.empty(1, dtype='64O')
	exp_cols['chip_hdr'][:] = None
	
	exp_cols['survey']      = np.empty(1, dtype='a4')
	exp_cols['survey'][:]   = survey
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
			
			uri = 'lsd:%s:hdr:%s/%s.txt' % (exp_table.name, fn, chip_xy)
			with exp_table.open_uri(uri, mode='w') as f:
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

		fix_ps1_coord_bugs(det_cols, file, chip_xy + '.psf')

		# Add computed columns
		add_lb(det_cols)
		det_cols['filterid']    = np.empty(len(dat), dtype='a6')
		det_cols['survey']      = np.empty(len(dat), dtype='a4')
		det_cols['mjd_obs']     = np.empty(len(dat), dtype='f8')
		det_cols['exp_id']      = np.empty(len(dat), dtype='u8')
		det_cols['chip_id']     = np.empty(len(dat), dtype='u1')
		det_cols['filterid'][:] = filterid
		det_cols['survey'][:]   = survey
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

	(exp_id,) = exp_table.append(exp_cols)
	det_cols_all['exp_id'][:] = exp_id

	ids = det_table.append(det_cols_all)

	yield (file, exp_id, fn, len(ids), len(ids))

#########

# image cache creator kernels
# 1) stream through entire detection table, recording exp_ids of images
#    from other cells. Key these by their cell_id, our cell_id, and return.
# 2) stream through the image table, collecting rows of images that are
#    referenced out of their cells. Return the rows keyed by destination
#    cells.
# 3) store the copies into the cache of each cell.

def make_image_cache(db, det_tabname, exp_tabname, snapid):
	# Entry point for creation of image cache

	# Only get the cells that were modified in snapshot 'snapid'
	cells = db.table(det_tabname).get_cells_in_snapshot(snapid)
	if len(cells) == 0:
		print >>sys.stderr, "Already up to date."
		return

	query = "_EXP FROM '%s'" % (det_tabname)
	for _ in db.query(query).execute([
					_exp_id_gather,
					(_exp_id_load,    db, exp_tabname),
					(_exp_store_rows, db, exp_tabname)
				      ],
				      cells=cells,
				      include_cached=True):
		pass;

def commit_hook__make_image_cache(db, table, det_tabname):
	# Commit hook
	make_image_cache(db, det_tabname, table.name, db.snapid)
	print >> sys.stderr, "[%s] Updating tablet catalog:" % (table.name),
	table.build_tablet_tree_cache()

def _exp_id_gather(qresult):
	# Fetch all exp_ids referenced from this cell
	for rows in qresult:
		cell_id = rows.info.cell_id

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

def _exp_id_load(kv, db, exp_tabname):
	#-- This kernel is called once for each exp_tab cell referenced from det_table
	#
	# kv = (exp_cell, detexps) with
	#    detexps = [ (det_cell1, exps1), (det_cell, exps2), ... ]
	# where exps are the list of exp_ids to load, and det_cell
	# are the cells that should get a copy of each of those
	# exposure IDs.
	#
	exp_cell, detexps = kv

	# Load the entire cell
	rows = db.query("_ID, * FROM '%s'" % exp_tabname).fetch_cell(exp_cell)
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

	#print exp_cell, " -> ", [ (cache_cell, len(rows[exp_tab.primary_table]['rows'])) for (det_cellt, rows) in ret ]

def _store_rows(cell_id, exp_table, rowlist):
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

	exp_table.append(arows, cell_id=cell_id, group='cached')
	return len(arows)

def _exp_store_rows(kv, db, exp_tabname):
	# Cache all rows to be cached in this cell
	cell_id, rowblocks = kv

	# Delete existing neighbors
	exp_table = db.table(exp_tabname)
	exp_table.drop_row_group(cell_id, 'cached')

	# Accumulate a few blocks, then add them to cache (appending
	# bit-by-bit is expensive because of the all the fsyncs() invloved)
	ncached = 0
	rowlist = []
	chk = 0
	for rows in rowblocks:
		chk += len(rows)
		rowlist.append(rows)

		if len(rowlist) >= 50:
			ncached += _store_rows(cell_id, exp_table, rowlist)
			rowlist = [ ]
	ncached += _store_rows(cell_id, exp_table, rowlist)
	assert chk == ncached

	# Return the number of new rows cached into this cell
	yield cell_id, ncached

###############
# object table creator

def _unique_mapper(qresult):
	for rows in qresult:
		yield np.unique(rows.column(0))

def get_unique(db, query, cache_uri=None):
	# Get all unique exposures returned by the query
	#
	# Try to fetch the result from the cache, otherwise
	# execite the query (for backwards compatibility)
	try:
		with db.open_uri(cache_uri) as f:
			vals = set( int(s.strip().split()[0]) for s in f.xreadlines() )
	except IOError:
		vals = set()
		for exps in db.query(query).execute([_unique_mapper]):
			for exp in exps:
				vals.add(exp)

	return vals

def get_new_exposures(db, obj_tabname, det_tabname, exp_tabname):
	""" Get all exposures mentioned in the detections table, and not
	    mentioned in the object table.
	"""
	all_exps = get_unique(db, "_ID from %s" % (exp_tabname), "lsd:%s:cache:all_exposures.txt" % exp_tabname)
	obj_exps = get_unique(db, "_EXP from %s, %s" % (obj_tabname, det_tabname), "lsd:%s:cache:all_matched_exposures.txt" % obj_tabname)

	return all_exps - obj_exps, obj_exps

def get_cells_with_dets_from_exps(db, explist, exp_tabname, det_tabname, fovradius=None):
	""" Get all cells that may contain detections from within exposures
	    listed in explist. If given, use fovradius (in degrees) as the
	    radius of the field of view of each exposure, to limit the number
	    of cells that are considered. If fovradius is None, assume the field
	    of view of each exposure may extend to the entire sky.
	"""
	explist = frozenset(explist)

	bb = {}
	ti = intervalset()
	for t, exp, lon, lat in db.query("_TIME, _EXP, _LON, _LAT from %s" % exp_tabname).iterate():
		if exp not in explist:
			continue

		if fovradius is not None:
			tt = np.floor(t)
			fov = bounds.beam(lon, lat, fovradius)
			try:
				bb[tt] |= fov
			except KeyError:
				bb[tt] = fov
		else:
			ti.add((t-0.2, t+0.2))

	if fovradius is None:
		b = [(bounds.ALLSKY, ti)]
	else:
		b = []
		for t0, fov in bb.iteritems():
			b += [ (fov, intervalset((t0, t0+1.))) ]

	# Fetch all detection table cells that are within this interval
	det_cells = db.table(det_tabname).get_cells(bounds=b)

	return det_cells

def create_object_table(db, obj_tabname, det_tabname):
	""" Create a new object table, linkage table, and the joins required
	    to join it to detections table.
	    
	    Skip table creation if tables already exists, and overwrite joins
	    if they already exist.
	"""
	# Create a new object table
	o2d_tabname = '_%s_to_%s' % (obj_tabname, det_tabname)
	obj_table = db.create_table(obj_tabname, obj_table_def, ignore_if_exists=True)
	o2d_table = db.create_table(o2d_tabname, o2d_table_def, ignore_if_exists=True)

	# Set up a one-to-X join relationship between the two tables (join obj_table:obj_id->det_table:det_id)
	db.define_default_join(obj_tabname, det_tabname,
		type = 'indirect',
		m1   = (o2d_tabname, "obj_id"),
		m2   = (o2d_tabname, "det_id"),
		_overwrite = True
			)

	# Set up a join between the indirection table and detections table (det_table:det_id->o2d_table:o2d_id)
	db.define_default_join(det_tabname, o2d_tabname,
		type = 'indirect',
		m1   = (o2d_tabname, "det_id"),
		m2   = (o2d_tabname, "o2d_id"),
		_overwrite = True
		)

	# Set up a join between the indirection table and detections table (det_table:det_id->o2d_table:o2d_id)
	db.define_default_join(obj_tabname, o2d_tabname,
		type = 'indirect',
		m1   = (o2d_tabname, "obj_id"),
		m2   = (o2d_tabname, "o2d_id"),
		_overwrite = True
		)

def make_object_catalog(db, obj_tabname, det_tabname, exp_tabname, radius=1./3600., explist=None, oldexps=None, fovradius=None):
	""" Create the object catalog
	"""

	# For debugging -- a simple check to see if matching works is to rerun
	# the match across a just matched table. In this case, we expect
	# all detections to be matched to existing objects, and no new ones added.
	_rematching = int(os.getenv('REMATCHING', False))

	o2d_tabname = '_%s_to_%s' % (obj_tabname, det_tabname)
	det_table = db.table(det_tabname)
	obj_table = db.table(obj_tabname)
	o2d_table = db.table(o2d_tabname)

	# Fetch all non-empty cells with detections. Group them by the same spatial
	# cell ID. This will be the list over which the first kernel will map.
	if explist is None:
		det_cells = det_table.get_cells()
	else:
		# Fetch only those cells that can contain data from the given exposure list
		print >> sys.stderr, "Enumerating cells with new detections: ",
		det_cells = get_cells_with_dets_from_exps(db, explist, exp_tabname, det_tabname, fovradius)
	print "%d cells to process." % (len(det_cells))
	det_cells_grouped = det_table.pix.group_cells_by_spatial(det_cells).items()

	t0 = time.time()
	pool = pool2.Pool()
	ntot = 0
	ntotobj = 0
	at = 0
	for (nexp, nobj, ndet, nnew, nmatch, ndetnc) in pool.map_reduce_chain(det_cells_grouped,
				      [
					(_obj_det_match, db, obj_tabname, det_tabname, o2d_tabname, radius, explist, _rematching),
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

	# Save the list of exposures that has been matched to the object table
	if oldexps is not None and len(explist):
		allexps = set(oldexps) | set(explist)

		uri = "lsd:%s:cache:all_matched_exposures.txt" % obj_tabname
		with db.open_uri(uri, mode='w') as f:
			for exp in allexps:
				f.write("%d\n" % exp)

	print "Matched a total of %d sources." % (ntot)
	print "Total of %d objects added." % (ntotobj)

def reserve_space(arr, minsize):
	l = len(arr)
	if l == 0: l = l + 1
	while l < minsize:
		l = 2*l
	arr.resize(l, refcheck=False)

# Single-pass detections->objects mapper.
def _obj_det_match(cells, db, obj_tabname, det_tabname, o2d_tabname, radius, explist=None, _rematching=False):
	"""
	This kernel assumes:
	   a) det_table and obj_table have equal partitioning (equally
	      sized/enumerated spatial cells)
	   b) both det_table and obj_table have up-to-date neighbor caches
	   c) temporal det_table cells within this spatial cell are stored
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
	obj_table = db.table(obj_tabname)
	det_table = db.table(det_tabname)
	o2d_table = db.table(o2d_tabname)
	pix = obj_table.pix

	# locate cell center (for gnomonic projection)
	(bounds, tbounds)  = pix.cell_bounds(obj_cell)
	(clon, clat) = bhpix.deproj_bhealpix(*bounds.center())

	# fetch existing static sky, convert to gnomonic
	objs  = db.query('_ID, _LON, _LAT FROM %s' % obj_tabname).fetch_cell(obj_cell, include_cached=True)
	xyobj = np.column_stack(gnomonic(objs['_LON'], objs['_LAT'], clon, clat))
	nobj  = len(objs)	# Total number of static sky objects
	tree  = None
	nobj_old = 0

	# for sanity checks/debugging (see below)
	expseen = set()

	## TODO: Debugging, remove when happy
	assert (np.unique(sorted(det_cells)) == sorted(det_cells)).all()
	##print "Det cells: ", det_cells

	# Loop, xmatch, and store
	det_query = db.query('_ID, _LON, _LAT, _EXP, _CACHED FROM %s' % det_tabname)
	for det_cell in sorted(det_cells):
		# fetch detections in this cell, convert to gnomonic coordinates
		# keep only detections with _EXP in explist, unless explist is None
		detections = det_query.fetch_cell(det_cell, include_cached=True)
		if explist is not None:
			keep = np.in1d(detections._EXP, explist)
			if not np.all(keep):
				detections = detections[keep]
			if len(detections) == 0:
				yield (None, None, None, None, None, None) # Yield just to have the progress counter properly incremented
				continue
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
		join  = ColGroup(dtype=o2d_table.dtype_for(['_ID', '_M1', '_M2', '_DIST', '_LON', '_LAT']))
		njoin = 0;
		nobj0 = nobj;

		##print "Cell", det_cell, " - Unique exposures: ", set(exposures)

		# Process detections exposure-by-exposure, as detections from
		# different exposures within a same temporal cell are allowed
		# to belong to the same object
		uexposures = set(exposures)
		for exposure in sorted(uexposures):
			# Sanity check: a consistent table cannot have two
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

				####
				#if np.uint64(13828114484734072082) in id2:
				#	np.savetxt('bla.%d.static=%d.txt' % (det_cell, pix.static_cell_for_cell(det_cell)), objs.as_ndarray(), fmt='%s')

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
			# Append the new objects to the object table, obtaining their IDs.
			assert not _rematching, 'cell_id=%s, nnew=%s\n%s' % (det_cell, nobjadded, objs[innew])
			ids[innew] = obj_table.append(objs[('_LON', '_LAT')][innew])

		# Set the indices of objects not in this cell to zero (== a value
		# no valid object in the database can have). Therefore, all
		# out-of-bounds links will have _M1 == 0 (#1), and will be removed
		# by the np1d call (#2)
		ids[~in_] = 0

		# 1) Change the relative index to true obj_id in the join table
		join['_M1'] = ids[join['_M1']]

		# 2) Keep only the joins to objects inside the cell
		join = join[ np.in1d(join['_M1'], ids[in_]) ]

		# Append to the join table, in *dec_cell* of obj_table (!important!)
		if len(join) != 0:
			# compute the cell_id part of the join table's
			# IDs.While this is unimportant now (as we could
			# just set all of them equal to cell_id part of
			# cell_id), if we ever decide to change the
			# pixelation of the table later on, this will
			# allow us to correctly split up the join table as
			# well.
			#_, _, t    = pix._xyt_from_cell_id(det_cell)	# This row points to a detection in the temporal cell ...
			#x, y, _, _ = pix._xyti_from_id(join['_M1'])	# ... but at the spatial location given by the object table.
			#join['_ID'][:] = pix._id_from_xyti(x, y, t, 0) # This will make the new IDs have zeros in the object part (so Table.append will autogen them)
			join['_ID'][:] = det_cell

			o2d_table.append(join)

		assert not cachedonly or (nobjadded == 0 and len(join) == 0)

		# return: Number of exposures, number of objects before processing this cell, number of detections processed (incl. cached),
		#         number of newly added objects, number of detections xmatched, number of detection processed that weren't cached
		# Note: some of the xmatches may be to newly added objects (e.g., if there are two 
		#       overlapping exposures within a cell; first one will add new objects, second one will match agains them)
		yield (len(uexposures), nobj0, len(detections), nobjadded, len(join), (cached == False).sum())

###############################
# Object table sanity checker

def _sanity_check_object_table_mapper(qresult, radius, explist):
	all_objs = np.empty(0, dtype=np.uint64)
	havedet_objs = np.empty(0, dtype=np.uint64)

	for rows in qresult:
		# Prep work for checking #3, at the end.
		# Collect all objects
		objs = np.unique(rows.obj_id)					# Get unique IDs
		objs = objs[~np.in1d(objs, all_objs, assume_unique=True)]	# Keep only those never before seen
		all_objs = np.append(all_objs, objs)					# Append to a list of all objects

		# Remove outer-joined rows
		rows = rows[~rows.dnull]

		# Prep work for checking #3, at the end.
		# Collect objects with linked detections
		objs = np.unique(rows.obj_id)
		objs = objs[~np.in1d(objs, havedet_objs, assume_unique=True)]	# Keep only those never before seen
		havedet_objs = np.append(havedet_objs, objs)			# Append to a list of having a detection

		# Remove all detections that did not originate from explist
		if explist is not None:
			rows = rows[np.in1d(rows.exp_id, explist)]

		# Check radius (cond #4)
		dist = gc_dist(rows.olon, rows.olat, rows.dlon, rows.dlat)
		assert np.all(dist < radius)

		# Regroup by det_id's cell and yield to reducer.
		# We do this because some det_id's from other cells may be linked
		# to an object from this cell
		cells = qresult.pix.cell_for_id(rows.det_id)
		for cell_id in np.unique(cells):
			yield (cell_id, rows[("det_id", "obj_id")][cells == cell_id])

	# Check that each object has exactly one detection (cond #3)
	all_objs.sort()
	havedet_objs.sort()
	assert np.all(all_objs == havedet_objs)

def _sanity_check_object_table_reducer(kv, db, det_tabname, explist):
	cell_id, rowlist = kv
	rows = colgroup.fromiter(rowlist, blocks=True)
	rows.sort(["det_id"])

	# Verify that each detection appears only once (cond #1)
	if not np.all(np.diff(rows.det_id) != 0):
		#a1 = np.arange(len(rows))[np.diff(rows.det_id) == 0]
		with open("cell.%s.txt" % cell_id, 'w') as fp:
			for row in rows:
				fp.write("%s\t%s\n" % (row['det_id'], row['obj_id']))
		print "ERROR -- same detection assigned to multiple objects"
		x = rows.det_id
		a1 = np.arange(len(x))[np.diff(x) == 0]
		print rows[a1]
		print rows[a1+1]
	#assert np.all(np.diff(rows.det_id) != 0)

	# Verify that all detections in this cell are linked (cond #2)
	# This can be restricted to only detections from exposures existing in explist
	det_rows = db.query("_ID as det_id, _EXP as exp_id FROM '%s'" % det_tabname).fetch_cell(cell_id)
	if explist is not None:
		det_rows = det_rows[np.in1d(det_rows.exp_id, explist)]
	det_rows.sort(["det_id"])
	assert np.all(np.unique(rows.det_id) == det_rows.det_id), "Not all detections were linked to objects (need to rerun make-object-catalog?): nlinked=%d ntotal=%d cell_id=%s" % (len(rows.det_id), len(det_rows), cell_id)

	yield True

def sanity_check_object_table(db, obj_tabname, det_tabname, radius, explist=None):
	"""
		Verify the following:
		1. Each detection is linked to a single object
		2. Every detection is linked exactly once
		3. Each object has at least one detection linked to it
		4. Each detection is within radius of its parent object
	"""

	qstr = "obj._ID as obj_id, det._ID as det_id, obj._LON as olon, obj._LAT as olat, det._LON as dlon, det._LAT as dlat, det._ISNULL as dnull, det._EXP as exp_id FROM {obj} as obj, {det}(outer) as det".format(obj=obj_tabname, det=det_tabname)
	for result in db.query(qstr).execute([
			(_sanity_check_object_table_mapper, radius, explist),
			(_sanity_check_object_table_reducer, db, det_tabname, explist)
		], group_by_static_cell=True):
		pass
