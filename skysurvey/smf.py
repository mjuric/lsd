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
	'meta': {
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
	else:
		det_cat = catalog.Catalog(det_catdir)
		exp_cat = catalog.Catalog(exp_catdir)

	det_c2f = gen_cat2fits(det_cat_def)
	exp_c2f = gen_cat2fits(exp_cat_def)
	
	t0 = time.time()
	at = 0; ntot = 0
	pool = pool2.Pool()
	for (file, nloaded, nin) in pool.imap_unordered(smf_files, import_from_smf_aux, (det_cat, exp_cat, det_c2f, exp_c2f)):
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

def load_columns_from_data(dat, c2f):
	cols = dict(( (name, dat.field(fitsname))   for (name, fitsname) in c2f.iteritems() if fitsname != ''))
	return cols

def load_columns_from_header(hdr, c2f):
	cols = dict(( (name, np.array([hdr[fitsname]]))   for (name, fitsname) in c2f.iteritems() if fitsname != ''))
	return cols

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
	# Load the .smf file
	hdus = pyfits.open(file)

	# Do the exposure table entry first; we'll need the
	# image ID to fill in the detections table
	imhdr = hdus[0].header
	exp_cols = load_columns_from_header(imhdr, exp_c2f)
	add_lb(exp_cols)

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
			nrows = nrows + len(dat)

	(exp_id,) = exp_cat.append(exp_cols)

	### Detections
	filterid = imhdr['FILTERID']
	mjd_obs  = imhdr['MJD-OBS']
	cat2type = gen_cat2type(det_cat_def)
	det_cols_all = None
	at = 0
	for (idx, (chip_id, chip_xy)) in enumerate(all_chips()):

		try:	
			dat = hdus[chip_xy + '.psf'].data
		except KeyError:
			continue

		# Slurp all columns from FITS to 
		det_cols  = load_columns_from_data(dat, det_c2f)

		# Add computed columns
		add_lb(det_cols)
		det_cols['filterid']    = np.empty(len(dat), dtype='a6')
		det_cols['mjd_obs']     = np.empty(len(dat), dtype='f8')
		det_cols['exp_id']      = np.empty(len(dat), dtype='u8')
		det_cols['chip_id']     = np.empty(len(dat), dtype='u1')
		det_cols['filterid'][:] = filterid
		det_cols['mjd_obs'][:]  = mjd_obs
		det_cols['exp_id'][:]   = exp_id
		det_cols['chip_id'][:]  = chip_id

		# Create output array, if needed
		if det_cols_all is None:
			det_cols_all = dict(( (col, np.empty(nrows, dtype=cat2type[col])) for col in det_cols ))

		# Store to output array
		for col in det_cols_all:
			det_cols_all[col][at:at+len(det_cols[col])] = det_cols[col]
		at = at + len(det_cols['exp_id'])

		#print "GO", idx, chip_id, chip_xy, len(det_cols['ra'])
	assert at == nrows

	##print "Saving ", len(det_cols_all['ra'])
	ids = det_cat.append(det_cols_all)

	return (file, len(ids), len(ids))
