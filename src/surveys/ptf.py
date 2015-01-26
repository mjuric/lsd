try:
	import astropy.io.fits as pyfits
except ImportError:
	import pyfits
import lsd.pool2 as pool2
import numpy as np
import time
from pyslalib.slalib import sla_eqgal, sla_caldj
import itertools as it
import sys, os
import logging
import datetime
import lsd.locking as locking
try:
	import astropy.wcs as pywcs
except ImportError:
	import pywcs

conversion_to_int = 1

logger = logging.getLogger("lsd.smf")

# Table defs for exposure table 
exp_table_def = \
{
    'filters': { 'complevel': 5, 'complib': 'blosc', 'fletcher32': False }, # Enable compression
    'schema': {
        #
        #    LSD column name      Type    FITS column      Description
        #
        'main': {
            'columns': [
                ('exp_id',      'u8',   '',         'LSD primary key for this processed image'),
                ('pid',         'u8',   'DBPID',    'PTF DB procimages primary key'),
                ('ptf_field',   'u4',   'PTFFIELD', 'PTF unique field ID'),
                ('fieldid',     'u8',   'DBFIELD',  'Database field ID'),
                ('ccdid',       'u1',   'CCDID',    'CCD number (0..11)'),
                ('nid',         'u4',   'DBNID',    'Database night ID'),
                ('fid',         'u1',   'FILTERID', 'Filter ID'),
                ('equinox',     'f4',   'EQUINOX',  'Celestial coordinate system'),
                ('ra',          'f8',   '',         'Right Ascension of the center of the CCD'),
                ('dec',         'f8',   '',         'Declination of the center of the CCD'),
                ('l',           'f8',   '',         'Galactic lon of the center of the CCD'),
                ('b',           'f8',   '',         'Galactic lat of the center of the CCD'),
                ('mjd',         'f8',   'OBSMJD',   'Time of observation'),
                ('hjd',         'f8',   'HJD',      '[day] Heliocentric Julian Day'),
                ('mjd_import',  'u2',   '',         'MJD-55562 when the exposure was imported into the DB'),
                ('medfwhm',     'f4',   'MEDFWHM',  '[arcsecond] Median FWHM'),
                ('fwhmsex',     'f4',   'FWHMSEX',  '[arcsec] SExtractor SEEING estimate'),
                ('limitmag',    'f4',   'LIMITMAG', '[magnitude/s-arcsecond^2] Limiting magnitude'),
                ('mumax_med',   'f4',   '',         'Median value of mu_max for bright stars'),
                ('mumax_rms',   'f4',   '',         'rms scatter in mu_max for bright stars'),
                ('n_bright',    'i2',   '',         'Number of bright stars used to calculate mumax_med'),
                ('cached',      'bool', '',         'Set to True if this is a row cached from a different cell'),
            ],
            'primary_key' : 'exp_id',
            'exposure_key': 'exp_id',
            'temporal_key': 'mjd',
            'spatial_keys': ('ra', 'dec'),
            "cached_flag" : "cached"
        },
        'abs_photo': {
            'columns': [
                # Sean's calibration vs SDSS
                ('phtcalex',    'bool', 'PHTCALEX', 'Was phot.-cal. module executed?'),
                ('phtcalfl',    'bool', 'PHTCALFL', 'Flag for image is photometric (0=N, 1=Y)'),
                ('pcalrmse',    'f4',   'PCALRMSE', 'RMSE from (zeropoint, extinction) data fit'),
                ('imagezpt',    'f8',   'IMAGEZPT', 'Image magnitude zeropoint'),
                ('colortrm',    'f8',   'COLORTRM', 'Image color term'),
                ('zptsigma',    'f4',   'ZPTSIGMA', 'Robust dispersion of SEx-SDSS magnitudes'),
                ('izporig',     'a10',  'IZPORIG',  'Photometric-calibration origin'),
                ('zprule',      'a10',  'ZPRULE',   'Photometric-calibration method'),
                ('magzpt',      'f8',   'MAGZPT',   'Magnitude zeropoint at airmass=1'),
                ('extinct',     'f8',   'EXTINCT',  'Extinction'),
                # Eran's absolute calibration
                ('apnstdi1',    'u4',   'APNSTDI1', 'Number of standard stars in first iteration'),
                ('apnstdif',    'u4',   'APNSTDIF', 'Number of standard stars in final iteration'),
                ('aprms',       'f4',   'APRMS',    'RMS in mag of final abs phot cal'),
                ('apbsrms',     'f4',   'APBSRMS',  'RMS in mag of final abs phot cal for bright stars'),
                ('apchi2',      'f4',   'APCHI2',   'Chi2 of final abs phot cal'),
                ('apdof',       'f4',   'APDOF',    'Dof of chi2 of final abs phot cal'),
                ('apmedjd',     'f8',   'APMEDJD',  'Median JD used in abs phot cal'),
                ('appar01',     'f8',   'APPAR01',  'Value of parameter abs phot cal 01'),
                ('appare01',    'f8',   'APPARE01', 'Error of parameter abs phot cal 01'),
                ('appar02',     'f8',   'APPAR02',  'Value of parameter abs phot cal 02'),
                ('appare02',    'f8',   'APPARE02', 'Error of parameter abs phot cal 02'),
                ('appar03',     'f8',   'APPAR03',  'Value of parameter abs phot cal 03'),
                ('appare03',    'f8',   'APPARE03', 'Error of parameter abs phot cal 03'),
                ('appar04',     'f8',   'APPAR04',  'Value of parameter abs phot cal 04'),
                ('appare04',    'f8',   'APPARE04', 'Error of parameter abs phot cal 04'),
                ('appar05',     'f8',   'APPAR05',  'Value of parameter abs phot cal 05'),
                ('appare05',    'f8',   'APPARE05', 'Error of parameter abs phot cal 05'),
                ('appar06',     'f8',   'APPAR06',  'Value of parameter abs phot cal 06'),
                ('appare06',    'f8',   'APPARE06', 'Error of parameter abs phot cal 06'),
                ('appar07',     'f8',   'APPAR07',  'Value of parameter abs phot cal 07'),
                ('appare07',    'f8',   'APPARE07', 'Error of parameter abs phot cal 07'),
                ('appar08',     'f8',   'APPAR08',  'Value of parameter abs phot cal 08'),
                ('appare08',    'f8',   'APPARE08', 'Error of parameter abs phot cal 08'),
                ('appar09',     'f8',   'APPAR09',  'Value of parameter abs phot cal 09'),
                ('appare09',    'f8',   'APPARE09', 'Error of parameter abs phot cal 09'),
                ('appar10',     'f8',   'APPAR10',  'Value of parameter abs phot cal 10'),
                ('appare10',    'f8',   'APPARE10', 'Error of parameter abs phot cal 10'),
                ('appar11',     'f8',   'APPAR11',  'Value of parameter abs phot cal 11'),
                ('appare11',    'f8',   'APPARE11', 'Error of parameter abs phot cal 11'),
            ],
        },
        'image_quality': {
            'columns': [
                ('exptime',     'f4', 'AEXPTIME',   'Actual exposure time (sec)'),
                ('airmass',     'f4', 'AIRMASS',    'Observation airmass'),
                ('ra_rms',      'f4', 'RA_RMS',     '[arcsec] RMS of SCAMP fit from 2MASS matching'),
                ('dec_rms',     'f4', 'DEC_RMS',    '[arcsec] RMS of SCAMP fit from 2MASS matching'),
                ('astromn',     'u4', 'ASTROMN',    'Number of stars in SCAMP astrometric solution'),
                ('seeing',      'f4', 'SEEING',     '[pix] Seeing FWHM' ),
                ('peakdist',    'f4', 'PEAKDIST',   'Mean dist brightest pixel-centroid pixel'),
                ('ellip',       'f4', 'ELLIP',      'Mean image ellipticity A/B'),
                ('ellippa',     'f4', 'ELLIPPA',    '[deg] Mean image ellipticity PA'),
                ('saturval',    'f4', 'SATURVAL',   '[DN] Saturation value of the CCD array'),
                ('mdskymag',    'f4', 'MDSKYMAG',   '[magnitude/s-arcsecond^2] Median sky'),
                ('medlong',     'f4', 'MEDELONG',   '[dimensionless] Median elongation'),
                ('stdelong',    'f4', 'STDELONG',   '[dimensionless] Std. dev. of elongation'),
                ('moonillf',    'f4', 'MOONILLF',   '[frac] Moon illuminated fraction'),
                ('moonphas',    'f4', 'MOONPHAS',   '[deg] Moon phase angle'),
            ],
        },
    }
}

# this table is used to convert data types of data in catalogs
catalog_conversion_table = \
    [
        ('xwin_image',          'f4', 'u4', 1000., 'XWIN_IMAGE', 'Windowed position estimate along x'),
        ('ywin_image',          'f4', 'u4', 1000., 'YWIN_IMAGE', 'Windowed position estimate along y'),
        ('x_world',             'f4', 'u4', 1000., 'X_WORLD', 'Barycenter position along world x axis'),
        ('y_world',             'f4', 'u4', 1000., 'Y_WORLD', 'Barycenter position along world y axis'),
        ('errthetawin_image',   'f4', 'i4', 1000., 'ERRTHETAWIN_IMAGE', 'Windowed error ellipse pos angle (CCW'),
        ('x2win_image',         'f4', 'u4', 1000., 'X2WIN_IMAGE', 'Windowed variance along x'),
        ('y2win_image',         'f4', 'u4', 1000., 'Y2WIN_IMAGE', 'Windowed variance along y'),
        ('xywin_image',         'f4', 'i4', 1000., 'XYWIN_IMAGE', 'Windowed covariance between x and y'),
        ('mag_best',            'f4', 'i4', 1000., 'MAG_BEST', 'Best of MAG_AUTO and MAG_ISOCOR'),
        ('magerr_best',         'f4', 'u4', 1000., 'MAGERR_BEST', 'RMS error for MAG_BEST'),
        ('thetawin_image',      'f4', 'i4', 1000., 'THETAWIN_IMAGE', 'Windowed position angle (CCW'),
        ('elongation',          'f4', 'u4', 1000., 'ELONGATION', 'A_IMAGE'),
        ('isoareaf_world',      'f4', 'u4', 1000., 'ISOAREAF_WORLD', 'Isophotal area (filtered) above Detection thres'),
        ('iso0',                'u4', 'u4', 1.,    'ISO0', 'Isophotal area at level 0'),
        ('iso1',                'u4', 'u4', 1.,    'ISO1', 'Isophotal area at level 1'),
        ('iso2',                'u4', 'u4', 1.,    'ISO2', 'Isophotal area at level 2'),
        ('iso3',                'u4', 'u4', 1.,    'ISO3', 'Isophotal area at level 3'),
        ('iso4',                'u4', 'u4', 1.,    'ISO4', 'Isophotal area at level 4'),
        ('iso5',                'u4', 'u4', 1.,    'ISO5', 'Isophotal area at level 5'),
        ('iso6',                'u4', 'u4', 1.,    'ISO6', 'Isophotal area at level 6'),
        ('iso7',                'u4', 'u4', 1.,    'ISO7', 'Isophotal area at level 7'),
        ('flux_best',           'f4', 'f4', 1.,    'FLUX_BEST', 'Best of FLUX_AUTO and FLUX_ISOCOR'),
        ('fluxerr_best',        'f4', 'f4', 1.,    'FLUXERR_BEST', 'RMS error for BEST flux'),
        ('flux_iso',            'f4', 'f4', 1.,    'FLUX_ISO', 'Isophotal flux'),
        ('fluxerr_iso',         'f4', 'f4', 1.,    'FLUXERR_ISO', 'RMS error for isophotal flux'),
        ('flux_aper',           'f4', 'f4', 1.,    'FLUX_APER', 'Flux vector within fixed circular aperture(s)'),
        ('fluxerr_aper',        'f4', 'f4', 1.,    'FLUXERR_APER', 'RMS error vector for aperture flux(es)'),
        ('theta_image',         'f4', 'i4', 1000., 'THETA_IMAGE', 'Position angle (CCW'),
        ('errawin_image',       'f4', 'u4', 1000., 'ERRAWIN_IMAGE', 'RMS windowed pos error along major axis'),
        ('errbwin_image',       'f4', 'u4', 1000., 'ERRBWIN_IMAGE', 'RMS windowed pos error along minor axis'),
        ('thetawin_world',      'f4', 'i4', 1000., 'THETAWIN_WORLD', 'Windowed position angle (CCW'),
        ('errawin_world',       'f4', 'u4', 1000., 'ERRAWIN_WORLD', 'World RMS windowed pos error along major axis'),
        ('errbwin_world',       'f4', 'u4', 1000., 'ERRBWIN_WORLD', 'World RMS windowed pos error along minor axis'),
        ('errthetawin_world',   'f4', 'i4', 1000., 'ERRTHETAWIN_WORLD', 'Windowed error ellipse pos. angle (CCW'),
    ]


# Table defs for detection table ("sources")
det_table_def = \
{
	'filters': { 'complevel': 5, 'complib': 'blosc', 'fletcher32': False }, # Enable compression and checksumming
	'schema': {
		#
		#	 LSD column name      Type    FITS (sweep files) column
		#
		'main': {
			'columns': [
				('det_id',              'u8',   'u8',   1.,		'',                 'LSD source ID'),       # LSD ID
                ('exp_id',              'u8',   'u8',   1.,		'',                 'LSD primary key for this processed image'),
                ('ra',                  'f8',   'f8',   1.,		'ALPHAWIN_J2000',   'Windowed right ascension (J2000)'),
                ('dec',                 'f8',   'f8',   1.,		'DELTAWIN_J2000',   'Windowed declination (J2000)'),
                ('mjd',                 'f8',   'f8',   1.,		'',                 'MJD of observation, copied from the header for convenience'),
                ('fid',                 'u1',	'u1',	1.,     '',					'Filter ID'),
                ('mag_abs',				'u2',	'u2',	1000.,  '',					'Magnitude from Eran\'s absolute photometry'),
                ('magerr_abs',          'u2',	'u2',	1000.,  'MAGERR_AUTO',		'RMS error for AUTO magnitude'),
                ('mu_max',				'i2',	'i2',	1000.,	'MU_MAX',			'Peak surface brightness above background'),
                ('mag_auto',            'i2',	'i2',	1000.,	'MAG_AUTO',			'Kron-like elliptical aperture magnitude'),
                ('mag_aper',            '5i2',	'5i2',	1000.,  'MAG_APER',			'Fixed aperture magnitude vector'),
                ('magerr_aper',         '5u2',	'5u2',	1000.,  'MAGERR_APER',		'RMS error vector for fixed aperture mag.'),
                ('flags',               'u2',   'u2',   1.,		'FLAGS',            'Extraction flags'),
				('imaflags_iso',        'u2',	'u2',	1.,		'IMAFLAGS_ISO',		'FLAG-image flags ORed over the iso. profile'),
				('nimaflags_iso',       'u2',	'u2',	1.,		'NIMAFLAGS_ISO',	'Number of flagged pixels entering IMAFLAGS_ISO'),
#                ('mjd_import',			'u2',   'u2',	1.,		'',					'MJD-55682 when this detection was imported into the DB'),
                ('cached',              'bool', 'bool', 1.,		'',                 'Set to True if this is a row cached from a different cell'),
            ],
            'primary_key': 'det_id',
            'exposure_key': 'exp_id',
            'temporal_key': 'mjd',
            'spatial_keys': ('ra', 'dec'),
            "cached_flag": "cached"
        },
#        'detxy': {
#            'columns': [
#                ('number',              'u4', 'u4', 1.,    'NUMBER',         'Running object number'),
#                ('xpeak_image',         'u4', 'u4', 1.,    'XPEAK_IMAGE',    'x-coordinate of the brightest pixel'),
#                ('ypeak_image',         'u4', 'u4', 1.,    'YPEAK_IMAGE',    'y-coordinate of the brightest pixel'),
#                ('x_image',             'u4', 'u4', 1000., 'X_IMAGE',        'Object position along x'),
#                ('y_image',             'u4', 'u4', 1000., 'Y_IMAGE',        'Object position along y'),
#                ('x2_image',            'u4', 'u4', 1000., 'X2_IMAGE',       'Variance along x'),
#                ('y2_image',            'u4', 'u4', 1000., 'Y2_IMAGE',       'Variance along y'),
#                ('xy_image',            'u4', 'u4', 1000., 'XY_IMAGE',       'Covariance between x and y'),
#                ('errx2_image',         'f8', 'f8', 1.,    'ERRX2WIN_IMAGE', 'Variance of windowed pos along x'),
#                ('erry2_image',         'f8', 'f8', 1.,    'ERRY2WIN_IMAGE', 'Variance of windowed pos along y'),
#                ('errxy_image',         'f8', 'f8', 1.,    'ERRXYWIN_IMAGE', 'Covariance of windowed pos between x and y'),
#                ('a_world',             'u4', 'u4', 1000., 'AWIN_WORLD',     'Windowed profile RMS along major axis (world un'),
#                ('b_world',             'u4', 'u4', 1000., 'BWIN_WORLD',     'Windowed profile RMS along minor axis (world un'),
#                ('theta_j2000',         'i4', 'i4', 1000., 'THETAWIN_J2000', 'Windowed position angle (east of north) (J2000)'),
#                ('isoarea_world',       'u4', 'u4', 1000., 'ISOAREA_WORLD',  'Isophotal area above Analysis threshold'),
#                ('fwhm_image',          'u4', 'u4', 1000., 'FWHM_IMAGE',     'FWHM assuming a gaussian core'),
#                ('mu_threshold',        'i4', 'i4', 1000., 'MU_THRESHOLD',   'Detection threshold above background'),
#                ('background',          'u4', 'u4', 1000., 'BACKGROUND',     'Background at centroid position'),
#                ('threshold',           'u4', 'u4', 1000., 'THRESHOLD',      'Detection threshold above background'),
#                ('class_star',          'u4', 'u4', 1000., 'CLASS_STAR',     'S/G classifier'),
#            ],
#        },
#        'photometry': {
#            'columns': [
#                ('flux_auto',           'f4',  'f4',  1.,      'FLUX_AUTO',       'Flux within a Kron-like elliptical aperture'),
#                ('fluxerr_auto',        'f4',  'f4',  1.,      'FLUXERR_AUTO',    'RMS error for AUTO flux'),
#                ('mag_iso',             'i4',  'i4',  1000.,   'MAG_ISO',         'Isophotal magnitude'),
#                ('magerr_iso',          'u4',  'u4',  1000.,   'MAGERR_ISO',      'RMS error for isophotal magnitude'),
#                ('mag_isocor',          'i4',  'i4',  1000.,   'MAG_ISOCOR',      'Corrected isophotal magnitude'),
#                ('magerr_isocor',       'u4',  'u4',  1000.,   'MAGERR_ISOCOR',   'RMS error for corrected isophotal magnitude'),
#                ('mag_petro',           'i4',  'i4',  1000.,   'MAG_PETRO',       'Petrosian-like elliptical aperture magnitude'),
#                ('magerr_petro',        'u4',  'u4',  1000.,   'MAGERR_PETRO',    'RMS error for PETROsian magnitude'),
#                ('kron_radius',         'u4',  'u4',  1000.,   'KRON_RADIUS',     'Kron apertures in units of A or B'),
#                ('petro_radius',        'u4',  'u4',  1000.,   'PETRO_RADIUS',    'Petrosian apertures in units of A or B'),
#                ('flux_radius',         '5u4', '5u4', 1000.,   'FLUX_RADIUS',     'Fraction-of-light radii'),
#            ],
#        },
	}
}

obj_table_def = \
{
    'filters': { 'complevel': 5, 'complib': 'blosc', 'fletcher32': False }, # Enable compression
    'schema': {
        #
        #    LSD column name      Type    FITS column      Description
        #
        'astrometry': {
            'columns': [
                ('obj_id',  'u8',   '',         'Unique LSD ID of this object'),
                ('ra',      'f8',   'ra_psf',   ''),
                ('dec',     'f8',   'dec_psf',  ''),
                ('cached',  'bool', '',         ''),
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
                ('o2d_id',      'u8', '', 'Unique ID of this row'),
                ('obj_id',      'u8', '', 'Object ID'),
                ('det_id',      'u8', '', 'Detection ID'),
                ('dist',        'f4', '', 'Distance (in degrees)'),
                ('ra',          'f8', 'ra_psf',     ''),
                ('dec',         'f8', 'dec_psf',    ''),
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

def import_from_catalogs(db, det_tabname, exp_tabname, catalog_files, create=False, all=False):
	""" Import a PTF catalog from a collection of SExtractor catalog files.

	    Note: Assumes underlying shared storage for all output table
	          cells (i.e., any worker is able to write to any cell).
	"""
	with locking.lock(db.path[0] + "/.__smf-import-lock.lock"):
		if not db.table_exists(det_tabname) and create:
			# Set up commit hooks
			exp_table_def['commit_hooks'] = [ ('Updating neighbors', 1, 'lsd.smf', 'make_image_cache', [det_tabname]) ]

			# Create new tables
			det_table = db.create_table(det_tabname, det_table_def)
			exp_table = db.create_table(exp_tabname, exp_table_def)

			# Set up a one-to-X join relationship between the two tables (join det_table:exp_id->exp_table:exp_id)
			db.define_default_join(det_tabname, exp_tabname,
				type = 'indirect',
				m1   = (det_tabname, "det_id"),
				m2   = (det_tabname, "exp_id")
				)
		else:
			det_table = db.table(det_tabname)
			exp_table = db.table(exp_tabname)

	# MJD of import
	now = datetime.datetime.now()
	(djm, j) = sla_caldj(now.year, now.month, now.day)
	djm -= 55682

	t0 = time.time()
	at = 0; ntot = 0
	pool = pool2.Pool()
	explist_file = open('explist.txt','w')
	for (file, nloaded, error_type, expID) in pool.imap_unordered(catalog_files, import_from_catalogs_aux, (det_table, exp_table, djm, all), progress_callback=pool2.progress_pass):
		at = at + 1
		ntot = ntot + nloaded
		t1 = time.time()
		time_pass = (t1 - t0) / 60
		time_tot = time_pass / at * len(catalog_files)
#		sfile = (file)[-65:] if len(file) > 70 else file
		sfile = file
		if error_type == 0:
			print('  ===> Imported %s [%d/%d, %5.2f%%] +%-6d %9d (%.0f/%.0f min.)' % (sfile, at, len(catalog_files), 100 * float(at) / len(catalog_files), nloaded, ntot, time_pass, time_tot))
			explist_file.write('%s\n' % str(expID))
		elif error_type == 1:
			print('%s is missing!' % (sfile))
		elif error_type == 2:
			print('%s has bad data type in exposure database!' % (sfile))
		elif error_type == 3:
			print('%s has bad data type in detection database!' % (sfile))
		elif error_type == 4:
			print('%s has bad WCS transform parameters!' % (sfile))
		else:
			print "Nothing"

		# show the dirt every 100 ingests
#		if at % 100. == 0:
#			gc.collect()
#			n_collected = gc.collect()
#			if n_collected > 0:
#				dump_garbage()

	del pool
	explist_file.close()

def import_from_catalogs_aux(file, det_table, exp_table, djm, all=False):
    # read SExtractor catalog file
#	print >>sys.stderr, file
	try:
		hdus = pyfits.open(file)
	except IOError:
		yield (file, 0, 1, 0)
	else:
		dat = hdus[1].data
		hdr_det = hdus[1].header
		hdr_exp = hdus[2].header
		hdus.close()

		# delete PV coefficients from the header
		for i in np.arange(17):
			del hdr_exp['PV1_'+str(i)]
			del hdr_exp['PV2_'+str(i)]
		# test WCS of this image
		try:
			wcs = pywcs.WCS(hdr_exp)
		except:
			yield (file, 0, 4, 0)
		else:
			# import header data into exposures table
			try:
				coldefs = exp_table_def['schema']['main']['columns'] + exp_table_def['schema']['image_quality']['columns']
				exp_cols = dict( (name, np.array([hdr_exp.get(fitsname)]).astype(coltype)) for (name, coltype, fitsname, _) in coldefs if fitsname != '')
				coldefs = exp_table_def['schema']['abs_photo']['columns']
				exp_cols.update(dict( (name, np.array([0]).astype(coltype)) for (name, coltype, fitsname, _) in coldefs if fitsname != ''))
				exp_cols.update(dict( (name, np.array([hdr_exp.get(fitsname)]).astype(coltype)) for (name, coltype, fitsname, _) in coldefs if (fitsname != '') & (hdr_exp.has_key(fitsname))))
			except TypeError:
				yield (file, 0, 2, 0)
			else:
				# find the RA and Dec for the center of this exposure
				pixcrd = np.array([[1024,2048]], np.float_)
				sky = wcs.all_pix2sky(pixcrd, 1)
				exp_cols['ra'] = np.array([sky[0][0]])
				exp_cols['dec'] = np.array([sky[0][1]])
				(l, b) = np.degrees(sla_eqgal(*np.radians((exp_cols['ra'].squeeze(), exp_cols['dec'].squeeze()))))
				exp_cols['l'] = np.array([l])
				exp_cols['b'] = np.array([b])
				exp_cols['mjd_import'] = np.array([djm]).astype('u4')
				exp_cols['mumax_med'] = np.array([-99.99]).astype('f4')
				exp_cols['mumax_rms'] = np.array([-99.99]).astype('f4')
				exp_cols['n_bright'] = np.array([0]).astype('i2')
				# Import sources into detection table
				coldefs = det_table_def['schema']['main']['columns']
				try:
					if (conversion_to_int == 1):
						det_cols = dict(( (name, np.around(dat.field(fitsname)*factor).astype(coltype[-2:])) for (name, _, coltype, factor, fitsname, _) in coldefs if fitsname != '' and factor > 1))
						det_cols.update(dict(( (name, (dat.field(fitsname)*factor).astype(coltype[-2:])) for (name, _, coltype, factor, fitsname, _) in coldefs if fitsname != '' and factor == 1)))
					else:
						det_cols = dict(( (name, dat.field(fitsname).astype(coltype[-2:])) for (name, coltype, _, _, fitsname, _) in coldefs if fitsname != ''))
				except TypeError:
					yield (file, 0, 3, 0)
				else:
					det_cols['mjd'] = (np.zeros(len(det_cols['ra'])) + hdr_exp['OBSMJD']).astype('f8')
					det_cols['fid'] = (np.zeros(len(det_cols['ra'])) + hdr_exp['DBFID']).astype('u1')
#					det_cols['mjd_import'] = (np.zeros(len(det_cols['ra'])) + djm).astype('u2')
					det_cols['mag_abs'] = (np.zeros(len(det_cols['ra'])) + 32000).astype('u2')

					# if there is Eran's absolute photometry...
					if exp_cols['apbsrms'] > 0:
						det_cols['mag_abs'] = np.round((dat.field('MAG_AUTO') + 2.5*np.log10(hdr_exp['AEXPTIME'])+ hdr_exp['APPAR01'] + hdr_exp['APPAR03']*hdr_exp['AIRMASS'] + hdr_exp['APPAR05']*(hdr_exp['OBSJD'] - hdr_exp['APMEDJD']) + hdr_exp['APPAR06']*(hdr_exp['OBSJD'] - hdr_exp['APMEDJD'])**2 + hdr_exp['APPAR07']*(dat.field('X_IMAGE') - 1024.)/2048. + hdr_exp['APPAR08']*(dat.field('Y_IMAGE') - 2048.)/4096. + hdr_exp['APPAR09']*((dat.field('Y_IMAGE') - 2048.)/4096.)**2 + hdr_exp['APPAR10']*((dat.field('Y_IMAGE') - 2048.)/4096.)**3 + hdr_exp['APPAR11']*(dat.field('X_IMAGE') - 1024.)/2048. * (dat.field('Y_IMAGE') - 2048.)/4096.)*1000.).clip(min=5000, max=32000).astype('u2')

						# use bright stars to correct for seeing

			# import header data into exposures table
			try:
				coldefs = exp_table_def['schema']['main']['columns'] + exp_table_def['schema']['image_quality']['columns']
				exp_cols = dict( (name, np.array([hdr_exp.get(fitsname)]).astype(coltype)) for (name, coltype, fitsname, _) in coldefs if fitsname != '')
				coldefs = exp_table_def['schema']['abs_photo']['columns']
				exp_cols.update(dict( (name, np.array([0]).astype(coltype)) for (name, coltype, fitsname, _) in coldefs if fitsname != ''))
				exp_cols.update(dict( (name, np.array([hdr_exp.get(fitsname)]).astype(coltype)) for (name, coltype, fitsname, _) in coldefs if (fitsname != '') & (hdr_exp.has_key(fitsname))))
			except TypeError:
				yield (file, 0, 2, 0)
			else:
				# find the RA and Dec for the center of this exposure
				pixcrd = np.array([[1024,2048]], np.float_)
				sky = wcs.all_pix2sky(pixcrd, 1)
				exp_cols['ra'] = np.array([sky[0][0]])
				exp_cols['dec'] = np.array([sky[0][1]])
				(l, b) = np.degrees(sla_eqgal(*np.radians((exp_cols['ra'].squeeze(), exp_cols['dec'].squeeze()))))
				exp_cols['l'] = np.array([l])
				exp_cols['b'] = np.array([b])
				exp_cols['mjd_import'] = np.array([djm]).astype('u4')
				exp_cols['mumax_med'] = np.array([-99.99]).astype('f4')
				exp_cols['mumax_rms'] = np.array([-99.99]).astype('f4')
				exp_cols['n_bright'] = np.array([0]).astype('i2')
				# Import sources into detection table
				coldefs = det_table_def['schema']['main']['columns']
				try:
					if (conversion_to_int == 1):
						det_cols = dict(( (name, np.around(dat.field(fitsname)*factor).astype(coltype[-2:])) for (name, _, coltype, factor, fitsname, _) in coldefs if fitsname != '' and factor > 1))
						det_cols.update(dict(( (name, (dat.field(fitsname)*factor).astype(coltype[-2:])) for (name, _, coltype, factor, fitsname, _) in coldefs if fitsname != '' and factor == 1)))
					else:
						det_cols = dict(( (name, dat.field(fitsname).astype(coltype[-2:])) for (name, coltype, _, _, fitsname, _) in coldefs if fitsname != ''))
				except TypeError:
					yield (file, 0, 3, 0)
				else:
					det_cols['mjd'] = (np.zeros(len(det_cols['ra'])) + hdr_exp['OBSMJD']).astype('f8')
					det_cols['fid'] = (np.zeros(len(det_cols['ra'])) + hdr_exp['DBFID']).astype('u1')
#					det_cols['mjd_import'] = (np.zeros(len(det_cols['ra'])) + djm).astype('u2')
					det_cols['mag_abs'] = (np.zeros(len(det_cols['ra'])) + 32000).astype('u2')

					# if there is Eran's absolute photometry...
					if exp_cols['apbsrms'] > 0:
						det_cols['mag_abs'] = np.round((dat.field('MAG_AUTO') + 2.5*np.log10(hdr_exp['AEXPTIME'])+ hdr_exp['APPAR01'] + hdr_exp['APPAR03']*hdr_exp['AIRMASS'] + hdr_exp['APPAR05']*(hdr_exp['OBSJD'] - hdr_exp['APMEDJD']) + hdr_exp['APPAR06']*(hdr_exp['OBSJD'] - hdr_exp['APMEDJD'])**2 + hdr_exp['APPAR07']*(dat.field('X_IMAGE') - 1024.)/2048. + hdr_exp['APPAR08']*(dat.field('Y_IMAGE') - 2048.)/4096. + hdr_exp['APPAR09']*((dat.field('Y_IMAGE') - 2048.)/4096.)**2 + hdr_exp['APPAR10']*((dat.field('Y_IMAGE') - 2048.)/4096.)**3 + hdr_exp['APPAR11']*(dat.field('X_IMAGE') - 1024.)/2048. * (dat.field('Y_IMAGE') - 2048.)/4096.)*1000.).clip(min=5000, max=32000).astype('u2')

						# use bright stars to correct for seeing
						bright = np.where( (det_cols['mag_abs'] > 14000) & (det_cols['mag_abs'] < 17000) )
						if len(bright[0]) > 0:
							sg = dat.field('MU_MAX') - dat.field('MAG_AUTO')
							mumax_med = np.median(sg[bright])
							mumax_rms = 0.741*(np.percentile(sg[bright],75) - np.percentile(sg[bright],25))
							exp_cols['mumax_med'] = np.array([mumax_med]).astype('f4')
							exp_cols['mumax_rms'] = np.array([mumax_rms]).astype('f4')
							exp_cols['n_bright'] = np.array([len(bright[0])]).astype('i2')

					(exp_id,) = exp_table.append(exp_cols)
					det_cols['exp_id'] = (np.zeros(len(det_cols['ra'])) + hdr_exp['DBPID']).astype('u8')
					det_cols['exp_id'][:] = exp_id
					ids = det_table.append(det_cols)
					yield (file, len(ids), 0, exp_id)


if __name__ == '__main__':
	make_object_catalog('ptf_obj', 'ptf_det')

