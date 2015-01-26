try:
	import astropy.io.fits as pyfits
except ImportError:
	import pyfits
import pool2
import numpy as np
from pyslalib.slalib import sla_eqgal
from itertools import izip
import time

sdss_table_def = \
{
	'filters': { 'complevel': 5, 'complib': 'blosc', 'fletcher32': False }, # Enable compression
	'schema': {
		#
		#	 LSD column name      Type    FITS (sweep files) column
		#
		'main': {
			'columns': [
				('sdss_id',		'u8',	''),			# LSD ID
				('ra',			'f8',	'ra'),                  # Right ascension
				('dec',			'f8',	'dec'),                 # Declination
				('l',			'f8',	''),			# Galactic longitude
				('b',			'f8',	''),			# Galactic latitude
				('type',		'i4',	'objc_type'),           # Object type (see SDSS docs for interpretation)
				('flags',		'i4',	'objc_flags'),          # Flags (see SDSS docs for interpretation)
				('flags2',		'i4',	'objc_flags2'),         # Flags (see SDSS docs for interpretation)
				('resolve_status',	'i2',	'resolve_status')       # See SDSS docs for interpretation
			],
			'primary_key' : 'sdss_id',
			'spatial_keys': ('ra', 'dec'),
		},
		'survey': {
			'columns': [
				('run', 		'i4',	'run'),                 # SDSS Run number
				('camcol',		'i4',	'camcol'),              # Camera column
				('field',		'i4',	'field'),               # SDSS field number
				('objid',		'i4',	'id'),                  # SDSS object id (within a given field)
			],
			'exposure_key': 'run',
		},
		'photometry': {
			'columns': [
				('u',			'f4',	''),			# u-band photometry (AB mags)
				('uErr',		'f4',	''),                    # u-band error estimate
				('uExt',		'f4',	''),                    # u-band extinction (from SFD'98)
				('uCalib',		'i2',	''),                    # u-band flags (see SDSS docs)
				('g',			'f4',	''),                    # g-band photometry (AB mags)
				('gErr',		'f4',	''),                    # g-band error estimate
				('gExt',		'f4',	''),                    # g-band extinction (from SFD'98)
				('gCalib',		'i2',	''),                    # g-band flags (see SDSS docs)
				('r',			'f4',	''),                    # r-band photometry (AB mags)
				('rErr',		'f4',	''),                    # r-band error estimate
				('rExt',		'f4',	''),                    # r-band extinction (from SFD'98)
				('rCalib',		'i2',	''),                    # r-band flags (see SDSS docs)
				('i',			'f4',	''),                    # i-band photometry (AB mags)
				('iErr',		'f4',	''),                    # i-band error estimate
				('iExt',		'f4',	''),                    # i-band extinction (from SFD'98)
				('iCalib',		'i2',	''),                    # i-band flags (see SDSS docs)
				('z',			'f4',	''),                    # z-band photometry (AB mags)
				('zErr',		'f4',	''),                    # z-band error estimate
				('zExt',		'f4',	''),                    # z-band extinction (from SFD'98)
				('zCalib',		'i2',	''),                    # z-band flags (see SDSS docs)
			],
		}
	}
}

# SDSS flag definitions
F1_SATURATED		 = (2**18)
F1_BRIGHT		 = (2**1)
F1_BINNED1		 = (2**28)
F1_NODEBLEND		 = (2**6)
F1_EDGE			 = (2**2)
F2_DEBLENDED_AS_MOVING	 = (2**0)
RS_SURVEY_PRIMARY	 = (2**8)

def import_from_sweeps(db, sdss_tabname, sweep_files, create=False, all=False):
	""" Import an SDSS catalog from a collection of SDSS sweep files.

	    Note: Assumes underlying shared storage for all output table
	          cells (i.e., any worker is able to write to any cell).
	"""
	if create:
		# Create the new database
		sdss_table = db.create_table(sdss_tabname, sdss_table_def)
	else:
		sdss_table = db.table(sdss_tabname)

	t0 = time.time()
	at = 0; ntot = 0
	pool = pool2.Pool()
	for (file, nloaded, nin) in pool.imap_unordered(sweep_files, import_from_sweeps_aux, (db, sdss_tabname, all), progress_callback=pool2.progress_pass):
		at = at + 1
		ntot = ntot + nloaded
		t1 = time.time()
		time_pass = (t1 - t0) / 60
		time_tot = time_pass / at * len(sweep_files)
		sfile = "..." + file[-67:] if len(file) > 70 else file
		print('  ===> Imported %-70s [%d/%d, %5.2f%%] +%-6d %9d (%.0f/%.0f min.)' % (sfile, at, len(sweep_files), 100 * float(at) / len(sweep_files), nloaded, ntot, time_pass, time_tot))
	del pool

def import_from_sweeps_aux(file, db, tabname, all=False):
	# import an SDSS run
	dat   = pyfits.getdata(file, 1)
	table = db.table(tabname)

	if not all:
		F1 = F1_BRIGHT | F1_SATURATED | F1_NODEBLEND | F1_EDGE;	# these must not be set for an object to qualify
		F2 = F2_DEBLENDED_AS_MOVING				# these must not be set for an object to qualify	# Compute which objects pass flag cuts

		f1 = dat.field('objc_flags')
		f2 = dat.field('objc_flags2')
		rs = dat.field('resolve_status')
		ok = (rs & RS_SURVEY_PRIMARY != 0) & (f1 & F1 == 0) & (f2 & F2 == 0)
		ok |= all
	else:
		ok = np.ones(len(dat.field('resolve_status')), dtype=bool)

	# Import objects passing some basic quality cuts
	if any(ok != 0):
		# Load the data, cutting on flags
		coldefs = sdss_table_def['schema']['main']['columns'] + sdss_table_def['schema']['survey']['columns']
		cols    = dict(( (name, dat.field(fitsname)[ok])   for (name, _, fitsname) in coldefs if fitsname != ''))
		(ext, flux, ivar, calib) = [ dat.field(col)[ok].transpose()    for col in ['extinction', 'modelflux', 'modelflux_ivar', 'calib_status'] ]

		# Compute magnitudes and related quantities for all bands
		for pos, band in enumerate('ugriz'):
			(fluxB, ivarB, extB, calibB) = (flux[pos], ivar[pos], ext[pos], calib[pos])

			# Compute magnitude from flux
			fluxB[fluxB <= 0] = 0.001
			mag = -2.5 * np.log10(fluxB) + 22.5
			magErr = (1.08574 / fluxB) / np.sqrt(ivarB)

			# Append these to the list of columns
			for suffix, col in izip(['', 'Err', 'Ext', 'Calib'], [mag, magErr, extB, calibB]):
				cols[band + suffix] = col

		# Add computed columns
		(ra, dec) = cols['ra'], cols['dec']
		l = np.empty_like(ra)
		b = np.empty_like(dec)
		for i in xrange(len(ra)):
			(l[i], b[i]) = np.degrees(sla_eqgal(*np.radians((ra[i], dec[i]))))
		cols['l']      = l
		cols['b']      = b

		ids = table.append(cols)
	else:
		ids = []

	yield (file, len(ids), len(ok))
