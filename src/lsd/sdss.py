import pyfits
import pool2
import numpy as np
from slalib import sla_eqgal
from itertools import izip
import time

sdss_table_def = \
{
	'filters': { 'complevel': 1, 'complib': 'zlib', 'fletcher32': True }, # Enable compression and checksumming
	'schema': {
		#
		#	 LSD column name      Type    FITS (sweep files) column
		#
		'main': {
			'columns': [
				('sdss_id',		'u8',	''),			# computed column
				('ra',			'f8',	'ra'),
				('dec',			'f8',	'dec'),
				('l',			'f8',	''),			# computed column
				('b',			'f8',	''),			# computed column
				('type',		'i4',	'objc_type'),
				('flags',		'i4',	'objc_flags'),
				('flags2',		'i4',	'objc_flags2'),
				('resolve_status',	'i2',	'resolve_status')
			],
			'primary_key' : 'sdss_id',
			'spatial_keys': ('ra', 'dec'),
		},
		'survey': {
			'columns': [
				('run', 		'i4',	'run'),
				('camcol',		'i4',	'camcol'),
				('field',		'i4',	'field'),
				('objid',		'i4',	'id'),
			],
			'exposure_key': 'run',
		},
		'photometry': {
			'columns': [
				('u',			'f4',	''),
				('uErr',		'f4',	''),
				('uExt',		'f4',	''),
				('uCalib',		'i2',	''),
				('g',			'f4',	''),
				('gErr',		'f4',	''),
				('gExt',		'f4',	''),
				('gCalib',		'i2',	''),
				('r',			'f4',	''),
				('rErr',		'f4',	''),
				('rExt',		'f4',	''),
				('rCalib',		'i2',	''),
				('i',			'f4',	''),
				('iErr',		'f4',	''),
				('iExt',		'f4',	''),
				('iCalib',		'i2',	''),
				('z',			'f4',	''),
				('zErr',		'f4',	''),
				('zExt',		'f4',	''),
				('zCalib',		'i2',	''),
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

def import_from_sweeps(db, sdss_tabname, sweep_files, create=False):
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
	for (file, nloaded, nin) in pool.imap_unordered(sweep_files, import_from_sweeps_aux, (db, sdss_tabname), progress_callback=pool2.progress_pass):
		at = at + 1
		ntot = ntot + nloaded
		t1 = time.time()
		time_pass = (t1 - t0) / 60
		time_tot = time_pass / at * len(sweep_files)
		sfile = "..." + file[-67:] if len(file) > 70 else file
		print('  ===> Imported %-70s [%d/%d, %5.2f%%] +%-6d %9d (%.0f/%.0f min.)' % (sfile, at, len(sweep_files), 100 * float(at) / len(sweep_files), nloaded, ntot, time_pass, time_tot))
	del pool

def import_from_sweeps_aux(file, db, tabname):
	# import an SDSS run
	dat   = pyfits.getdata(file, 1)
	table = db.table(tabname)

	F1 = F1_BRIGHT | F1_SATURATED | F1_NODEBLEND | F1_EDGE;	# these must not be set for an object to qualify
	F2 = F2_DEBLENDED_AS_MOVING				# these must not be set for an object to qualify	# Compute which objects pass flag cuts

	f1 = dat.field('objc_flags')
	f2 = dat.field('objc_flags2')
	rs = dat.field('resolve_status')
	ok = (rs & RS_SURVEY_PRIMARY != 0) & (f1 & F1 == 0) & (f2 & F2 == 0)

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
