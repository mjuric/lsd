import catalog
import pyfits
import pool2
import numpy as np
from slalib import sla_eqgal
from itertools import imap, izip
import time

def initialize_column_definitions():
	# Astrometry table
	astromCols = [
		# column in Catalog	Datatype	Column in sweep files
		('id',			'u8',	''),			# computed column
		('cached', 		'bool',	''),			# computed column
		('ra',			'f8',	'ra'),
		('dec',			'f8',	'dec'),
		('l',			'f8',	''),			# computed column
		('b',			'f8',	''),			# computed column
		('type',		'i4',	'objc_type'),
		('flags',		'i4',	'objc_flags'),
		('flags2',		'i4',	'objc_flags2'),
		('resolve_status',	'i2',	'resolve_status')
	]

	# Survey metadata table
	surveyCols = [
		('run', 		'i4',	'run'),
		('camcol',		'i4',	'camcol'),
		('field',		'i4',	'field'),
		('objid',		'i4',	'id'),
	]

	# Photometry table
	photoCols = [];
	for mag in 'ugriz':
		photoCols.append( (mag,           'f4',	'') )
		photoCols.append( (mag + 'Err',   'f4',	'') )
		photoCols.append( (mag + 'Ext',   'f4',	'') )
		photoCols.append( (mag + 'Calib', 'i2',	'') )

	return (astromCols, surveyCols, photoCols)

def to_dtype(cols):
	return list(( (name, dtype) for (name, dtype, _) in cols ))

(astromCols, surveyCols, photoCols) = initialize_column_definitions();

# SDSS flag definitions
F1_SATURATED		 = (2**18)
F1_BRIGHT		 = (2**1)
F1_BINNED1		 = (2**28)
F1_NODEBLEND		 = (2**6)
F1_EDGE			 = (2**2)
F2_DEBLENDED_AS_MOVING	 = (2**0)
RS_SURVEY_PRIMARY	 = (2**8)

def import_from_sweeps(catdir, sweep_files, create=False):
	""" Import SDSS (stellar) catalog from a collection of SDSS sweep files.

	    Note: Assumes underlying shared storage for all catalog
	          cells (i.e., any worker is able to write to any cell).
	"""
	if create:
		# Create the new database
		cat = catalog.Catalog(catdir, name='sdss', mode='c')
		cat.create_table('astrometry', { 'columns': to_dtype(astromCols), 'primary_key': 'id', 'spatial_keys': ('ra', 'dec'), "cached_flag": "cached" })
		cat.create_table('survey',     { 'columns': to_dtype(surveyCols) })
		cat.create_table('photometry', { 'columns': to_dtype(photoCols) })
	else:
		cat = catalog.Catalog(catdir)

	t0 = time.time()
	at = 0; ntot = 0
	pool = pool2.Pool()
	for (file, nloaded, nin) in pool.imap_unordered(sweep_files, import_from_sweeps_aux, (cat,), progress_callback=pool2.progress_pass):
	#for (file, nloaded, nin) in imap(lambda file: import_from_sweeps_aux(file, cat), sweep_files):
		at = at + 1
		ntot = ntot + nloaded
		t1 = time.time()
		time_pass = (t1 - t0) / 60
		time_tot = time_pass / at * len(sweep_files)
		sfile = "..." + file[-67:] if len(file) > 70 else file
		print('  ===> Imported %-70s [%d/%d, %5.2f%%] +%-6d %9d (%.0f/%.0f min.)' % (sfile, at, len(sweep_files), 100 * float(at) / len(sweep_files), nloaded, ntot, time_pass, time_tot))

def import_from_sweeps_aux(file, cat):
	# import an SDSS run
	dat = pyfits.getdata(file, 1)

	F1 = F1_BRIGHT | F1_SATURATED | F1_NODEBLEND | F1_EDGE;	# these must not be set for an object to qualify
	F2 = F2_DEBLENDED_AS_MOVING				# these must not be set for an object to qualify	# Compute which objects pass flag cuts

	f1 = dat.field('objc_flags')
	f2 = dat.field('objc_flags2')
	rs = dat.field('resolve_status')
	ok = (rs & RS_SURVEY_PRIMARY != 0) & (f1 & F1 == 0) & (f2 & F2 == 0)

	# Import objects
	if any(ok != 0):
		# Load the data, cutting on flags
		cols                     = dict(( (name, dat.field(fitsname)[ok])   for (name, _, fitsname) in astromCols + surveyCols if fitsname != ''))
		(ext, flux, ivar, calib) = [ dat.field(col)[ok].transpose()    for col in ['extinction', 'modelflux', 'modelflux_ivar', 'calib_status'] ]

		# Compute magnitudes in all bands
		for band in xrange(5):
			(fluxB, ivarB, extB, calibB) = (flux[band], ivar[band], ext[band], calib[band])

			# Compute magnitude from flux
			fluxB[fluxB <= 0] = 0.001
			mag = -2.5 * np.log10(fluxB) + 22.5
			magErr = (1.08574 / fluxB) / np.sqrt(ivarB)

			# Append these to the list of columns
			bands = 'ugriz'
			for name, col in izip(['', 'Err', 'Ext', 'Calib'], [mag, magErr, extB, calibB]):
				cols[bands[band] + name] = col

		# Add computed columns
		(ra, dec) = cols['ra'], cols['dec']
		l = np.empty_like(ra)
		b = np.empty_like(dec)
		for i in xrange(len(ra)):
			(l[i], b[i]) = np.degrees(sla_eqgal(*np.radians((ra[i], dec[i]))))
		cols['l']      = l
		cols['b']      = b

		# sanity check
		for (name, _, _) in astromCols + surveyCols + photoCols:
			assert name in cols or name in ['id', 'cached'], name

		ids = cat.append(cols)
	else:
		ids = []

	return (file, len(ids), len(ok))

