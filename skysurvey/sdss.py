import catalog
import pyfits
import pool2
import numpy as np

def initialize_column_definitions():
	objCols = [
		# column in Catalog	Datatype	Column in sweep files
		('run', 		'i4',	'run'),
		('camcol',		'i4',	'camcol'),
		('field',		'i4',	'field'),
		('objid',		'i4',	'id'),

		('ra',			'f8',	'ra'),
		('dec',			'f8',	'dec'),
		('type',		'i4',	'objc_type'),
		('flags',		'i4',	'objc_flags'),
		('flags2',		'i4',	'objc_flags2'),
		('resolve_status',	'i2',	'resolve_status')
	]
	magCols = [];
	for mag in 'ugriz':
		magCols.append( (mag,           'f4') )
		magCols.append( (mag + 'Err',   'f4') )
		magCols.append( (mag + 'Ext',   'f4') )
		magCols.append( (mag + 'Calib', 'f4') )

	columns = [('id', 'u8'), ('cached', 'b')] + [f[0:2] for f in objCols] + magCols
	return (columns, objCols)

(columns, objCols) = initialize_column_definitions();

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
		cat = catalog.Catalog(catdir, 'c', columns)
	else:
		cat = catalog.Catalog(catdir)

	at = 0; ntot = 0
	pool = pool2.Pool(4)
	for (file, nloaded, nin) in pool.imap_unordered(sweep_files, import_from_sweeps_aux, (cat,)):
	#for (file, nloaded, nin) in imap(lambda file: import_from_sweeps_aux(file, catdir), sweep_files):
		at = at + 1
		ntot = ntot + nloaded
		print('  ===> Imported ' + file + ('[%d/%d, %5.2f%%] +%-6d %9d' % (at, len(sweep_files), 100 * float(at) / len(sweep_files), nloaded, ntot)))

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
		cols                     = [ dat.field(col[2])[ok]             for col in objCols ]
		(ext, flux, ivar, calib) = [ dat.field(col)[ok].transpose()    for col in ['extinction', 'modelflux', 'modelflux_ivar', 'calib_status'] ]
		(ra, dec)                = cols[4], cols[5]

		# Compute magnitudes in all bands
		for band in xrange(5):
			(fluxB, ivarB, extB, calibB) = (flux[band], ivar[band], ext[band], calib[band])

			fluxB[fluxB <= 0] = 0.001
			mag = -2.5 * np.log10(fluxB) + 22.5
			magErr = (1.08574 / fluxB) / np.sqrt(ivarB)

			cols += [mag, magErr, extB, calibB]

		cached = np.zeros(len(ra), dtype='b')
		cols.insert(0, cached)

		# Transform a list of columns into a list of rows, and store
		ids = cat.insert(zip(*cols), ra, dec)
	else:
		ids = []

	return (file, len(ids), len(ok))

