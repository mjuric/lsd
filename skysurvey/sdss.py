import catalog
import pyfits
import pool2
import utils
import numpy as np
from slalib import sla_eqgal
from table import Table

def initialize_column_definitions():
	# Columns from sweep files
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

	# Magnitude-related columns
	magBands = ['u', 'g', 'r', 'i', 'z']
	magSuffix = ['', 'Err', 'Ext', 'Calib']
	magCols = [];
	for mag in magBands:
		for suffix in magSuffix:
			magCols.append( (mag + suffix, 'f4') )

	# All columns
	columns = [('id', 'u8'), ('cached', 'bool')] +	\
		  [('l',  'f8'), ('b', 'f8')] +		\
		  [f[0:2] for f in objCols] +		\
		  magCols

	return (columns, objCols, magCols, magBands, magSuffix)

(columns, objCols, magCols, magBands, magSuffix) = initialize_column_definitions();

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

	catalog.accumulate = True
	at = 0; ntot = 0
	pool = pool2.Pool()
	for (file, nloaded, nin) in pool.imap_unordered(sweep_files, import_from_sweeps_aux, (cat,)):
	#for (file, nloaded, nin) in imap(lambda file: import_from_sweeps_aux(file, cat), sweep_files):
		at = at + 1
		ntot = ntot + nloaded
#		print('  ===> Imported ' + file + ('[%d/%d, %5.2f%%] +%-6d %9d' % (at, len(sweep_files), 100 * float(at) / len(sweep_files), nloaded, ntot)))

	catalog.accumulate = False
	files = utils.shell('find "' + catdir + '" -name "*.pkl"').splitlines()
	for fn in files:
		path = fn[:fn.rfind('/')];
		catalog.ColTable(path)

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
	cols = Table()
	if any(ok != 0):
		# Load the data, cutting on flags
		for (catcol, _, sweepcol) in objCols:
			cols[catcol] = dat.field(sweepcol)[ok]

		# Compute magnitudes in all bands
		(ext, flux, ivar, calib) = [ dat.field(col)[ok].transpose()    for col in ['extinction', 'modelflux', 'modelflux_ivar', 'calib_status'] ]
		for band in xrange(5):
			(fluxB, ivarB, extB, calibB) = (flux[band], ivar[band], ext[band], calib[band])

			fluxB[fluxB <= 0] = 0.001
			mag = -2.5 * np.log10(fluxB) + 22.5
			magErr = (1.08574 / fluxB) / np.sqrt(ivarB)

			b = magBands[band]
			cols[b] = mag
			cols[b + 'Err'] = magErr
			cols[b + 'Ext'] = extB
			cols[b + 'Calib'] = calibB

		# Compute and add "derived" columns
		(ra, dec) = cols["ra"], cols["dec"]
		l = np.empty_like(ra)
		b = np.empty_like(dec)
		for i in xrange(len(ra)):
			(l[i], b[i]) = np.degrees(sla_eqgal(*np.radians((ra[i], dec[i]))))
		cached = np.zeros(len(ra), dtype='bool')

		cols["cached"] = cached
		cols["l"] = l
		cols["b"] = b

		# Transform a list of columns into a list of rows, and store
		ids = cat.append(cols, ra, dec)
	else:
		ids = []

	return (file, len(ids), len(ok))

