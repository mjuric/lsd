import catalog
import pyfits
import pool2
import time
import numpy as np
from slalib import sla_eqgal
from itertools import imap

def importDVO(fn, DVO):
	# Create/open the database
	db = StarDB(fn, 6)
	db.table_fields = tableFields;

	# import the entire DVO database
	findcpt = 'find "' + DVO + '" -name "*.cpt"';
	files = shell(findcpt).splitlines();
	at = 0
	for file in files:
		# Load object data
		dat = pyfits.getdata(file, 1)
		ra  = dat.field('ra')
		dec = dat.field('dec')
		cols = [ dat.field(coldef[0]) for coldef in objCols ]

		# Load magnitudes (they're stored in groups of 5 rows)
		mag = pyfits.getdata(file[:-1]+'s', 1)
		magRawCols = [ mag.field(coldef[0]) for coldef in magData ];
		magCols    = [ col[band::5]         for band in xrange(5)    for col in magRawCols ]
		cols += magCols

		# Transform a list of columns into a list of rows, and store
		rows = zip(*cols)
		db.append(ra, dec, rows)

		at = at + 1
		print('    from ' + file + ('[%d/%d, %5.2f%%] +%-6d %9d' % (at, len(files), 100 * float(at) / len(files), len(rows), db.nrows())))

	db.close();


def initialize_column_definitions():
	# Construct table definition
	objCols = [
		('ra',		'f8'),
		('dec',		'f8'),
		('flags',	'u4'),
		('cat_id',	'u4'),
		('ext_id',	'u8')
	];
	magData = [ # FITS field, output suffix, type
		('MAG', '',		'f4'),
		('MAG_ERR', 'Err',	'f4'),
		('FLAGS', 'Flags',	'f4')
		];
	magCols = [];
	for mag in 'grizy': magCols.extend([ (mag + v[1], v[2]) for v in magData ])
	tableFields = objCols + magCols;

	# All columns
	columns = [('id', 'u8'), ('cached', 'b')] +	\
		  [('l',  'f8'), ('b', 'f8')] +		\
		  objCols + 				\
		  magCols

	return (columns, objCols, magData)

(columns, objCols, magData) = initialize_column_definitions();

def import_from_dvo(catdir, dvo_files, create=False):
	""" Import a PS1 catalog from DVO

	    Note: Assumes underlying shared storage for all catalog
	          cells (i.e., any worker is able to write to any cell).
	"""
	if create:
		# Create the new database
		cat = catalog.Catalog(catdir, 'c', columns)
	else:
		cat = catalog.Catalog(catdir)

	t0 = time.time()
	at = 0; ntot = 0
	pool = pool2.Pool()
	for (file, nloaded, nin) in pool.imap_unordered(dvo_files, import_from_dvo_aux, (cat,)):
	#for (file, nloaded, nin) in imap(lambda file: import_from_dvo_aux(file, cat), dvo_files):
		at = at + 1
		ntot = ntot + nloaded
		t1 = time.time()
		time_pass = (t1 - t0) / 60
		time_tot = time_pass / at * len(dvo_files)
		print('  ===> Imported %s [%d/%d, %5.2f%%] +%-6d %9d (%.0f/%.0f min.)' % (file, at, len(dvo_files), 100 * float(at) / len(dvo_files), nloaded, ntot, time_pass, time_tot))

def import_from_dvo_aux(file, cat):
	# Load object data
	dat = pyfits.getdata(file, 1)
	ra  = dat.field('ra')
	dec = dat.field('dec')
	cols = [ dat.field(coldef[0]) for coldef in objCols ]

	# Load magnitudes (they're stored in groups of 5 rows)
	mag = pyfits.getdata(file[:-1]+'s', 1)
	magRawCols = [ mag.field(coldef[0]) for coldef in magData ];
	magCols    = [ col[band::5]         for band in xrange(5)    for col in magRawCols ]
	cols += magCols

	# Compute and add "derived" columns
	l = np.empty_like(ra)
	b = np.empty_like(dec)
	for i in xrange(len(ra)):
		(l[i], b[i]) = np.degrees(sla_eqgal(*np.radians((ra[i], dec[i]))))
	cached = np.zeros(len(ra), dtype='b')

	cols.insert(0, cached)
	cols.insert(1, l)
	cols.insert(2, b)

	# Transform a list of columns into a list of rows, and store
	ids = cat.insert(zip(*cols), ra, dec)

	return (file, len(ids), len(ids))
