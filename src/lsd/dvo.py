import catalog
import pyfits
import pool2
import time
import numpy as np
from slalib import sla_eqgal
from itertools import imap, izip
import hashlib

def initialize_column_definitions():
	# Construct table definition
	astromCols = [
		('id',		'u8',	''),		# computed column
		('cached', 	'bool',	''),		# computed column
		('ra',		'f8',	'ra'),
		('dec',		'f8',	'dec'),
		('l',		'f8',	''),		# computed column
		('b',		'f8',	''),		# computed column
		('flags',	'u4',	'flags'),
		('obj_id',	'u4',	'obj_id'),
		('cat_id',	'u4',	'cat_id'),
		('ext_id',	'u8',	'ext_id')
	];

	magData = [ # FITS field, output suffix, type
		('MAG', '',		'f4'),
		('MAG_ERR', 'Err',	'f4'),
		('FLAGS', 'Flags',	'u4')
		];
	photoCols = [];
	for mag in 'grizy':
		photoCols.extend([ (mag + suffix, dtype, fitscol) for (fitscol, suffix, dtype) in magData ])

	return (astromCols, photoCols, magData)

def to_dtype(cols):
	return list(( (name, dtype) for (name, dtype, _) in cols ))

(astromCols, photoCols, magData) = initialize_column_definitions();

def import_from_dvo(catdir, dvo_files, create=False):
	""" Import a PS1 catalog from DVO

	    Note: Assumes underlying shared storage for all catalog
	          cells (i.e., any worker is able to write to any cell).
	"""
	if create:
		# Create the new database
		cat = catalog.Table(catdir, name='ps1', mode='c')
		cat.create_table('astrometry', { 'columns': to_dtype(astromCols), 'primary_key': 'id', 'spatial_keys': ('ra', 'dec'), "cached_flag": "cached" })
		cat.create_table('photometry', { 'columns': to_dtype(photoCols) })
		cat.create_table('import',     { 'columns': [
				('file_id', 'a20'),
				('hdr', 'i8'),
				('cksum', 'a32'),
				('imageid',	'64i8'),
				('blobarr',	'64i8'),
			],
			'blobs': [
				'hdr',
				'blobarr'
			] })
	else:
		cat = catalog.Table(catdir)

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
	dat, hdr = pyfits.getdata(file, 1, header=1)
	cols     = dict(( (name, dat.field(fitsname))   for (name, _, fitsname) in astromCols if fitsname != ''))

	# Load magnitudes (they're stored in groups of 5 rows)
	mag = pyfits.getdata(file[:-1]+'s', 1)
	magRawCols = [ mag.field(fitsname) for fitsname, _, _ in magData ];
	photoData  = [ col[band::5]        for band in xrange(5)    for col in magRawCols ]
	assert len(photoData) == len(photoCols)
	for i, (col, _, _) in enumerate(photoCols):
		cols[col] = photoData[i]

	#idx = np.arange(len(cols['cat_id']))
	#i   = idx.argsort(
	#ra  = np.sort(cols['ra']);   dra = np.diff(ra)
	#dec = np.sort(cols['dec']); ddec = np.diff(dec)
	#assert (abs(dra) + abs(ddec) > 1.e-14).all(), str(sorted(dra)[:10]) + '\n' + str(sorted(ddec)[:10])
	#assert len(np.unique1d(cols['obj_id'])) == len(cols['obj_id'])
	#assert len(np.unique1d(cols['ext_id'])) == len(cols['ext_id'])
	#assert len(np.unique1d(cols['cat_id'])) == len(cols['cat_id'])

	#print ''
	#for i, col in enumerate(magRawCols):
	#	print magData[i][0],'=',col[10:15]
	#for col, _, _ in photoCols:
	#	print col,'=',cols[col][2]
	#exit()

	# Add computed columns
	(ra, dec) = cols['ra'], cols['dec']
	l = np.empty_like(ra)
	b = np.empty_like(dec)
	for i in xrange(len(ra)):
		(l[i], b[i]) = np.degrees(sla_eqgal(*np.radians((ra[i], dec[i]))))
	cols['l']      = l
	cols['b']      = b

	# Add the ID of the file the object came from
	fn = '/'.join(file.split('/')[-2:])
	assert(len(fn) < 20)
	cols['file_id'] = np.empty(len(l), dtype='a20')
	cols['file_id'][:] = fn

	# Add some blobs (this is mostly for testing)
	cols['hdr'] = np.empty(len(l), dtype=np.object_)
	cols['hdr'][:] = str(hdr)

	# BLOB checksum (for debugging)
	cols['cksum'] = np.empty(len(l), dtype='a32')
	cols['cksum'][:] = hashlib.md5(str(hdr)).hexdigest()

	# Add some array data
	cols['imageid'] = np.empty(len(l), dtype='64i8')
	cols['imageid'][:] = np.arange(len(l)*64, dtype='i8').reshape((len(l), 64))

	# Add a blob array
	s = np.array([ str(i) for i in np.random.random_integers(0, 100, 64) ])
	cols['blobarr'] = np.empty(len(l), dtype='64O')
	cols['blobarr'][:] = s[ np.random.random_integers(0, 63, len(l)*64) ].reshape((len(l), 64))

	# sanity check
	for (name, _, _) in astromCols + photoCols:
		assert name in cols or name in ['id', 'cached'], name

	ids = cat.append(cols)

	return (file, len(ids), len(ids))
