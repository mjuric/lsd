import pool2
import numpy as np
import time

conversion_to_int = 0

small_table_def = \
{
	'filters': { 'complevel': 1, 'complib': 'zlib', 'fletcher32': True }, # Enable compression and checksumming
	'schema': {
		#
		'main': {
			'columns': [
				('small_id',	'u8',	'u8',	1.,		'',			'LSD ID'),
				('ra',			'f8',	'f8',	1.,		'ra',		'Right ascension'),
				('dec',			'f8',	'f8',	1.,		'dec',		'Declination'),
				('objid',		'u4',	'u4',	1.,		'objid',	'object id'),
			],
			'primary_key' : 'small_id',
			'spatial_keys': ('ra', 'dec'),
		}
	}
}

def import_from_catalogs(db, small_tabname, catalog_files, create=False):
	""" Import a catalog.

	    Note: Assumes underlying shared storage for all output table
	          cells (i.e., any worker is able to write to any cell).
	"""
	if create:
		# Create the new database
		small_table = db.create_table(small_tabname, small_table_def)
	else:
		small_table = db.table(small_tabname)

	t0 = time.time()
	at = 0; ntot = 0
	pool = pool2.Pool()
	for (file, nloaded) in pool.imap_unordered(catalog_files, import_from_catalogs_aux, (db, small_tabname), progress_callback=pool2.progress_pass):
		at = at + 1
		ntot = ntot + nloaded
		t1 = time.time()
		time_pass = (t1 - t0) / 60
		time_tot = time_pass / at * len(catalog_files)
		sfile = "..." + file[-67:] if len(file) > 70 else file
		print('  ===> Imported %-70s [%d/%d, %5.2f%%] +%-6d %9d (%.0f/%.0f min.)' % (sfile, at, len(catalog_files), 100 * float(at) / len(catalog_files), nloaded, ntot, time_pass, time_tot))
	del pool

def import_from_catalogs_aux(file, db, tabname):
	# import an table
	dat   = np.genfromtxt(file, names='ra, dec, objid', dtype='f8, f8, u4')
	table = db.table(tabname)

	# Load the data, cutting on flags
	coldefs = small_table_def['schema']['main']['columns']
	cols = dict(( (name, dat[fitsname].astype(coltype[-3:])) for (name, coltype, _, _, fitsname, _) in coldefs if fitsname != ''))

	ids = table.append(cols)

	yield (file, len(ids))
