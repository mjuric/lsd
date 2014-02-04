import pyfits
import pool2
import numpy as np
import time
import locking

conversion_to_int = 1

twomass_table_def = \
{
	'filters': { 'complevel': 5, 'complib': 'blosc', 'fletcher32': False }, # Enable compression
	'schema': {
		#
		#	 LSD column name      Type    FITS (sweep files) column
		#
		'main': {
			'columns': [
				('twomass_id',	'u8',	'u8',	1.,		'',			'LSD ID'),
				('ra',			'f8',	'f8',	1.,		'ra',		'Right ascension'),
				('dec',			'f8',	'f8',	1.,		'dec',		'Declination'),
				('J',			'u2',	'u2',	1000.,	'J',		'J-band (Vega mags)'),
				('JErr',		'u2',	'u2',	1000.,	'JErr',		'J-band error estimate'),
				('H',			'u2',	'u2',	1000.,	'H',		'H-band (Vega mags)'),
				('HErr',		'u2',	'u2',	1000.,	'HErr',		'H-band error estimate'),
				('K',			'u2',	'u2',	1000.,	'K',		'K-band (Vega mags)'),
				('KErr',		'u2',	'u2',	1000.,	'KErr',		'K-band error estimate'),
			],
			'primary_key' : 'twomass_id',
			'spatial_keys': ('ra', 'dec'),
		}
	}
}

def import_from_sweeps(db, twomass_tabname, sweep_files, create=False):
	""" Import an 2MASS catalog from a collection of 2MASS files.

	    Note: Assumes underlying shared storage for all output table
	          cells (i.e., any worker is able to write to any cell).
	"""
	with locking.lock(db.path[0] + "/.__smf-import-lock.lock"):
		if not db.table_exists(twomass_tabname) and create:
			# Create the new database
			twomass_table = db.create_table(twomass_tabname, twomass_table_def)
		else:
			twomass_table = db.table(twomass_tabname)

	t0 = time.time()
	at = 0; ntot = 0
	pool = pool2.Pool()
	for (file, nloaded) in pool.imap_unordered(sweep_files, import_from_sweeps_aux, (db, twomass_tabname), progress_callback=pool2.progress_pass):
		at = at + 1
		ntot = ntot + nloaded
		t1 = time.time()
		time_pass = (t1 - t0) / 60
		time_tot = time_pass / at * len(sweep_files)
		sfile = "..." + file[-67:] if len(file) > 70 else file
		print('  ===> Imported %-70s [%d/%d, %5.2f%%] +%-6d %9d (%.0f/%.0f min.)' % (sfile, at, len(sweep_files), 100 * float(at) / len(sweep_files), nloaded, ntot, time_pass, time_tot))
	del pool

def import_from_sweeps_aux(file, db, tabname):
	# import a 2MASS catalog
	dat   = np.genfromtxt(file, names='ra, dec, J, JErr, H, HErr, K, KErr', skip_header=1, dtype='f8, f8, f4, f4, f4, f4, f4, f4')
	table = db.table(tabname)
	dat['J'] = dat['J'].clip(min=2, max=32)
	dat['H'] = dat['H'].clip(min=2, max=32)
	dat['K'] = dat['K'].clip(min=2, max=32)
	bad = np.where(dat['JErr'] < 0)
	dat['JErr'][bad] = 32
	bad = np.where(dat['HErr'] < 0)
	dat['HErr'][bad] = 32
	bad = np.where(dat['KErr'] < 0)
	dat['KErr'][bad] = 32

	# Load the data, cutting on flags
	coldefs = twomass_table_def['schema']['main']['columns']
	if (conversion_to_int == 1):
		cols = dict(( (name, np.around(np.abs(dat[fitsname]*factor)).astype(coltype[-2:])) for (name, _, coltype, factor, fitsname, _) in coldefs if fitsname != '' and factor > 1))
		cols.update(dict(( (name, dat[fitsname].astype(coltype[-2:])) for (name, _, coltype, factor, fitsname, _) in coldefs if fitsname != '' and factor == 1)))
	else:
		cols = dict(( (name, dat[fitsname].astype(coltype[-2:])) for (name, coltype, _, _, fitsname, _) in coldefs if fitsname != ''))

	ids = table.append(cols)

	yield (file, len(ids))
