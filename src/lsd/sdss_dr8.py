import pyfits
import pool2
import numpy as np
import time
import locking

conversion_to_int = 1

sdss_table_def = \
{
	'filters': { 'complevel': 5, 'complib': 'blosc', 'fletcher32': False }, # Enable compression
	'schema': {
		#
		#	 LSD column name      Type    FITS (sweep files) column
		#
		'main': {
			'columns': [
				('sdss_id',		'u8',	'u8',	1.,		'',			'LSD ID'),
				('ra',			'f8',	'f8',	1.,		'ra',		'Right ascension'),
				('dec',			'f8',	'f8',	1.,		'dec',		'Declination'),
				('type',		'u1',	'u1',	1.,		'type',		'Object type (resolved = 3, unresolved = 6)'),
				('flags',		'u8',	'u8',	1.,		'flags',	'SDSS photo. flags'),
				('objid',		'u8',	'u8',	1.,		'objid',	'SDSS DR8 database object id'),
				('rExt',		'u2',	'u2',	1000.,	'rExt',		'r-band extinction (from SFD98)'),
				('u',			'u2',	'u2',	1000.,	'u',		'u-band cModelMag (AB mags)'),
				('uErr',		'u2',	'u2',	1000.,	'uErr',		'u-band error estimate'),
				('g',			'u2',	'u2',	1000.,	'g',		'g-band cModelMag (AB mags)'),
				('gErr',		'u2',	'u2',	1000.,	'gErr',		'g-band error estimate'),
				('r',			'u2',	'u2',	1000.,	'r',		'r-band cModelMag (AB mags)'),
				('rErr',		'u2',	'u2',	1000.,	'rErr',		'r-band error estimate'),
				('i',			'u2',	'u2',	1000.,	'i',		'i-band cModelMag (AB mags)'),
				('iErr',		'u2',	'u2',	1000.,	'iErr',		'i-band error estimate'),
				('z',			'u2',	'u2',	1000.,	'z',		'z-band cModelMag (AB mags)'),
				('zErr',		'u2',	'u2',	1000.,	'zErr',		'z-band error estimate'),
			],
			'primary_key' : 'sdss_id',
			'spatial_keys': ('ra', 'dec'),
		}
	}
}

def import_from_sweeps(db, sdss_tabname, sweep_files, create=False):
	""" Import an SDSS catalog from a collection of SDSS sweep files.

	    Note: Assumes underlying shared storage for all output table
	          cells (i.e., any worker is able to write to any cell).
	"""
	with locking.lock(db.path[0] + "/.__smf-import-lock.lock"):
		if not db.table_exists(sdss_tabname) and create:
			# Create the new database
			sdss_table = db.create_table(sdss_tabname, sdss_table_def)
		else:
			sdss_table = db.table(sdss_tabname)

	t0 = time.time()
	at = 0; ntot = 0
	pool = pool2.Pool()
	for (file, nloaded) in pool.imap_unordered(sweep_files, import_from_sweeps_aux, (db, sdss_tabname), progress_callback=pool2.progress_pass):
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
	dat   = np.genfromtxt(file, names='ra, dec, rExt, u, g, r, i, z, uErr, gErr, rErr, iErr, zErr, flags, type, objid', skip_header=1, dtype='f8, f8, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, f4, u8, u1, u8')
	table = db.table(tabname)
	dat['u'] = dat['u'].clip(min=5, max=32)
	dat['g'] = dat['g'].clip(min=5, max=32)
	dat['r'] = dat['r'].clip(min=5, max=32)
	dat['i'] = dat['i'].clip(min=5, max=32)
	dat['z'] = dat['z'].clip(min=5, max=32)
	bad = np.where(dat['uErr'] < 0)
	dat['uErr'][bad] = 32
	bad = np.where(dat['gErr'] < 0)
	dat['gErr'][bad] = 32
	bad = np.where(dat['rErr'] < 0)
	dat['rErr'][bad] = 32
	bad = np.where(dat['iErr'] < 0)
	dat['iErr'][bad] = 32
	bad = np.where(dat['zErr'] < 0)
	dat['zErr'][bad] = 32

	# Load the data, cutting on flags
	coldefs = sdss_table_def['schema']['main']['columns']
	if (conversion_to_int == 1):
		cols = dict(( (name, np.around(np.abs(dat[fitsname]*factor)).astype(coltype[-2:])) for (name, _, coltype, factor, fitsname, _) in coldefs if fitsname != '' and factor > 1))
		cols.update(dict(( (name, dat[fitsname].astype(coltype[-2:])) for (name, _, coltype, factor, fitsname, _) in coldefs if fitsname != '' and factor == 1)))
	else:
		cols = dict(( (name, dat[fitsname].astype(coltype[-2:])) for (name, coltype, _, _, fitsname, _) in coldefs if fitsname != ''))

	ids = table.append(cols)

	yield (file, len(ids))
