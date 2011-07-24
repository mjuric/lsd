""" Import rows from a series of FITS files """

import numpy as np
from itertools import izip
import pyfits

class FITSImporter:
	def __init__(self, db, tabname, usecols, dtype, setcols={}, hdus=[1], import_primary_key=False):
		self.tabname = tabname
		self.dtype   = dtype
		self.usecols = usecols
		self.setcols = setcols
		self.hdus    = hdus
		self.import_primary_key = import_primary_key

	def __call__(self, db, fn):
		""" Load a FITS file and import it into the named table

		    To be used as an importer for import_from_chunks
		"""
		hdus = pyfits.open(fn)
		try:
        	        rows = None
        		for hdu_name in self.hdus:
        		        data = hdus[hdu_name].data
        		        rr = np.empty(len(data), dtype=self.dtype)
        		        for dbname, fitsname in self.usecols:
        		                rr[dbname] = data.field(fitsname)
                                rows = np.append(rows, rr) if rows is not None else rr
                finally:
                        hdus.close()

		# Add explicitly set columns (if any)
		if len(self.setcols):
			from lsd.colgroup import ColGroup
			rows = ColGroup(rows)
			for col, (dtype, val) in self.setcols.iteritems():
				a    = np.empty(len(rows), dtype=dtype)
				a[:] = val
				rows[col] = a

		# Append to the table
		ids = db.table(self.tabname).append(rows, _update=self.import_primary_key)
		assert len(ids) == len(rows)

		# Return the total number of rows in the input file, and the number of rows actually
		# imported
		return (len(ids), len(ids))

def get_importer(db, args):
	"""
		This function is part of the importer API.

		It must return a tuple containing:
			1) an importer callable
			2) the list of with which the callable will be called
			   in parallel (usually a list of files)
	"""
	table = db.table(args.table)

	# List of columns with explicitly set values
	setcols = {}
	for cv in args.set:
		(name, val) = cv.split('=')
		if name not in table.dtype.names:
			raise Exception("Unknown column '%s', given as argument to --set" % name)
		dtype = table.dtype.fields[name][0]
		setcols[name] = (dtype, dtype.type(val))

	# List of columns to use, table_column:fits_column
	if args.cols is None:
		# Get the list of columns from the table, except for those
		# beginning with _ (specials, aliases, etc.)
		primary_key = table.primary_key.name if not args.import_primary_key else None
		usecols = [ (name, name) for name in table.dtype.names if name[0] != '_' and name != primary_key and name not in setcols ]
		dtype = table.dtype_for([ name for (name, _) in usecols])
	else:
		cols = [ s.split(':') for s in args.cols.split(',') ]
		usecols = []
		for spec in cols:
			try:
				dbname, fitsname = spec
			except ValueError:
				dbname = fitsname = spec[0]
			if dbname not in table.dtype.names:
				raise Exception("Unknown column '%s' in table %s" % (name, args.table))
			if dbname in setcols:
				raise Exception("Cannot both explicitly set and read a column from file ('%s')" % name)
			usecols.append((dbname, fitsname))
		dtype = table.dtype_for([ name for (name, _) in usecols])

        # HDUs
        hdus = []
        for hdu_name in args.hdus:
                if hdu_name[0] == '#':
                        hdu_name = int(hdu_name[1:])
                hdus.append(hdu_name)

	# Create a FITS importer
	ldr = FITSImporter(db, args.table, usecols, dtype, setcols=setcols, hdus=hdus)

	# Return importer and a list of chunks
	return ldr, args.file

def csv_list(value):
	return [ (s.strip()) for s in value.split(',') ]

def add_arg_parsers(subparsers):
	"""
		This function is part of the importer API.
		
		It must add the required parser(s) and set the get_importer callback.
	"""
	parser = subparsers.add_parser('fits', help=__doc__)
	parser.set_defaults(get_importer=get_importer)
	parser.add_argument('table', help='Name of the table into which to import', type=str)
	parser.add_argument('file', help='One or more fits files. If ending in .gz or .bz2, they are assumed to be compressed.', type=str, nargs='+')
	parser.add_argument('-c', '--cols', help='Comma separated list of <colname>:<file_column>, where <file_column> is the name of the column in the FITS file that is to be loaded into <colname>.', type=str)
	parser.add_argument('--import-primary-key', help='The input files contain the primary key values. Load these instead of automatically asigning new ones.', default=False, action='store_true')
	parser.add_argument('--set', help='Comma separated list of <colname>=<value>, giving the values to which the specified columns are to be set. The columns given here must not appear in argument to --cols.', default=[], type=csv_list)
	parser.add_argument('--hdus', help="Comma separated list of HDU names (or HDU indices) to import. The values are interpreted as a HDU names, unless they start with a '#', in which case they're interpreted as an index (e.g., #1 refers to the first HDU after the primary one).", default=['#1'], type=csv_list)
