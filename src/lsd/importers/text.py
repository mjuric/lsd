""" Import rows from a series of text files """

import numpy as np
import warnings
from itertools import izip

def conv_bool(s):
	""" Convert string s to bool, recognizing True/False as literals """
	s = s.strip().upper()
	if s == 'TRUE':  return True
	if s == 'FALSE': return False
	return bool(int(s))

def conv_dms(ss):
	""" Convert a hexagessimal d:m:s.s coordinate to decimal """
	ss = ss.strip()
	(d, m, s) = ([ float(v) for v in ss.split(':') ] + [ 0., 0.])[:3]
	v = abs(d) + m/60. + s/3600.
	return v if ss[0] != '-' else -v

def conv_hms(ss):
	""" Convert a hexagessimal h:m:s.s coordinate to decimal """
	return 15.*conv_dms(ss)

def _open_file(fname):
	""" Transparently open bzipped/gzipped/raw file, based on suffix """
	# lifted from numpy.loadtxt
	if fname.endswith('.gz'):
		import gzip
		fh = gzip.GzipFile(fname)
	elif fname.endswith('.bz2'):
		import bz2
		fh = bz2.BZ2File(fname)
	else:
		fh = file(fname)

	return fh

class TextImporter:
	comments = [ '#', ';' ]
	delimiter = None	# Any sequence of whitespaces

	def __init__(self, db, tabname, force, delimiter, usecols, dtype, skip_header=0, dms=[], hms=[], setcols={}):
		self.tabname = tabname
		self.delimiter = delimiter.decode('string_escape')
		self.dtype   = dtype
		self.usecols = usecols
		self.force = force
		self.skip_header = skip_header
		self.setcols = setcols

		# Add user-defined converter to recognize True/False as boolean values
		self.converters = { col: conv_bool for col, (name, type) in izip(usecols, dtype.descr) if np.dtype(type) == np.bool }
		for col, (name, type) in izip(usecols, dtype.descr):
			if np.dtype(type) == np.bool:
				self.converters[col] = conv_bool
			elif name in dms:
				self.converters[col] = conv_dms
			elif name in hms:
				self.converters[col] = conv_hms

	def __call__(self, db, fn):
		""" Load a Text file and import it into the named table

		    To be used as an importer for import_from_chunks
		"""
		# Allow errors in files
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			rows = np.genfromtxt(_open_file(fn), dtype=self.dtype, usecols=self.usecols, converters=self.converters, delimiter=self.delimiter, skip_header=self.skip_header, invalid_raise=not self.force)

		# Count up the number of lines
		nlines = 0
		with _open_file(fn) as fp:
			for s in fp:
				s = s.strip()
				if not s or s[0] in self.comments:
					continue
				nlines += 1

		# Add explicitly set columns (if any)
		if len(self.setcols):
			from lsd.colgroup import ColGroup
			rows = ColGroup(rows)
			for col, (dtype, val) in self.setcols.iteritems():
				a    = np.empty(len(rows), dtype=dtype)
				a[:] = val
				rows[col] = a

		# Append to the table
		ids = db.table(self.tabname).append(rows)
		assert len(ids) == len(rows)

		# Return the total number of rows in the input file, and the number of rows actually
		# imported
		return (len(ids), nlines)

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

	# List of columns to use
	if args.cols is None:
		# Get the list of columns from the table, except for those
		# beginning with _ (specials, aliases, etc.)
		primary_key = table.primary_key.name if not args.import_primary_key else None
		usecolsn = [ name for name in table.dtype.names if name[0] != '_' and name != primary_key and name not in setcols ]
		usecols = range(len(usecolsn))
		dtype = table.dtype_for(usecolsn)
	else:
		cols = [ s.split(':') for s in args.cols.split(',') ]
		usecols = []
		usecolsn = []
		idx = 0
		for spec in cols:
			try:
				name, idx = spec
			except ValueError:
				name, = spec
				idx = idx + 1
			usecols.append(int(idx)-1)
			if name not in table.dtype.names:
				raise Exception("Unknown column '%s' in table %s" % (name, args.table))
			if name in setcols:
				raise Exception("Cannot both explicitly set and read a column from file ('%s')" % name)
			usecolsn.append(name)
		dtype = table.dtype_for(usecolsn)

	# Create a text importer
	ldr = TextImporter(db, args.table, args.force, args.delimiter, usecols, dtype, skip_header=args.skip_header, dms=args.dms, hms=args.hms, setcols=setcols)

	# Return importer and a list of chunks
	return ldr, args.file

def csv_list(value):
	return [ (s.strip()) for s in value.split(',') ]

def add_arg_parsers(subparsers):
	"""
		This function is part of the importer API.
		
		It must add the required parser(s) and set the get_importer callback.
	"""
	parser = subparsers.add_parser('text', help=__doc__)
	parser.set_defaults(get_importer=get_importer)
	parser.add_argument('table', help='Name of the table into which to import', type=str)
	parser.add_argument('file', help='One or more text files. If ending in .gz or .bz2, they are assumed to be compressed.', type=str, nargs='+')
	parser.add_argument('-d', '--delimiter', help='The string used to separate values. By default, any consecutive whitespaces will act as delimiter', type=str)
	parser.add_argument('-c', '--cols', help='Comma separated list of <colname>:<file_column>, where <file_column> is a 1-based index of the column in the file that is to be loaded into <colname>', type=str)
	parser.add_argument('-f', '--force', help='Ignore any errors found in input data files', default=False, action='store_true')
	parser.add_argument('--import-primary-key', help='The input files contain the primary key values. Load these instead of automatically asigning new ones.', default=False, action='store_true')
	parser.add_argument('--skip-header', help='Number of lines to skip at the beginning of each input file', default=0, type=int)
	parser.add_argument('--dms', help='Columns that are stored in hexagessimal dd:mm:ss.ss format', default=[], type=csv_list)
	parser.add_argument('--hms', help='Columns that are stored in hexagessimal hh:mm:ss.ss format', default=[], type=csv_list)
	parser.add_argument('--set', help='Comma separated list of <colname>=<value>, giving the values to which the specified columns are to be set. The columns given here must not appear in argument to --cols.', default=[], type=csv_list)
