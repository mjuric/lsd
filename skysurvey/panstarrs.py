def importDVO(fn, DVO):
	""" Import Pan-STARRS data from a DVO database.
	
	    Note: Hasn't yet been ported to new implementation of Catalog
	"""
	# Construct table definition
	objFields = [
		('ra',		np.float64),
		('dec',		np.float64),
		('flags',	np.uint32),
		('cat_id',	np.uint32),
		('ext_id',	np.uint64)
	];
	magData = [ # FITS field, output suffix, type
		('MAG', '',		np.float32),
		('MAG_ERR', 'Err',	np.float32),
		('FLAGS', 'Flags',	np.float32)
		];
	magFields = [];
	for mag in 'grizy': magFields.extend([ (mag + v[1], v[2]) for v in magData ])
	tableFields = objFields + magFields;

	# Create the new database
	cat = Catalog(fn, 6)
	cat.table_fields = tableFields;

	# import the entire DVO database
	findcpt = 'find "' + DVO + '" -name "*.cpt"';
	files = utils.shell(findcpt).splitlines();
	at = 0
	for file in files:
		# Load object data
		dat = pyfits.getdata(file, 1)
		ra  = dat.field('ra')
		dec = dat.field('dec')
		cols = [ dat.field(coldef[0]) for coldef in objFields ]

		# Load magnitudes (they're stored in groups of 5 rows)
		mag = pyfits.getdata(file[:-1]+'s', 1)
		magRawCols = [ mag.field(coldef[0]) for coldef in magData ];
		magCols    = [ col[band::5]         for band in xrange(5)    for col in magRawCols ]
		cols += magCols

		# Transform a list of columns into a list of rows, and store
		rows = zip(*cols)
		cat.insert(rows, ra, dec)

		at = at + 1
		print('    from ' + file + ('[%d/%d, %5.2f%%] +%-6d %9d' % (at, len(files), 100 * float(at) / len(files), len(rows), cat.nrows())))

	cat.close();

