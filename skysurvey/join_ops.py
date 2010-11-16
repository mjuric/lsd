import catalog

#
# The result is a J = Table() with id, idx, isnull columns for each joined catalog
# To resolve a column from table X, load its tablet and do rows[J['X.idx']]
#
def join_catalog(cat, join_tree, tablecache):
	# Load subordinates
	joins = []
	for c, subjoin, jointype in joins:
		joins += [ (join_catalog(c, subjoin, tablecache), jointype) ]

	# Load ourselves
	keys = Table(cat.load_primary(tablecache))
	
	# Cull based on bounds
	keys = cut_on_bounds(bounds)

	# Perform joins
	for idx, jointype in idx:
		idx1, idx2, isnull = do_join(keys['primary'], idx['primary'], jointype)
		keys = keys[idx1]
		idx = idx[idx2]
		keys.append_table(idx)
		keys.add_column('primary.isnull', isnull)

	# Return
	return keys

class ColDict:
	catalogs = None		# Cache of loaded catalogs and tables
	columns  = None		# Cache of already referenced columns
	cell_id  = None		# cell_id on which we're operating
	
	primary_catalog = None	# the primary catalog
	include_cached = None	# whether we should include the cached data within the cell

	orig_rows= None		# Debugging/sanity checking: dict of catname->number_of_rows that any tablet of this catalog correctly fetched with fetch_tablet() should have

	def __init__(self, query, cat, cell_id, bounds, include_cached):

		self.cell_id = cell_id
		self.columns = {}
		self.primary_catalog = cat.name
		self.include_cached = include_cached

		# parse query
		(select_clause, where_clause, join_clause) = qp.parse(query, TableColsProxy(cat))
		#print (query, select_clause, where_clause, join_clause)
		#exit()

		# Fetch all rows of the base table, including the cached ones (if requested)
		rows2 = cat.fetch_tablet(cell_id=cell_id, table=cat.primary_table, include_cached=include_cached)
		idx2  = np.arange(len(rows2))
		self.orig_rows = { cat.name: len(rows2) }

#		print >> sys.stderr, "CELL:", cell_id
		# Reject objects out of bounds, in space and time (the latter only if applicable)
		have_bounds_t = False
		if len(rows2) and bounds is not None:
			schema = cat._get_schema(cat.primary_table)
			raKey, decKey = schema["spatial_keys"]
			tKey          = schema["temporal_key"] if "temporal_key" in schema else None

			# inbounds[j, i] is true if the object in row j falls within bounds[i]
			inbounds = np.ones((len(rows2), len(bounds)), dtype=np.bool)
			x, y = None, None

			for (i, (bounds_xy, bounds_t)) in enumerate(bounds):
				if bounds_xy is not None:
					if x is None:
#						print "IN", len(rows2)
						(x, y) = bhpix.proj_bhealpix(rows2[raKey], rows2[decKey]) 
#						print "OUT"

#					print "INp", len(rows2)
					inbounds[:, i] &= bounds_xy.isInsideV(x, y)
#					print "OUTp"

				if bounds_t is not None:
					have_bounds_t = True
					if tKey is not None:
#						print "INt", len(rows2)
						inbounds[:, i] &= bounds_t.isInside(rows2[tKey])
#						print "OUTt"

			# Remove all objects that don't fall inside _any_ of the bounds
			in_  = np.any(inbounds, axis=1)

#			print >> sys.stderr, "Primary table filter: (all), (kept by each)", len(inbounds), np.sum(inbounds, axis=0)
#			exit();

			idx2 = idx2[in_]
			inbounds = inbounds[in_]

		# Initialize catalogs and table lists, plus the array of primary keys
		self.catalogs = \
		{
			cat.name:
			{
				'cat':    cat,
				'join' :  (idx2, np.zeros(len(idx2), dtype=bool)),
				'tables':
				{
					cat.primary_table: rows2
				}
			}
		}
#		self.keys     = rows2[idx2][cat.tables[cat.primary_table]["primary_key"]]
		self.keys     = rows2[cat.tables[cat.primary_table]["primary_key"]][idx2]

		def fetch_cached_tablet(cat, cell_id, table):
			""" A nested function designed to return the tablet
			    from cache (if loaded), or call cat.fetch_tablet
			    otherwise.
			    
			    Used as a parameter to cat.fetch_joins
			"""
			if cat.name in self.catalogs and table in self.catalogs[cat.name]['tables']:
				return self.catalogs[cat.name]['tables'][table]
				
			# TODO: we should cache the resulting tablet, for ColDict use later on
			#       (marginally non-trivial, need to be careful about caching and which
			#       rows to keep and which not to)
			return cat.fetch_tablet(cell_id, table)

		# Load catalog indices to be joined
		for (catname, join_type) in join_clause:
			if cat.name == catname:
				continue

			assert catname not in self.catalogs, "Same catalog, '%s', listed twice in XMATCH clause" % catname;
			assert include_cached == False, "include_cached=True in JOINs is a recipe for disaster. Don't do it"

			# load the xmatch table and instantiate the second catalog object
			(m1, m2) = cat.fetch_joins(cell_id, self.keys, catname, fetch_tablet=fetch_cached_tablet)
			cat2     = cat.get_joined_catalog(catname)
			rows2    = cat2.fetch_tablet(cell_id=cell_id, table=cat2.primary_table, include_cached=True)
			schema   = cat2._get_schema(cat2.primary_table)
			id2      = rows2[schema["primary_key"]]
			self.orig_rows[cat2.name] = len(rows2)

			# Join the tables (jmap and rows2), using (m1, m2) linkage information
			#table_join.cell_id = cell_id	# debugging (remove once happy)
			#table_join.cat = cat		# debugging (remove once happy)
			(idx1, idx2, isnull) = table_join(self.keys, id2, m1, m2, join_type)

			# Reject rows that are out of the time interval in this table.
			# We have to do this here as well, to support filtering on time in static_sky->temporal_sky joins
			if len(rows2) and have_bounds_t and "temporal_key" in schema:
				tKey = schema["temporal_key"]
				t    = rows2[tKey][idx2]
				in_  = np.zeros(len(idx2), dtype=bool)

				# This essentially looks for at least one bound specification that contains a given row
				for (i, (_, bounds_t)) in enumerate(bounds):
					if bounds_t is not None:
						in_t = bounds_t.isInside(t)
						in_ |= inbounds[idx1, i] & in_t
					else:
						in_ |= inbounds[idx1, i]

#				print >> sys.stderr, "Time join (all, kept): ", len(in_), in_.sum()
				#if len(in_) !=  in_.sum():
				#	print "T[~in]    =",    t[~in_]
				#	print "idx1[~in] =", idx1[~in_]
				#	print "idx1[in ] =", idx1[in_][0:10]
				#	print "idx1      =", idx1[0:20]
				#	print "idx2      =", idx2[0:20]
				#	assert 0, "Test this codepath"

				if not in_.all():
					# Keep only the rows that were found to be within at least one bound
					idx1   =   idx1[in_]
					idx2   =   idx2[in_]
					isnull = isnull[in_]

			# update the keys and index maps for the already joined catalogs
			self.keys = self.keys[idx1]
			for (catname, v) in self.catalogs.iteritems():
				(idx, isnull_) = v['join']
				v['join']      = idx[idx1], isnull_[idx1]

			#print "XX:", len(self.keys), len(rows2), idx1, idx2, isnull

			# add the newly joined catalog
			rownums  = np.arange(max(len(rows2), 1)) # The max(..,1) is for outer joins where len(rows2) = 0,
								 # but table_join (below) returns idx2=0 (with isnull=True).
								 # Otherwise, line (*) will fail
			self.catalogs[cat2.name] = \
			{
				'cat': 		cat2,
				'join':		(rownums[idx2], isnull),	# (*) -- see above
				'tables':
				{
					cat2.primary_table: rows2
				}
			}
			assert len(self.catalogs[cat2.name]['join'][0]) == len(self.keys)

		# In the cached tables, keep only the rows that remain after the JOIN has been performed
		for catname in self.catalogs:
			tables = self.catalogs[catname]['tables']
			for table, rows in tables.iteritems():
				#if(len(self.keys)): print 'Before(%s.%s): %d' % (catname, table, len(rows))
				tables[table] = self._filter_joined(rows, catname)
				#if(len(self.keys)): print 'After(%s.%s): %d' % (catname, table, len(tables[table]))

		#if(len(self.keys)):
		#	print "Nonzero match!", len(self.keys)
		#	exit()

		# eval individual columns in select clause to slurp them up from disk
		# and have them ready for the where clause
		retcols = []
		nrows = None
		global_ = globals()
		for (asname, name) in select_clause:
			col = eval(name, global_, self)

			self[asname] = col
			retcols.append(asname)

			if nrows != None:
				assert nrows == len(col)
			nrows = len(col)

		# evaluate the WHERE clause, to obtain the final filter
		in_    = np.empty(nrows, dtype=bool)
		in_[:] = eval(where_clause, global_, self)

		if nrows == 0:
			# We need to handle this separately, because of multidimensional columns
			self.rows_ = Table( [ (name, self[name]) for name in retcols ] )
		else:
			self.rows_ = Table( [ (name, self[name][in_]) for name in retcols ] )

	def rows(self):
		return self.rows_

	def _filter_joined(self, rows, catname):
		# Join
		cat = self.catalogs[catname]['cat']
		idx, isnull = self.catalogs[catname]['join']
		if len(rows) == 0:
			rows = np.zeros(1, dtype=rows.dtype)
		rows = rows[idx]
		if len(rows):
			for name in rows.dtype.names:
				rows[name][isnull] = cat.NULL
		return rows

	def load_column(self, name, table, catname):
		# Load the column from table 'table' of the catalog 'catname'
		# Also cache the loaded tablet, for future reuse

		cat = self.catalogs[catname]['cat']
		include_cached = self.include_cached if catname == self.primary_catalog else True

		# See if we have already loaded the required tablet
		if table in self.catalogs[catname]['tables']:
			#print table, name, ':', self.catalogs[catname]['tables'][table].dtype.names
			col = self.catalogs[catname]['tables'][table][name]
		else:
			# Load
			rows = cat.fetch_tablet(self.cell_id, table, include_cached=include_cached)
			assert len(rows) == self.orig_rows[catname]

			# Join
			rows = self._filter_joined(rows, catname)

			# Cache
			self.catalogs[catname]['tables'][table] = rows

			# return the requested column (further caching is the responsibility of the caller)
			col = rows[name]

		# Resolve blobs (if blob)
		schema = cat._get_schema(table)
		if 'blobs' in schema and name in schema['blobs']:
			assert col.dtype == np.int64, "Data structure error: blob reference columns must be of int64 type"
			refs = col
			col = cat.fetch_blobs(cell_id=self.cell_id, table=table, column=name, refs=refs, include_cached=include_cached)
			assert refs.shape == col.shape

		# Return the resolved column
		return col.view(iarray)

	def __getitem__(self, name):
		# An already loaded column?
		if name in self.columns:
			return self.columns[name]

		# A yet unloaded column from one of the joined catalogs
		# (including the primary)? Try to find it in tables of
		# joined catalogs. It may be prefixed by catalog name, in
		# which case we force the lookup of that catalog only.
		if name.find('.') == -1:
			# Do this to ensure the primary catalog is the first
			# to get looked up when resolving column names.
			#
			# FIXME: We could achieve the same effect by storing
			# self.catalogs in OrderedDict; alas, this would
			# break Python 2.6 compatibility.
			cats = [ (self.primary_catalog, self.catalogs[self.primary_catalog]) ]
			cats.extend(( v for v in self.catalogs.iteritems() if v[0] != self.primary_catalog ))
			colname = name
		else:
			# Force lookup of a specific catalog
			(catname, colname) = name.split('.')
			cats = [ (catname, self.catalogs[catname]) ]
		for (catname, v) in cats:
			cat = v['cat']
			colname = cat.resolve_alias(colname)
			for (table, schema) in cat.tables.iteritems():
				columns = set(( name for name, _ in schema['columns'] ))
				if colname in columns:
					self[name] = self.load_column(colname, table, catname)
					#print "Loaded column %s.%s.%s for %s (len=%s)" % (catname, table, colname2, name, len(self.columns[name]))
					return self.columns[name]

		# A name of a catalog? Return a proxy object
		if name in self.catalogs:
			return CatProxy(self, name)

		# This object is unknown to us -- let it fall through, it may
		# be a global/Python variable or function
		raise KeyError(name)

	def __setitem__(self, key, val):
		if len(self.columns):
			assert len(val) == len(self.columns.values()[0]), "%s: %d != %d" % (key, len(val), len(self.columns.values()[0]))

		self.columns[key] = val

class CatProxy:
	coldict = None
	prefix = None

	def __init__(self, coldict, prefix):
		self.coldict = coldict
		self.prefix = prefix

	def __getattr__(self, name):
		return self.coldict[self.prefix + '.' + name]

