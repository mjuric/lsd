#!/usr/bin/env python
from catalog import Catalog
import query_parser as qp
import numpy as np
from table import Table
import bhpix
import utils
import pool2
import native
from pixelization import Pixelization

#
# The result is a J = Table() with id, idx, isnull columns for each joined catalog
# To resolve a column from table X, load its tablet and do rows[J['X.idx']]
#
# Table should contain:
#	- idx to subtable rows, isnull for subtable rows, for each joined subtable
#	- the primary key against which to join a table one level up in the hierarchy
#

class TableCache:
	cache = {}		# Cache of loaded tables, in the form of cache[cell_id][include_cached][catname][tablename] = rows

	root = None		# The root catalog (Catalog instance). Used for figuring out if _not_ to load the cached rows.
	include_cached = False	# Should we load the cached object from the root catalog

	def __init__(self, root, include_cached = False):
		self.cache = {}
		self.root = root
		self.include_cached = include_cached

	def load_column(self, cell_id, name, table, cat):
		# Return the column 'name' from table 'table' of the catalog 'cat'.
		# Load the tablet if necessary, and cache it for further reuse.
		#
		# NOTE: This method DOES NOT resolve blobrefs to BLOBs
		include_cached = self.include_cached if cat.name == self.root.name else True

		# Create self.cache[cell_id][cat.name] hierarchy if needed
		if  cell_id not in self.cache:
			self.cache[cell_id] = {}
		if cat.name not in self.cache[cell_id]:
			self.cache[cell_id][cat.name] = {}
		if include_cached not in self.cache[cell_id][cat.name]:		# This bit is to support (in the future) self-joins. Note that (yes) this can be implemented in a much smarter way.
			self.cache[cell_id][cat.name][include_cached] = {}

		# See if we have already loaded the required tablet
		tcache = self.cache[cell_id][cat.name][include_cached]
		if table in tcache:
			col = tcache[table][name]
		else:
			# Load and cache the tablet
			rows = cat.fetch_tablet(cell_id, table, include_cached=include_cached)
			tcache[table] = rows

			col = rows[name]

		return col

	def resolve_blobs(self, cell_id, col, name, table, cat):
		# Resolve blobs (if blob column). NOTE: the resolved blobs
		# will not be cached.
		schema = cat._get_schema(table)
		if 'blobs' in schema and name in schema['blobs']:
			assert col.dtype == np.int64, "Data structure error: blob reference columns must be of int64 type"

			include_cached = self.include_cached if cat.name == self.root.name else True
			col = cat.fetch_blobs(cell_id, table, column=name, refs=col, include_cached=include_cached)

		return col

class CatalogEntry:
	cat      = None	# Catalog instance
	name     = None # Catalog name, as named in the query (may be different from cat.name, if 'cat AS other' construct was used)
	relation = None	# Relation to use to join with the parent
	joins    = None	# List of JoinEntries

	def __init__(self, cat, name):
		self.cat   = cat
		self.name  = name if name is not None else name
		self.joins = []

	def get_cells(self, bounds):
		""" Get populated cells of catalog self.cat that overlap
			the requested bounds. Do it recursively.
			
			The bounds can be either a list of (Polygon, intervalset) tuples,
			or None.
			
			For a static cell_id, a dynamic catalog will return all
			temporal cells within the same spatial cell. For a dynamic
			cell_id, a static catalog will return the overlapping
			static cell_id.
		"""

		# Fetch our populated cells
		pix = self.cat.pix
		cells = self.cat.get_cells(bounds, return_bounds=True)

		# Fetch the children's populated cells
		for ce in self.joins:
			cc, op = ce.get_cells(bounds)
			if   op == 'and':
				ret = dict()
				for cell_id, cbounds in cc.iteritems():
					assert cell_id not in cells or cells[cell_id] == cbounds # TODO: Debugging -- make sure the timespace constraints are the same
					if cell_id in cells:
						ret[cell_id] = cbounds
					else:
						static_cell = pix.static_cell_for_cell(cell_id)
						if static_cell in cells:
							ret[static_cell] = cells[static_cell]
							ret[cell_id] = cbounds
			elif op == 'or':
				ret = cells
				for cell_id, cbounds in cc.iteritems():
					assert cell_id not in cells or cells[cell_id] == cbounds # Debugging -- make sure the timespace constraints are the same
					ret[cell_id] = cbounds
			cells = ret

		if self.relation is None:
			# Remove all static cells if there's even a single temporal cell
			cells2 = dict(( v for v in cells.iteritems() if pix.is_temporal_cell(v[0]) ))
			return cells2 if cells2 else cells
		else:
			return cells, self.relation.join_op()

	def evaluate_join(self, cell_id, bounds, tcache, r = None, key = None):
		""" Constructs a JOIN index array.

		    This is a table consisting of the following columns:
		    	_ID : the root catalog primary key of each row

		    	<catname> : the index array that will evaluate_join 
				    the JOIN result given a column from the
				    catalog 'catname', when applied as:
				    
				    	res = col[<catname>]

		    	<catname>.isnull : a boolean array indicating whether
		    	            the <catname> index is a dummy (zero) and
		    	            actually the column has JOINed to NULL
		    	            (this only happens with outer joins)
		"""
		# Load ourselves
		id = tcache.load_column(cell_id, self.cat.get_primary_key(), self.cat.primary_table, self.cat)
		s = Table()
		mykey = '%s._ID' % self.name
		s.add_column(mykey, id)						# The key on which we'll do the joins
		s.add_column(self.name, np.arange(len(id)))	# The index into the rows of this tablet, as loaded from disk

		if self.relation is None:
			# We are root.
			r = s

			# Setup spatial bounds filter
			if bounds is not None:
				ra, dec = self.cat.get_spatial_keys()
				lon = tcache.load_column(cell_id,  ra, self.cat.primary_table, self.cat)
				lat = tcache.load_column(cell_id, dec, self.cat.primary_table, self.cat)

				r = self.filter_space(r, lon, lat, bounds)	# Note: this will add the _INBOUNDS column to r
		else:
			# We're a child. Join with the parent catalog
			idx1, idx2, isnull = self.relation.join(cell_id, r[key], s[mykey], tcache)

			# Perform the joins
			r = r[idx1]
			s = s[idx2]
			r.add_columns(s.items())

			# Add the NULL column only if there were any NULLs
			if isnull.any():
				r.add_column("%s._NULL" % self.name, isnull)

			# Perform spacetime cuts, if we have a time column
			if bounds is not None:
				tk = self.cat.get_temporal_key()
				if tk is not None:
					t = tcache.load_column(cell_id,  tk, self.cat.primary_table, self.cat)
					r = self.filter_spacetime(r, t[idx2], bounds)

		# Let children JOIN themselves onto us
		for ce in self.joins:
			r = ce.evaluate_join(cell_id, bounds, tcache, r, mykey)

		# Cleanup: once we've joined with the parent and all children,
		# no need to keep the primary key around
		r.drop_column(mykey)

		# Cleanup: if we are root, remove the _INBOUNDS helper column
		if self.relation is None and bounds is not None:
			r.drop_column('_INBOUNDS')

		return r

	def filter_space(self, r, lon, lat, bounds):
		# _INBOUNDS is a cache of spatial bounds hits, so we can
		# avoid repeated (expensive) Polygon.isInside* calls in
		# filter_time()
		inbounds = np.ones((len(r), len(bounds)), dtype=np.bool)
		r.add_column('_INBOUNDS', inbounds) # inbounds[j, i] is true if the object in row j falls within bounds[i]

		x, y = None, None
		for (i, (bounds_xy, _)) in enumerate(bounds):
			if bounds_xy is not None:
				if x is None:
					(x, y) = bhpix.proj_bhealpix(lon, lat)

				inbounds[:, i] &= bounds_xy.isInsideV(x, y)

		# Keep those that fell within at least one of the bounds present in the bounds set
		# (and thus may appear in the final result, depending on time cuts later on)
		in_  = np.any(inbounds, axis=1)
		if not in_.all():
			r = r[in_]
		return r

	def filter_spacetime(self, r, t, bounds):
		# Cull on time (for all)
		inbounds = r['_INBOUNDS']

		# This essentially looks for at least one bound specification that contains a given row
		in_ = np.zeros(len(inbounds), dtype=np.bool)
		for (i, (_, bounds_t)) in enumerate(bounds):
			if bounds_t is not None:
				in_t = bounds_t.isInside(t)
				in_ |= inbounds[:, i] & in_t
			else:
				in_ |= inbounds[:, i]

		if not in_.all():
			r = r[in_]
		return r

	def _str_tree(self, level):
		s = '    '*level + '\-- ' + self.name
		if self.relation is not None:
			s += '(%s)' % self.relation
		s += '\n'
		for e in self.joins:
			s += e._str_tree(level+1)
		return s

	def __str__(self):
		return self._str_tree(0)

class JoinRelation:
	def join(self, cell_id, id1, id2, tcache):	# Returns idx1, idx2, isnull
		raise NotImplementedError('You must override this method from a derived class')
	def join_op(self):	# Returns 'and' if the relation has an inner join-like effect, and 'or' otherwise
		raise NotImplementedError('You must override this method from a derived class')

class IndirectJoin(JoinRelation):
	m1_colspec = None	# (cat, table, column) tuple giving the location of m1
	m2_colspec = None	# (cat, table, column) tuple giving the location of m2
	jointype   = 'inner'

	def join_op(self):
		return 'and' if self.jointype == 'inner' else 'or'

	def fetch_join_map(self, cell_id, m1_colspec, m2_colspec, tcache):
		"""
			Return a list of crossmatches corresponding to ids
		"""
		cat1, table_from, column_from = m1_colspec
		cat2, table_to  , column_to   = m2_colspec
		
		# This allows joins from static to temporal catalogs where
		# the join table is in a static cell (not implemented yet). 
		# However, it also support an (implemented) case of tripple
		# joins of the form static-static-temporal, where the cell_id
		# will be temporal even when fetching the static-static join
		# table for the two other catalogs.
		cell_id_from = cat1.static_if_no_temporal(cell_id, table_from)
		cell_id_to   = cat2.static_if_no_temporal(cell_id, table_to)

		if    not cat1.tablet_exists(cell_id_from, table_from) \
		   or not cat2.tablet_exists(cell_id_to,   table_to):
			return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)

		m1 = tcache.load_column(cell_id_from, column_from, table_from, cat1)
		m2 = tcache.load_column(cell_id_to  , column_to  , table_to  , cat2)
		assert len(m1) == len(m2)

		return m1, m2

	def join(self, cell_id, id1, id2, tcache):
		"""
		    Perform a JOIN on id1, id2
		"""
		(m1, m2) = self.fetch_join_map(cell_id, self.m1_colspec, self.m2_colspec, tcache)
		return native.table_join(id1, id2, m1, m2, self.jointype)

	def __init__(self, cat, jointype, joindef):
		table_from  = joindef['table_from']
		table_to    = joindef['table_to']
		column_to   = joindef['column_to']
		column_from = joindef['column_from']

		self.m1_colspec = (cat, table_from, column_from)
		self.m2_colspec = (cat, table_to  , column_to  )
		self.jointype = jointype

	def __str__(self):
		return "%s via [%s.%s.%s, %s.%s.%s]" % (
			self.jointype,
			self.m1_colspec[0].name, self.m1_colspec[1], self.m1_colspec[2],
			self.m2_colspec[0].name, self.m2_colspec[1], self.m2_colspec[2]
		)

def construct_join_tree(from_clause):
	cats = {}

	# Instantiate the catalogs
	for catname, catpath, jointype in from_clause:
		cats[catname] = ( CatalogEntry(Catalog(catpath), catname), jointype )

	# Discover and set up links
	for (catname, (e, ejtype)) in cats.iteritems():
		# Check for catalogs that this one can be join onto (where this one is on the left hand side of the relation)
		for jcatname, jdef in e.cat.joined_catalogs.iteritems():
			if jcatname not in cats: continue

			# RHS catalog
			je, jtype = cats[jcatname]
			e.joins.append(je)

			assert je.relation is None
			je.relation = IndirectJoin(e.cat, jtype, jdef)

		# Check for catalogs that can be joined onto this one (where this one is on the right hand side of the relation)
		# For example, this can be used to JOIN the user's personal tables, when the future INTO <cat> feature gets
		# implemented
		if getattr(e.cat, 'joined_from_catalogs', None) is None:
			continue
		for jcatname, jdef in e.cat.joined_from_catalogs.iteritems():
			if jcatname not in cats: continue

			je, jtype = cats[jcatname]
			je.joins.append(e)

			assert e.relation is None
			e.relation = IndirectJoin(je.cat, ejtype, jdef)

	# Discover the root (the one and only one catalog that has no links pointing to it)
	root = None
	for _, (e, jtype) in cats.iteritems():
		if e.relation is None:
			assert root is None	# Can't have more than one roots
			assert jtype == 'inner'	# Can't have something like 'ps1_obj(outer)' on root catalog
			root = e
	assert root is not None			# Can't have zero roots

	# TODO: Verify all catalogs are reachable from the root (== there are no circular JOINs)

	# Return the root of the tree, and a dict of all the Catalog instances
	return root, dict((  (catname, cat) for (catname, (cat, _)) in cats.iteritems() ))

class iarray(np.ndarray):
	""" Subclass of ndarray allowing per-row indexing.
	
	    Used by ColDict.load_column()
	"""
	def __call__(self, *args):
		"""
		   Apply numpy indexing on a per-row basis. A rough
		   equivalent of:
			
			self[ arange(len(self)) , *args]

		   where any tuples in args will be converted to a
		   corresponding slice, while integers and numpy
		   arrays will be passed in as-given. Any numpy array
		   given as index must be of len(self).

		   Simple example: assuming we have a chip_temp column
		   that was defined with 64f8, to select the temperature
		   of the chip corresponding to the observation, do:
		   
		   	chip_temp(chip_id)

		   Note: A tuple of the form (x,y,z) is will be
		   conveted to [x:y:z] slice. (x,y) converts to [x:y:]
		   and (x,) converts to [x::]
		"""
		# Note: numpy multidimensional indexing is mind numbing...
		# The stuff below works for 1D arrays, I don't guarantee it for higher dimensions.
		#       See: http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
		idx = [ (slice(*arg) if len(arg) > 1 else np.s_[arg[0]:]) if isinstance(arg, tuple) else arg for arg in args ]

		firstidx = np.arange(len(self)) if len(self) else np.s_[:]
		idx = tuple([ firstidx ] + idx)

		return self.__getitem__(idx)

class TableColsProxy:
	catalogs = None
	root_catalog = None

	def __init__(self, root_catalog, catalogs):
		self.root_catalog = root_catalog
		self.catalogs = catalogs

	def __getitem__(self, catname):
		# Return a list of columns in catalog catname
		if catname == '':
			catname = self.root_catalog
		return self.catalogs[catname].cat.all_columns()

class QueryInstance(object):
	# Internal working state variables
	tcache   = None		# TableCache() instance
	columns  = None		# Cache of already referenced and evaluated columns (dict)
	cell_id  = None		# cell_id on which we're operating
	jmap 	 = None		# index map used to materialize the JOINs

	# These will be filled in from a QueryEngine instance
	catalogs = None		# A name:Catalog dict() with all the catalogs listed on the FROM line.
	root	 = None		# CatalogEntry instance with the primary (root) catalog
	query_clauses = None	# Tuple with parsed query clauses

	def __init__(self, q, cell_id, bounds, include_cached):
		self.catalogs	   = q.catalogs
		self.root	   = q.root
		self.query_clauses = q.query_clauses

		self.cell_id	= cell_id
		self.bounds	= bounds

		self.tcache	= TableCache(self.root, include_cached)
		self.columns	= {}

	def __iter__(self):
		return self.execute()

	def execute(self):
		(select_clause, where_clause, _) = self.query_clauses

		# Evaluate the JOIN map
		self.jmap   	    = self.root.evaluate_join(self.cell_id, self.bounds, self.tcache)

		# eval individual columns in select clause to slurp them up from disk
		# and have them ready for the WHERE clause
		#
		# TODO: We can optimize this by evaluating WHERE first, using the result
		#       to cull the output number of rows, and then evaluating the columns.
		#		When doing so, care must be taken to fall back onto evaluating a
		#		column from the SELECT clause, if it's referenced in WHERE
		retcols = []
		nrows = None
		global_ = globals()
		for (asname, name) in select_clause:
			col = self[name]	# For debugging
#			col = eval(name, global_, self)
#			exit()

			self[asname] = col
			retcols.append(asname)

			if nrows != None:
				assert nrows == len(col)
			nrows = len(col)

		# evaluate the WHERE clause, to obtain the final filter
		in_    = np.empty(nrows, dtype=bool)
		in_[:] = eval(where_clause, global_, self)

		if nrows == 0 or in_.all():
			# We need to handle the nrows=0 case separately, because of multidimensional columns
			self.rows_ = Table( [ (name, self[name]) for name in retcols ] )
		else:
			self.rows_ = Table( [ (name, self[name][in_]) for name in retcols ] )

		yield self.rows_

	#################

	def load_column(self, name, table, catname):
		# Load the column from table 'table' of the catalog 'catname'
		# Also cache the loaded tablet, for future reuse
		cat = self.catalogs[catname].cat

		# Load the column (via cache)
		col = self.tcache.load_column(self.cell_id, name, table, cat)

		# Join
		if len(col):
			idx			= self.jmap[catname]
			col         = col[idx]

			isnullKey  = catname + '._NULL'
			if isnullKey in self.jmap:
				isnull		= self.jmap[isnullKey]
				col[isnull] = cat.NULL

		# Resolve blobs (if a blobref column)
		col = self.tcache.resolve_blobs(self.cell_id, col, name, table, cat)

		# Return the column as an iarray
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
			cats = [ (self.root.name, self.catalogs[self.root.name]) ]
			cats.extend(( (name, cat) for name, cat in self.catalogs.iteritems() if name != self.root.name ))
			colname = name
		else:
			# Force lookup of a specific catalog
			(catname, colname) = name.split('.')
			cats = [ (catname, self.catalogs[catname]) ]
		for (catname, e) in cats:
			cat = e.cat
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

class QueryEngine(object):
	catalogs = None		# A name:Catalog dict() with all the catalogs listed on the FROM line.
	root	 = None		# CatalogEntry instance with the primary (root) catalog
	query_clauses  = None	# Parsed query clauses

	def __init__(self, dbdir, query):
		# parse query
		(select_clause, where_clause, from_clause) = qp.parse(query)

		# patch paths in from_clause
		if dbdir is not None:
			from_clause = [ (name, '%s/%s' % (dbdir, path), jointype) for name, path, jointype in from_clause ]

		self.root, self.catalogs = construct_join_tree(from_clause);
		select_clause = qp.resolve_wildcards(select_clause, TableColsProxy(self.root.name, self.catalogs))
		self.query_clauses = (select_clause, where_clause, from_clause)

	def on_cell(self, cell_id, bounds=None, include_cached=False):
		return QueryInstance(self, cell_id, bounds, include_cached)

class Query(object):
	def __init__(self, dbdir, query):
		self.qengine		= QueryEngine(dbdir, query)

	def execute(self, kernels, bounds=None, include_cached=False, testbounds=True, nworkers=None, progress_callback=None):
		""" A MapReduce implementation, where rows from individual cells
		    get mapped by the mapper, with the result reduced by the reducer.
		    
		    Mapper, reducer, and all *_args must be pickleable.
		    
		    The mapper must be a callable expecting at least one argument, 'rows'.
		    'rows' is always the first argument; if any extra arguments are passed 
		    via mapper_args, they will come after it.
		    'rows' will be a numpy array of table records (with named columns)

		    The mapper must return a sequence of key-value pairs. All key-value
		    pairs will be merged by key into (key, [values..]) pairs that shall
		    be passed to the reducer.

		    The reducer must expect two parameters, the first being the key
		    and the second being a sequence of all values that the mappers
		    returned for that key. The return value of the reducer is passed back
		    to the user and is user-defined.
   
		    If the reducer is None, only the mapping step is performed and the
		    return value of the mapper is passed to the user.
		"""
		partspecs = self.qengine.root.get_cells(bounds)

		# tell _mapper not to test spacetime boundaries if the user requested so
		if not testbounds:
			partspecs = [ (part_id, None) for (part_id, _) in partspecs ]

		# Insert our mapper into the kernel chain
		kernels = list(kernels)
		kernels[0] = (_mapper, kernels[0], self.qengine, include_cached)

		# start and run the workers
		pool = pool2.Pool(nworkers)
		for result in pool.map_reduce_chain(partspecs.items(), kernels, progress_callback=progress_callback):
			yield result

	def iterate(self, bounds=None, include_cached=False, return_blocks=False, filter=None, testbounds=True, nworkers=None, progress_callback=None):
		""" Yield rows (either on a row-by-row basis if return_blocks==False
		    or in chunks (numpy structured array)) within a
		    given footprint. Calls 'filter' callable (if given) to filter
		    the returned rows.

		    See the documentation for Catalog.fetch for discussion of
		    'filter' callable.
		"""

		mapper = filter if filter is not None else _iterate_mapper

		for rows in self.execute(
								[mapper], bounds, include_cached,
								testbounds=testbounds, nworkers=nworkers, progress_callback=progress_callback):
			if return_blocks:
				yield rows
			else:
				for row in rows:
					yield row

	def fetch(self, bounds=None, include_cached=False, return_blocks=False, filter=None, testbounds=True, nworkers=None, progress_callback=None):
		""" Return a table (numpy structured array) of all rows within a
		    given footprint. Calls 'filter' callable (if given) to filter
		    the returned rows. Returns None if there are no rows to return.

		    The 'filter' callable should expect a single argument, rows,
		    being the set of rows (numpy structured array) to filter. It
		    must return the set of filtered rows (also as numpy structured
		    array). E.g., identity filter function would be:
		    
		    	def identity(rows):
		    		return rows

		    while a function filtering on column 'r' may look like:
		    
		    	def r_filter(rows):
		    		return rows[rows['r'] < 21.5]
		   
		    The filter callable must be piclkeable. Extra arguments to
		    filter may be given in 'filter_args'
		"""

		ret = None
		for rows in self.iterate(bounds, include_cached, return_blocks=True, filter=filter, testbounds=testbounds, nworkers=nworkers, progress_callback=progress_callback):
			# ensure enough memory has been allocated (and do it
			# intelligently if not)
			if ret is None:
				ret = rows
				nret = len(rows)
			else:
				while len(ret) < nret + len(rows):
					ret.resize(2 * max(len(ret),1))

				# append
				ret[nret:nret+len(rows)] = rows
				nret = nret + len(rows)

		ret.resize(nret)
		return ret

###############################################################
# Aux. functions implementing Query.iterate() and
# Query.fetch()
def _iterate_mapper(qresult):
	for rows in qresult:
		if len(rows):	# Don't return empty sets
			yield rows

class CatProxy:
	coldict = None
	prefix = None

	def __init__(self, coldict, prefix):
		self.coldict = coldict
		self.prefix = prefix

	def __getattr__(self, name):
		return self.coldict[self.prefix + '.' + name]

def _mapper(partspec, mapper, qengine, include_cached, _pass_empty=False):
	(cell_id, bounds) = partspec
	mapper, mapper_args = utils.unpack_callable(mapper)

	# Pass on to mapper (and yield its results)
	for result in mapper(qengine.on_cell(cell_id, bounds, include_cached), *mapper_args):
		yield result

def test_kernel(qresult):
	for rows in qresult:
		yield qresult.cell_id, len(rows)

if __name__ == "__main__":
	def test():
		import footprint as foot
		query = "_ID, ps1_det._ID, ps1_exp._ID, sdss._ID FROM '../../../ps1/sdss'(outer) as sdss, '../../../ps1/ps1_obj' as ps1_obj, '../../../ps1/ps1_det' as ps1_det, '../../../ps1/ps1_exp'(outer) as ps1_exp WHERE True"
		query = "_ID, ps1_det._ID, ps1_exp._ID, sdss._ID FROM sdss(outer), ps1_obj, ps1_det, ps1_exp(outer)"
		cell_id = np.uint64(6496868600547115008 + 0xFFFFFFFF)
		include_cached = False
		bounds = [(foot.rectangle(120, 20, 160, 60), None)]
		bounds = None
#		for rows in QueryEngine(query).on_cell(cell_id, bounds, include_cached):
#			pass;

#		dq = Query(query)
#		for res in dq.execute([test_kernel], bounds, include_cached, nworkers=1):
#			print res

		dq = Query('../../../ps1', query)
#		for res in dq.iterate(bounds, include_cached, nworkers=1):
#			print res

		res = dq.fetch(bounds, include_cached, nworkers=1)
		print len(res)

	test()
