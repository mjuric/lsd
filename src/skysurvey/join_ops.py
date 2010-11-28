#!/usr/bin/env python
from catalog import Catalog
import query_parser as qp
import numpy as np
from table import Table
import bhpix
import utils
import pool2
import native
import os, json, glob
import sys
import copy
import urlparse
from contextlib import contextmanager

#
# The result is a J = Table() with id, idx, isnull columns for each joined catalog
# To resolve a column from table X, load its tablet and do rows[J['X.idx']]
#
# Table should contain:
#	- idx to subtable rows, isnull for subtable rows, for each joined subtable
#	- the primary key against which to join a table one level up in the hierarchy
#

class TableCache:
	""" An object caching loaded tablets.
		TODO: Perhaps merge it with DB? Or make it a global?
	"""
	cache = {}		# Cache of loaded tables, in the form of cache[cell_id][include_cached][catname][tablename] = rows

	root = None		# The root catalog (Catalog instance). Used for figuring out if _not_ to load the cached rows.
	include_cached = False	# Should we load the cached object from the root catalog

	def __init__(self, root, include_cached = False):
		self.cache = {}
		self.root = root
		self.include_cached = include_cached

	def load_column(self, cell_id, name, cat):
		# Return the column 'name' from table 'table' of the catalog 'cat'.
		# Load the tablet if necessary, and cache it for further reuse.
		#
		# NOTE: This method DOES NOT resolve blobrefs to BLOBs
		include_cached = self.include_cached if cat.name == self.root.name else True

		# Figure out which table contains this column
		table = cat.columns[name].table

		# Create self.cache[cell_id][cat.name] hierarchy if needed
		if  cell_id not in self.cache:
			self.cache[cell_id] = {}
		if cat.name not in self.cache[cell_id]:
			self.cache[cell_id][cat.name] = {}
		if include_cached not in self.cache[cell_id][cat.name]:		# This bit is to support (in the future) self-joins. Note that (yes) this can be implemented in a much smarter way.
			self.cache[cell_id][cat.name][include_cached] = {}

		# See if we have already loaded the required tablet
		tcache = self.cache[cell_id][cat.name][include_cached]
		if table not in tcache:
			# Load and cache the tablet
			rows = cat.fetch_tablet(cell_id, table, include_cached=include_cached)

			# Ensure it's as long as the primary table
			if table != cat.primary_table:
				nrows = len(self.load_column(cell_id, cat.primary_key.name, cat))
				rows.resize(nrows)

			tcache[table] = rows
		else:
			rows = tcache[table]

		col = rows[name]
		return col

	def resolve_blobs(self, cell_id, col, name, cat):
		# Resolve blobs (if blob column). NOTE: the resolved blobs
		# will not be cached.

		if cat.columns[name].is_blob:
			include_cached = self.include_cached if cat.name == self.root.name else True
			col = cat.fetch_blobs(cell_id, column=name, refs=col, include_cached=include_cached)

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
					if not (cell_id not in cells or cells[cell_id] == cbounds): # TODO: Debugging -- make sure the timespace constraints are the same
						aa = cells[cell_id]
						bb = cbounds
						pass
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
					if not (cell_id not in cells or cells[cell_id] == cbounds): # TODO: Debugging -- make sure the timespace constraints are the same
						aa = cells[cell_id]
						bb = cbounds
						pass
					assert cell_id not in cells or cells[cell_id] == cbounds # Debugging -- make sure the timespace constraints are the same
					ret[cell_id] = cbounds
			cells = ret

		if self.relation is None:
			# Remove all static cells if there's even a single temporal cell
			cells2 = dict(( v for v in cells.iteritems() if pix.is_temporal_cell(v[0]) ))
			cells = cells2 if cells2 else cells
			return cells
		else:
			return cells, self.relation.join_op()

	def evaluate_join(self, cell_id, bounds, tcache, r = None, key = None, idxkey = None):
		""" Constructs a JOIN index array.

			* If the result of the join is no rows, return None

		    * If the result is not empty, the return is a Table() instance
		      that _CAN_ (but DOESN'T HAVE TO; see below) have the 
		      following columns:

		    	<catname> : the index array that will materialize
				    the JOIN result when applied to the corresponding
				    catalog's column as:
				    
				    	res = col[<catname>]

		    	<catname>._NULL : a boolean array indicating whether
		    	            the <catname> index is a dummy (zero) and
		    	            actually the column has JOINed to NULL
		    	            (this only happens with outer joins)

		      -- If the result of a JOIN for a particular catalog leaves
		         no NULLs, there'll be no <catname>._NULL column.
		      -- If the col[<catname>] would be == col, there'll be no
		         <catname> column.
		"""
		hasBounds = bounds != [(None, None)]
		if len(bounds) > 1:
			pass;
		# Skip everything if this is a single-table read with no bounds
		# ("you don't pay for what you don't use")
		if self.relation is None and not hasBounds and not self.joins:
			return Table()

		# Load ourselves
		id = tcache.load_column(cell_id, self.cat.get_primary_key(), self.cat)
		s = Table()
		mykey = '%s._ID' % self.name

		s.add_column(mykey, id)						# The key on which we'll do the joins
		s.add_column(self.name, np.arange(len(id)))	# The index into the rows of this tablet, as loaded from disk

		if self.relation is None:
			# We are root.
			r = s

			# Setup spatial bounds filter
			if hasBounds:
				ra, dec = self.cat.get_spatial_keys()
				lon = tcache.load_column(cell_id,  ra, self.cat)
				lat = tcache.load_column(cell_id, dec, self.cat)

				r = self.filter_space(r, lon, lat, bounds)	# Note: this will add the _INBOUNDS column to r
		else:
			# We're a child. Join with the parent catalog
			idx1, idx2, isnull = self.relation.join(cell_id, r[key], s[mykey], r[idxkey], s[self.name], tcache)

			# Handle the special case of OUTER JOINed empty 's'
			if len(s) == 0 and len(idx2) != 0:
				assert isnull.all()
				s = Table(dtype=s.dtype, size=1)

			# Perform the joins
			r = r[idx1]
			s = s[idx2]
			r.add_columns(s.items())

			# Add the NULL column only if there were any NULLs
			if isnull.any():
				r.add_column("%s._NULL" % self.name, isnull)

		# Perform spacetime cuts, if we have a time column
		# (and the JOIN didn't result in all NULLs)
		if hasBounds:
			tk = self.cat.get_temporal_key()
			if tk is not None:
				t = tcache.load_column(cell_id,  tk, self.cat)
				if len(t):
					r = self.filter_spacetime(r, t[idx2], bounds)

		# Let children JOIN themselves onto us
		for ce in self.joins:
			r = ce.evaluate_join(cell_id, bounds, tcache, r, mykey, self.name)

		# Cleanup: once we've joined with the parent and all children,
		# no need to keep the primary key around
		r.drop_column(mykey)

		if self.relation is None:
			# Return None if the query yielded no rows
			if len(r) == 0:
				return None
			# Cleanup: if we are root, remove the _INBOUNDS helper column
			if '_INBOUNDS' in r:
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
	kind   = 'inner'
	def __init__(self, db, catR, catS, kind, **joindef):
		self.db   = db
		self.kind = kind
		self.catR = catR
		self.catS = catS

	def join_op(self):	# Returns 'and' if the relation has an inner join-like effect, and 'or' otherwise
		return 'and' if self.kind == 'inner' else 'or'

	def join(self, cell_id, id1, id2, idx1, idx2, tcache):	# Returns idx1, idx2, isnull
		raise NotImplementedError('You must override this method from a derived class')

class IndirectJoin(JoinRelation):
	m1_colspec = None	# (cat, table, column) tuple giving the location of m1
	m2_colspec = None	# (cat, table, column) tuple giving the location of m2

	def fetch_join_map(self, cell_id, m1_colspec, m2_colspec, tcache):
		"""
			Return a list of crossmatches corresponding to ids
		"""
		cat1, column_from = m1_colspec
		cat2, column_to   = m2_colspec
		
		# This allows joins from static to temporal catalogs where
		# the join table is in a static cell (not implemented yet). 
		# However, it also support an (implemented) case of tripple
		# joins of the form static-static-temporal, where the cell_id
		# will be temporal even when fetching the static-static join
		# table for the two other catalogs.
		cell_id_from = cat1.static_if_no_temporal(cell_id)
		cell_id_to   = cat2.static_if_no_temporal(cell_id)

		if    not cat1.tablet_exists(cell_id_from) \
		   or not cat2.tablet_exists(cell_id_to):
			return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64)

		m1 = tcache.load_column(cell_id_from, column_from, cat1)
		m2 = tcache.load_column(cell_id_to  , column_to  , cat2)
		assert len(m1) == len(m2)

		return m1, m2

	def join(self, cell_id, id1, id2, idx1, idx2, tcache):
		"""
		    Perform a JOIN on id1, id2, that were obtained by
		    indexing their origin catalogs with idx1, idx2.
		"""
		(m1, m2) = self.fetch_join_map(cell_id, self.m1_colspec, self.m2_colspec, tcache)
		return native.table_join(id1, id2, m1, m2, self.kind)

	def __init__(self, db, catR, catS, kind, **joindef):
		JoinRelation.__init__(self, db, catR, catS, kind, **joindef)

		m1_catname, m1_colname = joindef['m1']
		m2_catname, m2_colname = joindef['m2']

		self.m1_colspec = (db.catalog(m1_catname), m1_colname)
		self.m2_colspec = (db.catalog(m2_catname), m2_colname)

	def __str__(self):
		return "%s indirect via [%s.%s, %s.%s]" % (
			self.kind,
			self.m1_colspec[0], self.m1_colspec[1].name,
			self.m2_colspec[0], self.m2_colspec[1].name,
		)

#class EquijoinJoin(IndirectJoin):
#	def __init__(self, db, catR, catS, kind, **joindef):
#		JoinRelation.__init__(self, db, catR, catS, kind, **joindef)
#
#		# Emulating direct join with indirect one
#		# TODO: Write a specialized direct join routine
#		self.m1_colspec = catR, joindef['id1']
#		self.m2_colspec = catS, joindef['id2']

def create_join(db, fn, jkind, catR, catS):
	data = json.loads(file(fn).read())
	assert 'type' in data

	jclass = data['type'].capitalize() + 'Join'
	return globals()[jclass](db, catR, catS, jkind, **data)

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
		# Return a list of columns in catalog catname, unless
		# they're pseudocolumns
		if catname == '':
			catname = self.root_catalog
		cat = self.catalogs[catname].cat
		return [ name for (name, coldef) in cat.columns.iteritems() if not cat._is_pseudotable(coldef.table) ]

class QueryInstance(object):
	# Internal working state variables
	tcache   = None		# TableCache() instance
	columns  = None		# Cache of already referenced and evaluated columns (dict)
	cell_id  = None		# cell_id on which we're operating
	jmap 	 = None		# index map used to materialize the JOINs
	bounds   = None

	# These will be filled in from a QueryEngine instance
	db       = None		# The controlling database instance
	catalogs = None		# A name:Catalog dict() with all the catalogs listed on the FROM line.
	root	 = None		# CatalogEntry instance with the primary (root) catalog
	query_clauses = None	# Tuple with parsed query clauses
	pix      = None         # Pixelization object (TODO: this should be moved to class DB)
	locals   = None		# Extra local variables to be made available within the query

	def __init__(self, q, cell_id, bounds, include_cached):
		self.db            = q.db
		self.catalogs	   = q.catalogs
		self.root	   = q.root
		self.query_clauses = q.query_clauses
		self.pix           = q.root.cat.pix
		self.locals        = q.locals

		self.cell_id	= cell_id
		self.bounds	= bounds

		self.tcache	= TableCache(self.root, include_cached)
		self.columns	= {}
		
	def peek(self):
		assert self.cell_id is None
		return self.eval_select()

	def __iter__(self):
		assert self.cell_id is not None # Cannot call iter when peeking

		# Evaluate the JOIN map
		self.jmap   	    = self.root.evaluate_join(self.cell_id, self.bounds, self.tcache)

		if self.jmap is not None:
			# TODO: We could optimize this by evaluating WHERE first, using the result
			#       to cull the output number of rows, and then evaluating the columns.
			#		When doing so, care must be taken to fall back onto evaluating a
			#		column from the SELECT clause, if it's referenced in WHERE

			globals_ = self.prep_globals()

			# eval individual columns in select clause to slurp them up from disk
			# and have them ready for the WHERE clause
			rows = self.eval_select(globals_)

			if len(rows):
				in_  = self.eval_where(globals_)

				if not in_.all():
					rows = rows[in_]

				# Attach metadata
				rows.cell_id = self.cell_id
				rows.where__ = in_

				yield rows

		# We yield nothing if the result set is empty.

	def prep_globals(self):
		globals_ = globals()

		# Add implicit global objects present in queries
		globals_['_PIX'] = self.root.cat.pix
		globals_['_DB']  = self.db
		
		return globals_

	def eval_where(self, globals_ = None):
		(_, where_clause, _, _) = self.query_clauses

		if globals_ is None:
			globals_ = self.prep_globals()

		# evaluate the WHERE clause, to obtain the final filter
		in_    = np.empty(len(next(self.columns.itervalues())), dtype=bool)
		in_[:] = eval(where_clause, globals_, self)

		return in_

	def eval_select(self, globals_ = None):
		(select_clause, _, _, _) = self.query_clauses

		if globals_ is None:
			globals_ = self.prep_globals()

		rows = Table()
		for (asname, name) in select_clause:
#			col = self[name]	# For debugging
			col = eval(name, globals_, self)
#			exit()

			self[asname] = col
			rows.add_column(asname, col)

		return rows

	def postprocess(self, rows):
		# This handles INTO clauses. Stores the data into
		# destination catalog, returning the IDs of stored rows.
		(_, _, _, into_clause) = self.query_clauses
		if into_clause is None:
			return rows

		# Insert into the destination catalog
		(into_cat, dtype, key) = into_clause
		cat = self.db.catalog(into_cat)
		if key is not None:
			rows.add_column('_ID', self[key][rows.where__])
			ids = cat.append(rows, _update=True)
		else:
			rows.add_column('_ID', rows.cell_id, 'u8')
			ids = cat.append(rows)

		# Return IDs only
		for col in rows.keys():
			rows.drop_column(col)
		rows.add_column('_ID', ids)
#		print "HERE", rows; exit()
		
		return rows

	#################

	def load_column(self, name, catname):
		# If we're just peeking, construct the column from schema
		if self.cell_id is None:
			assert self.bounds is None
			cdef = self.catalogs[catname].cat.columns[name]
			dtype = np.dtype(cdef.dtype)

			# Handle blobs
			if cdef.is_blob:
				col = np.empty(0, dtype=np.dtype(('O', dtype.shape)))
			else:
				col = np.empty(0, dtype=dtype)
		else:
			# Load the column from table 'table' of the catalog 'catname'
			# Also cache the loaded tablet, for future reuse
			cat = self.catalogs[catname].cat

			# Load the column (via cache)
			col = self.tcache.load_column(self.cell_id, name, cat)

			# Join/filter if needed
			if catname in self.jmap:
				isnullKey  = catname + '._NULL'
				idx			= self.jmap[catname]
				if len(col):
					col         = col[idx]
	
					if isnullKey in self.jmap:
						isnull		= self.jmap[isnullKey]
						col[isnull] = cat.NULL
				elif len(idx):
					# This tablet is empty, but columns show up here because of an OUTER join.
					# Just return NULLs of proper dtype
					assert self.jmap[isnullKey].all()
					assert (idx == 0).all()
					col = np.zeros(shape=(len(idx),) + col.shape[1:], dtype=col.dtype)

			# Resolve blobs (if a blobref column)
			col = self.tcache.resolve_blobs(self.cell_id, col, name, cat)

		# Return the column as an iarray
		col = col.view(iarray)
		return col

	def load_pseudocolumn(self, name):
		""" Generate per-query pseudocolumns.
		
		    Developer note: When adding a new pseudocol, make sure to also add
		       it to the if() statement in __getitem__
		"""
		# Detect the number of rows
		nrows = len(self['_ID'])

		if name == '_ROWNUM':
			# like Oracle's ROWNUM, but on a per-cell basis (and zero-based)
			return np.arange(nrows, dtype=np.uint64)
		elif name == '_CELLID':
			ret = np.empty(nrows, dtype=np.uint64)
			ret[:] = self.cell_id
			return ret
		else:
			raise Exception('Unknown pseudocolumn %s' % name)

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
			colname = e.cat.resolve_alias(colname)
			if colname in e.cat.columns:
				self[name] = self.load_column(colname, catname)
				#print "Loaded column %s.%s.%s for %s (len=%s)" % (catname, table, colname2, name, len(self.columns[name]))
				return self.columns[name]

		# A query pseudocolumn?
		if name in ['_ROWNUM', '_CELLID']:
			col = self[name] = self.load_pseudocolumn(name)
			return col

		# A name of a catalog? Return a proxy object
		if name in self.catalogs:
			return CatProxy(self, name)

		# Is this one of the local variables passed in by the user?
		if name in self.locals:
			return self.locals[name]

		# This object is unknown to us -- let it fall through, it may
		# be a global variable or function
		raise KeyError(name)

	def __setitem__(self, key, val):
		if len(self.columns):
			assert len(val) == len(next(self.columns.itervalues())), "%s: %d != %d" % (key, len(val), len(self.columns.values()[0]))

		self.columns[key] = val

class QueryEngine(object):
	db       = None		# The controlling database instance

	catalogs = None		# A name:Catalog dict() with all the catalogs listed on the FROM line.
	root	 = None		# CatalogEntry instance with the primary (root) catalog
	query_clauses  = None	# Parsed query clauses
	locals   = None		# Extra variables to be made local to the query

	def __init__(self, db, query, locals = {}):
		self.db = db

		# parse query
		(select_clause, where_clause, from_clause, into_clause) = qp.parse(query)

		self.root, self.catalogs = db.construct_join_tree(from_clause);
		select_clause            = qp.resolve_wildcards(select_clause, TableColsProxy(self.root.name, self.catalogs))

		self.query_clauses       = (select_clause, where_clause, from_clause, into_clause)

		self.locals = locals

	def create_into_catalog(self):
		# Called from Query.execute
		_, _, _, into_clause = self.query_clauses
		if into_clause is None:
			return

		(catname, dtype, key) = into_clause
		assert dtype is None, "User-supplied dtype not supported yet."

		db = self.db
		dtype = self.peek().dtype
		schema = { 'columns': [] }
		if db.catalog_exists(catname):
			# Must have a designated key for updating to work
			if key is None:
				raise Exception('If updating into an existing catalog (%s), you must specify the column with IDs that will be updated (" ... INTO ... AT keycol") construct).' % catname)

			cat = db.catalog(catname)

			# Find any new columns we'll need to create
			for name in dtype.names:
				rname = cat.resolve_alias(name)
				if rname not in cat.columns:
					schema['columns'].append((name, utils.str_dtype(dtype[name])))
		else:
			cat = db.catalog(catname, True)

			# Create all columns
			schema['columns'] = [ (name, utils.str_dtype(dtype[name])) for name in dtype.names ]
			schema['primary_key'] = '_id'

		# Adding columns starting with '_' is prohibited. Enforce it here
		for (col, _) in schema['columns']:
			if col[0] == '_':
				raise Exception('Storing columns starting with "_" is prohibited. Use the "col AS alias" construct to rename the offending column ("%s")' % col)

		# Add a primary key column (if needed)
		if 'primary_key' in schema and schema['primary_key'] not in dict(schema['columns']):
			schema['columns'].insert(0, (schema['primary_key'], 'u8'))

		# Create a new cgroup (if needed)
		if schema['columns']:
			for x in xrange(1, 100000):
				tname = 'auto%03d' % x
				if tname not in cat._tables:
					break
			cat.create_table(tname, schema)

		#print "XXX:", tname, schema; exit()

	def on_cell(self, cell_id, bounds=None, include_cached=False):
		return QueryInstance(self, cell_id, bounds, include_cached)

	def peek(self):
		return QueryInstance(self, None, None, None).peek()

class Query(object):
	db      = None
	qengine = None

	def __init__(self, db, query, locals = {}):
		self.db		 = db
		self.qengine = QueryEngine(db, query, locals=locals)

	def execute(self, kernels, bounds=None, include_cached=False, cells=[], testbounds=True, nworkers=None, progress_callback=None, yield_empty=False):
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
		partspecs = dict()
		# Add explicitly requested cells
		for cell_id in cells:
			partspecs[cell_id] = [(None, None)]
		# Add cells within bounds
		if len(cells) == 0 or bounds is not None:
			partspecs.update(self.qengine.root.get_cells(bounds))

		# tell _mapper not to test spacetime boundaries if the user requested so
		if not testbounds:
			partspecs = dict([ (part_id, None) for (part_id, _) in partspecs ])

		# Insert our mapper into the kernel chain
		kernels = list(kernels)
		kernels[0] = (_mapper, kernels[0], self.qengine, include_cached)

		# Create the destination for INTO queries (if needed)
		self.qengine.create_into_catalog()

		# start and run the workers
		pool = pool2.Pool(nworkers)
		yielded = False
		for result in pool.map_reduce_chain(partspecs.items(), kernels, progress_callback=progress_callback):
			yield result
			yielded = True

		# Yield an empty row, if requested
		if not yielded and yield_empty:
			yield self.qengine.peek()

	def iterate(self, bounds=None, include_cached=False, cells=[], return_blocks=False, filter=None, testbounds=True, nworkers=None, progress_callback=None, yield_empty=False):
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
				cells=cells, testbounds=testbounds, nworkers=nworkers, progress_callback=progress_callback,
				yield_empty=yield_empty):
			if return_blocks:
				yield rows
			else:
				for row in rows:
					yield row

	def fetch(self, bounds=None, include_cached=False, cells=[], filter=None, testbounds=True, nworkers=None, progress_callback=None):
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
		for rows in self.iterate(
				bounds, include_cached,
				cells=cells, return_blocks=True, filter=filter, testbounds=testbounds,
				yield_empty=True,
				nworkers=nworkers, progress_callback=progress_callback):
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

		if ret is not None:
			ret.resize(nret)

		return ret

	def fetch_cell(self, cell_id, include_cached=False):
		""" Execute the query on a given (single) cell.

		    Does not launch extra workers, nor shows the
		    progress bar.
		"""
		return self.fetch(cells=[cell_id], include_cached=include_cached, nworkers=1, progress_callback=pool2.progress_pass)

class DB(object):
	path = None		# Root of the database, where all catalogs reside

	def __init__(self, path):
		self.path = path
		
		if not os.path.isdir(path):
			raise Exception('"%s" is not an acessible directory.' % (path))

		self.catalogs = dict()	# A cache of catalog instances

	def resolve_uri(self, uri):
		""" Resolve the given URI into something that can be
		    passed on to file() for loading.
		"""
		if uri[:4] != 'lsd:':
			return uri

		_, catname, _ = uri.split(':', 2)
		return catalog(catname).resolve_uri(uri)

	@contextmanager
	def open_uri(self, uri, mode='r', clobber=True):
		if uri[:4] != 'lsd:':
			# Interpret this as a general URL
			import urllib

			f = urllib.urlopen(uri)
			yield f

			f.close()
		else:
			# Pass on to the controlling catalog
			_, catname, _ = uri.split(':', 2)
			with self.catalog(catname).open_uri(uri, mode, clobber) as f:
				yield f

	def query(self, query, locals={}):
		return Query(self, query, locals=locals)

	def _aux_create_table(self, cat, tname, schema):
		schema = copy.deepcopy(schema)

		# Remove any extra fields off schema['column']
		schema['columns'] = [ v[:2] for v in schema['columns'] ]

		cat.create_table(tname, schema)

	def create_catalog(self, catname, catdef):
		"""
			Creates the catalog given the extended schema description.
		"""
		cat = self.catalog(catname, create=True)

		# Add fgroups
		if 'fgroups' in catdef:
			for fgroup, fgroupdef in catdef['fgroups'].iteritems():
				cat.define_fgroup(fgroup, fgroupdef)

		# Add filters
		cat.set_default_filters(**catdef.get('filters', {}))

		# Add column groups
		tschema = catdef['schema']

		# Ensure we create the primary table first
		schemas = []
		for tname, schema in tschema.iteritems():
			schemas.insert(np.where('primary_key' in schema, 0, len(schemas)), (tname, schema))
		assert len(schemas) and 'primary_key' in schemas[0][1]
		for tname, schema in schemas:
			self._aux_create_table(cat, tname, schema)

		# Add aliases (must do this last, as the aliased columns have to exist)
		if 'aliases' in catdef:
			for alias, colname in catdef['aliases'].iteritems():
				cat.define_alias(alias, colname)

		return cat

	def define_join(self, name, type, **joindef):
		#- .join file structure:
		#	- indirect joins:			Example: ps1_obj:ps1_det.join
		#		type:	indirect		"type": "indirect"
		#		m1:	(cat1, col1)		"m1:":	["ps1_obj2det", "id1"]
		#		m2:	(cat2, col2)		"m2:":	["ps1_obj2det", "id2"]
		#	- equijoins:				Example: ps1_det:ps1_exp.join		(!!!NOT IMPLEMENTED!!!)
		#		type:	equi			"type": "equijoin"
		#		id1:	colA			"id1":	"exp_id"
		#		id2:	colB			"id2":	"exp_id"
		#	- direct joins:				Example: ps1_obj:ps1_calib.join.json	(!!!NOT IMPLEMENTED!!!)
		#		type:	direct			"type": "direct"

		fname = '%s/%s.join' % (self.path, name)
		if os.access(fname, os.F_OK):
			raise Exception('Join relation %s already exist (in file %s)' % (name, fname))

		joindef['type'] = type
	
		f = open(fname, 'w')
		f.write(json.dumps(joindef, indent=4, sort_keys=True))
		f.close()

	def define_default_join(self, obj_catdir, o2d_catdir, type, **joindef):
		return self.define_join('.%s:%s' % (obj_catdir, o2d_catdir), type, **joindef)

	def catalog_exists(self, catname):
		try:
			self.catalog(catname)
			return True
		except IOError:
			return False

	def catalog(self, catname, create=False):
		""" Given the catalog name, returns a Catalog object.
		"""
		if catname not in self.catalogs:
			catpath = '%s/%s' % (self.path, catname)
			if not create:
				self.catalogs[catname] = Catalog(catpath)
			else:
				self.catalogs[catname] = Catalog(catpath, name=catname, mode='c')
		else:
			assert not create

		return self.catalogs[catname]

	def construct_join_tree(self, from_clause):
		catlist = []

		# Instantiate the catalogs
		for catname, catpath, jointype in from_clause:
			catlist.append( ( catname, (CatalogEntry(self.catalog(catpath), catname), jointype) ) )
		cats = dict(catlist)

		# Discover and set up JOIN links based on defined JOIN relations
		# TODO: Expand this to allow the 'joined to via' and related syntax, once it becomes available in the parser
		for catname, (e, _) in catlist:
			# Check for catalogs that can be joined onto this one (where this one is on the right hand side of the relation)
			# Look for default .join files named ".<catname>:*.join"
			pattern = "%s/.%s:*.join" % (self.path, catname)
			for fn in glob.iglob(pattern):
				jcatname = fn[fn.rfind(':')+1:fn.rfind('.join')]
				if jcatname not in cats:
					continue

				je, jkind = cats[jcatname]
				if je.relation is not None:	# Already joined
					continue

				je.relation = create_join(self, fn, jkind, e.cat, je.cat)
				e.joins.append(je)
	
		# Discover the root (the one and only one catalog that has no links pointing to it)
		root = None
		for _, (e, jkind) in cats.iteritems():
			if e.relation is None:
				assert root is None	# Can't have more than one roots
				assert jkind == 'inner'	# Can't have something like 'ps1_obj(outer)' on root catalog
				root = e
		assert root is not None			# Can't have zero roots
	
		# TODO: Verify all catalogs are reachable from the root (== there are no circular JOINs)
	
		# Return the root of the tree, and a dict of all the Catalog instances
		return root, dict((  (catname, cat) for (catname, (cat, _)) in cats.iteritems() ))

	def build_neighbor_cache(self, cat_path, margin_x_arcsec=30):
		""" Cache the objects found within margin_x (arcsecs) of
		    each cell into neighboring cells as well, to support
		    efficient nearest-neighbor lookups.

		    This routine works in tandem with _cache_maker_mapper
		    and _cache_maker_reducer auxilliary routines.
		"""
		margin_x = np.sqrt(2.) / 180. * (margin_x_arcsec/3600.)

		ntotal = 0
		ncells = 0
		query = "_ID, _LON, _LAT FROM '%s'" % (cat_path)
		for (_, ncached) in self.query(query).execute([
						(_cache_maker_mapper,  margin_x, self, cat_path),
						(_cache_maker_reducer, self, cat_path)
					]):
			ntotal = ntotal + ncached
			ncells = ncells + 1
			#print self._cell_prefix(cell_id), ": ", ncached, " cached objects"
		print "Total %d cached objects in %d cells" % (ntotal, ncells)

	def compute_summary_stats(self, *cat_paths):
		""" Compute frequently used summary statistics and
		    store them into the dbinfo file. This should be called
		    to refresh the stats after insertions.
		"""
		from tasks import compute_counts
		for cat_path in cat_paths:
			cat = self.catalog(cat_path)
			cat._nrows = compute_counts(self, cat_path)
			cat._store_schema()

###################################################################
## Auxilliary functions implementing DB.build_neighbor_cache
## functionallity
def _cache_maker_mapper(qresult, margin_x, db, cat_path):
	# Map: fetch all rows to be copied to adjacent cells,
	# yield them keyed by destination cell ID
	for rows in qresult:
		cell_id = rows.cell_id
		p, _ = qresult.pix.cell_bounds(cell_id)

		# Find all objects within 'margin_x' from the cell pixel edge
		# The pixel can be a rectangle, or a triangle, so we have to
		# handle both situations correctly.
		(x1, x2, y1, y2) = p.boundingBox()
		d = x2 - x1
		(cx, cy) = p.center()
		if p.nPoints() == 4:
			s = 1. - 2*margin_x / d
			p.scale(s, s, cx, cy)
		elif p.nPoints() == 3:
			if (cx - x1) / d > 0.5:
				ax1 = x1 + margin_x*(1 + 2**.5)
				ax2 = x2 - margin_x
			else:
				ax1 = x1 + margin_x
				ax2 = x2 - margin_x*(1 + 2**.5)

			if (cy - y1) / d > 0.5:
				ay2 = y2 - margin_x
				ay1 = y1 + margin_x*(1 + 2**.5)
			else:
				ay1 = y1 + margin_x
				ay2 = y2 - margin_x*(1 + 2**.5)
			p.warpToBox(ax1, ax2, ay1, ay2)
		else:
			raise Exception("Expecting the pixel shape to be a rectangle or triangle!")

		# Now reject everything not within the margin, and
		# (for simplicity) send everything within the margin,
		# no matter close to which edge it actually is, to
		# all neighbors.
		(x, y) = bhpix.proj_bhealpix(rows['_LON'], rows['_LAT'])
		inMargin = ~p.isInsideV(x, y)

		if not inMargin.any():
			continue

		# Fetch only the rows that are within the margin
		idsInMargin = rows['_ID'][inMargin]
		q           = db.query("* FROM '%s' WHERE np.in1d(_ID, idsInMargin, assume_unique=True)" % cat_path, {'idsInMargin': idsInMargin} )
		data        = q.fetch_cell(cell_id)

		# Send these to neighbors
		if data is not None:
			for neighbor in qresult.pix.neighboring_cells(cell_id):
				yield (neighbor, data)

		##print "Scanned margins of %s*.h5 (%d objects)" % (db.catalog(cat_path)._cell_prefix(cell_id), len(data))

def _cache_maker_reducer(kv, db, cat_path):
	# Cache all rows to be cached in this cell
	cell_id, rowblocks = kv

	# Delete existing neighbors
	cat = db.catalog(cat_path)
	cat.drop_row_group(cell_id, 'cached')

	# Add to cache
	ncached = 0
	for rows in rowblocks:
		cat.append(rows, cell_id=cell_id, group='cached')
		ncached += len(rows)
		##print cell_id, len(rows), cat.pix.path_to_cell(cell_id)

	# Return the number of new rows cached into this cell
	yield cell_id, ncached

###############################################################
# Aux. functions implementing Query.iterate() and
# Query.fetch()
def _iterate_mapper(qresult):
	for rows in qresult:
		if len(rows):	# Don't return empty sets
#			print qresult.bounds
#			bounds = qresult.root.cat.pix.cell_bounds(qresult.cell_id)
#			(x, y) = bhpix.proj_bhealpix(rows['_LON'], rows['_LAT'])
#			print "INSIDE:", bounds[0].isInsideV(x, y), bounds[1].isInside(rows['mjd_obs'])
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
	qresult = qengine.on_cell(cell_id, bounds, include_cached)
	for result in mapper(qresult, *mapper_args):
		result = qresult.postprocess(result)
		yield result

###############################

def _fitskw_dumb(hdrs, kw):
	# Easy way
	res = []
	for ahdr in hdrs:
		hdr = pyfits.Header( txtfile=StringIO(ahdr) )
		res.append(hdr[kw])
	return res

def fits_quickparse(header):
	""" An ultra-simple FITS header parser. Does not support
	    CONTINUE statements, HIERARCH, or anything of the sort;
	    just plain vanilla:
	    	key = value / comment
	    one-liners. The upshot is that it's fast.

	    Assumes each 80-column line has a '\n' at the end
	"""
	res = {}
	for line in header.split('\n'):
		at = line.find('=')
		if at == -1 or at > 8:
			continue

		# get key
		key = line[0:at].strip()

		# parse value (string vs number, remove comment)
		val = line[at+1:].strip()
		if val[0] == "'":
			# string
			val = val[1:val[1:].find("'")]
		else:
			# number or T/F
			at = val.find('/')
			if at == -1: at = len(val)
			val = val[0:at].strip()
			if val.lower() in ['t', 'f']:
				# T/F
				val = val.lower() == 't'
			else:
				# Number
				val = float(val)
				if int(val) == val:
					val = int(val)
		res[key] = val
	return res;

def fitskw(hdrs, kw, default=0):
	""" Intelligently extract a keyword kw from an arbitrarely
	    shaped object ndarray of FITS headers.
	"""
	shape = hdrs.shape
	hdrs = hdrs.reshape(hdrs.size)

	res = []
	cache = dict()
	for ahdr in hdrs:
		ident = id(ahdr)
		if ident not in cache:
			if ahdr is not None:
				#hdr = pyfits.Header( txtfile=StringIO(ahdr) )
				hdr = fits_quickparse(ahdr)
				cache[ident] = hdr.get(kw, default)
			else:
				cache[ident] = default
		res.append(cache[ident])

	#assert res == _fitskw_dumb(hdrs, kw)

	res = np.array(res).reshape(shape)
	return res

def ffitskw(uris, kw, default = False, db=None):
	""" Intelligently load FITS headers stored in
	    <uris> ndarray, and fetch the requested
	    keyword from them.
	"""

	if len(uris) == 0:
		return np.empty(0)

	uuris, idx = np.unique(uris, return_inverse=True)
	idx = idx.reshape(uris.shape)

	if db is None:
		# _DB is implicitly defined inside queries
		db = _DB

	ret = []
	for uri in uuris:
		if uri is not None:
			with db.open_uri(uri) as f:
				hdr_str = f.read()
			hdr = fits_quickparse(hdr_str)
			ret.append(hdr.get(kw, default))
		else:
			ret.append(default)

	# Broadcast
	ret = np.array(ret)[idx]

	assert ret.shape == uris.shape, '%s %s %s' % (ret.shape, uris.shape, idx.shape)

	return ret

def OBJECT(uris, db=None):
	""" Dereference blobs referred to by URIs,
	    assuming they're pickled Python objects.
	"""
	return deref(uris, db, True)

def BLOB(uris, db=None):
	""" Dereference blobs referred to by URIs,
	    loading them as plain files
	"""
	return deref(uris, db, False)

def deref(uris, db=None, unpickle=False):
	""" Dereference blobs referred to by URIs,
	    either as BLOBs or Python objects
	"""
	if len(uris) == 0:
		return np.empty(0, dtype=object)

	uuris, idx = np.unique(uris, return_inverse=True)
	idx = idx.reshape(uris.shape)

	if db is None:
		# _DB is implicitly defined inside queries
		db = _DB

	ret = np.empty(len(uuris), dtype=object)
	for i, uri in enumerate(uuris):
		if uri is not None:
			with db.open_uri(uri) as f:
				if unpickle:
					ret[i] = cPickle.load(f)
				else:
					ret[i] = f.read()
		else:
			ret[i] = None

	# Broadcast
	ret = np.array(ret)[idx]

	assert ret.shape == uris.shape, '%s %s %s' % (ret.shape, uris.shape, idx.shape)

	return ret

###############################

def test_kernel(qresult):
	for rows in qresult:
		yield qresult.cell_id, len(rows)

if __name__ == "__main__":
	def test():
		from tasks import compute_coverage
		db = DB('../../../ps1/db')
		query = "_LON, _LAT FROM sdss(outer), ps1_obj, ps1_det, ps1_exp(outer)"
		compute_coverage(db, query)
		exit()

		import footprint as foot
		query = "_ID, ps1_det._ID, ps1_exp._ID, sdss._ID FROM '../../../ps1/sdss'(outer) as sdss, '../../../ps1/ps1_obj' as ps1_obj, '../../../ps1/ps1_det' as ps1_det, '../../../ps1/ps1_exp'(outer) as ps1_exp WHERE True"
		query = "_ID, ps1_det._ID, ps1_exp._ID, sdss._ID FROM sdss(outer), ps1_obj, ps1_det, ps1_exp(outer)"
		cell_id = np.uint64(6496868600547115008 + 0xFFFFFFFF)
		include_cached = False
		bounds = [(foot.rectangle(120, 20, 160, 60), None)]
		bounds = None
#		for rows in QueryEngine(query).on_cell(cell_id, bounds, include_cached):
#			pass;

		db = DB('../../../ps1/db')
#		dq = db.query(query)
#		for res in dq.execute([test_kernel], bounds, include_cached, nworkers=1):
#			print res

		dq = db.query(query)
#		for res in dq.iterate(bounds, include_cached, nworkers=1):
#			print res

		res = dq.fetch(bounds, include_cached, nworkers=1)
		print len(res)

	test()
