#!/usr/bin/env python
"""
TODO: A module in need of urgent refactoring.

Contains (among others) DB, Query, QueryEngine and QueryInstance classes, as
well as the JOIN machinery.

"""
import os, json, glob, copy
import numpy as np
import cPickle
import pyfits
import logging

from contextlib  import contextmanager
from collections import defaultdict

import query_parser as qp
import bhpix
import utils
import pool2
import mr
import native
import colgroup

from interval    import intervalset
from colgroup    import ColGroup
from table       import Table

import caching

@caching.cached
def cached_proj_bhealpix(lon, lat):
	return bhpix.proj_bhealpix(lon, lat)

@caching.cached
def cached_isInsideV(bounds_xy, x, y):
	return bounds_xy.isInsideV(x, y)

def set_NULL(col, mask=np.s_[:]):
	""" Set the NULL marker apropriate for the datatype """
	col[mask] = 0

class TabletCache:
	""" An cache of tablets loaded while performing a Query.

		TODO: Perhaps merge it with DB? Or make it a global?
	"""
	cache = {}		# Cache of loaded tables, in the form of cache[cell_id][include_cached][table][cgroup] = rows

	root_path = None	# The name of the root table (string). Used for figuring out if _not_ to load the cached rows.
	include_cached = False	# Should we load the cached rows from the root table?

	def __init__(self, root_path, include_cached = False):
		self.cache = {}
		self.root_path = root_path
		self.include_cached = include_cached

	def load_column(self, cell_id, name, table, autoexpand=True, resolve_blobs=False):
		# Return the column 'name' from table 'table'.
		# Load its tablet if necessary, and cache it for further reuse.
		#
		# NOTE: Unless resolve_blobs=True, this method DOES NOT resolve blobrefs to BLOBs
		include_cached = self.include_cached if table.path == self.root_path else True

		# Resolve a column name alias
		name = table.resolve_alias(name)

		# Figure out which table contains this column
		cgroup = table.columns[name].cgroup

		# Create self.cache[cell_id][table.name][include_cached][cgroup] hierarchy if needed
		if  cell_id not in self.cache:
			self.cache[cell_id] = {}
		if table.name not in self.cache[cell_id]:
			self.cache[cell_id][table.name] = {}
		if include_cached not in self.cache[cell_id][table.name]:		# This bit is to support (in the future) self-joins. Note that (yes) this can be implemented in a much smarter way.
			self.cache[cell_id][table.name][include_cached] = {}

		# See if we have already loaded the required tablet
		tcache = self.cache[cell_id][table.name][include_cached]
		if cgroup not in tcache:
			# Load and cache the tablet
			rows = table.fetch_tablet(cell_id, cgroup, include_cached=include_cached)

			# Ensure it's as long as the primary table (this allows us to support "sparse" tablets)
			if autoexpand and cgroup != table.primary_cgroup:
				nrows = len(self.load_column(cell_id, table.primary_key.name, table))
				rows.resize(nrows)

			tcache[cgroup] = rows
		else:
			rows = tcache[cgroup]

		col = rows[name]
		
		# resolve blobs, if requested
		if resolve_blobs:
			col = self.resolve_blobs(cell_id, col, name, table)

		return col

	def resolve_blobs(self, cell_id, col, name, table):
		# Resolve blobs (if blob column). NOTE: the resolved blobs
		# will not be cached.

		if table.columns[name].is_blob:
			include_cached = self.include_cached if table.path == self.root_path else True
			col = table.fetch_blobs(cell_id, column=name, refs=col, include_cached=include_cached)

		return col

class TableEntry:
	""" A table that is a member of a Query. See DB.construct_join_tree() for details.
	"""
	table    = None	# A Table() instance
	name     = None # Table name, as named in the query (may be different from table.name, if 'table AS other' construct was used)
	relation = None	# Relation to use to join with the parent
	joins    = None	# List of JoinEntries

	def __init__(self, table, name):
		self.table = table
		self.name  = name if name is not None else name
		self.joins = []

	def get_cells(self, bounds, include_cached=True):
		""" Get populated cells of self.table that overlap
			the requested bounds. Do it recursively.
			
			The bounds can be either a list of (Polygon, intervalset) tuples,
			or None.
			
			For a static cell_id, a dynamic table will return all
			temporal cells within the same spatial cell. For a dynamic
			cell_id, a static table will return the overlapping
			static cell_id.
		"""

		# Fetch our populated cells
		pix = self.table.pix
		cells = self.table.get_cells(bounds, return_bounds=True, include_cached=include_cached)

		# Autodetect if we're a static or temporal table
		self.static = True
		for cell_id in cells:
			if self.table.pix.is_temporal_cell(cell_id):
				self.static = False
				break
		# print "TTT:", self.table.name, ":", self.static

		# Fetch the children's populated cells
		for ce in self.joins:
			cc, op = ce.get_cells(bounds)
			if   op == 'and':
				# Construct new cell list by adding cells from both existing, and 
				# the new one that are covered in the other list.
				ret = dict()
				c1, c2 = cc, cells
				for _ in xrange(2):
					for cell_id, cbounds in c1.iteritems():
						assert cell_id not in c2 or c2[cell_id] == cbounds # TODO: Debugging -- make sure the timespace constraints are the same
						if cell_id in c2:
							ret[cell_id] = cbounds
						else:
							static_cell = pix.static_cell_for_cell(cell_id)
							if static_cell in c2:
								ret[static_cell] = c2[static_cell]	# Keep the static cell for future comparisons
								ret[cell_id] = cbounds
					c1, c2 = cells, cc
				cells = ret
			elif op == 'or':
				print "or"
				for cell_id, cbounds in cc.iteritems():
					assert cell_id not in cells or cells[cell_id] == cbounds # TODO: Debugging -- make sure the timespace constraints are the same
#					if not (cell_id not in cells or cells[cell_id] == cbounds): # TODO: Debugging -- make sure the timespace constraints are the same
#						aa = cells[cell_id]
#						bb = cbounds
#						pass
					assert cell_id not in cells or cells[cell_id] == cbounds # Debugging -- make sure the timespace constraints are the same
					cells[cell_id] = cbounds

		if self.relation is None:
			# Remove all static cells if there's even a single temporal cell
			cells2 = dict(( v for v in cells.iteritems() if pix.is_temporal_cell(v[0]) ))
			cells = cells2 if cells2 else cells
			return cells
		else:
			return cells, self.relation.join_op()

	def evaluate_join(self, cell_id, bounds, tcache, r = None, idxkey = None, rtable = None):
		""" Constructs a JOIN index array.

			* If the result of the join is no rows, return None

		    * If the result is not empty, the return is a ColGroup() instance
		      that _CAN_ (but DOESN'T HAVE TO; see below) have the 
		      following columns:

		    	<tabname> : the index array that will materialize
				    the JOIN result when applied to the corresponding
				    table's column as:
				    
				    	res = col[<tabname>]

		    	<tabname>._ISNULL : a boolean array indicating whether
		    	            the <tabname> index is a dummy (zero) and
		    	            actually the column has JOINed to NULL
		    	            (this only happens with outer joins)

		      -- If the col[<tabname>] would be == col, there'll be no
		         <tabname> column.
		"""
		hasBounds = bounds != [(None, None)] and bounds != [(None, intervalset((-np.inf, np.inf)))]
		if len(bounds) > 1:
			pass;
		# Skip everything if this is a single-table read with no bounds
		# ("you don't pay for what you don't use")
		if self.relation is None and not hasBounds and not self.joins:
			return ColGroup()

		# Load ourselves
		id = tcache.load_column(cell_id, self.table.get_primary_key(), self.table)
		s = ColGroup()

		s.add_column('%s._ID' % self.name, id)		# The primary key of the left table
		s.add_column(self.name, np.arange(len(id)))	# The index into rows of this tablet, as loaded from disk

		if self.relation is None:
			# We are root.
			r = s

			# Setup spatial bounds filter
			if hasBounds:
				ra, dec = self.table.get_spatial_keys()
				lon = tcache.load_column(cell_id,  ra, self.table)
				lat = tcache.load_column(cell_id, dec, self.table)

				r = self.filter_space(r, lon, lat, bounds)	# Note: this will add the _INBOUNDS column to r
		else:
			# We're a child. Join with the parent table
			idx1, idx2, m = self.relation.join(cell_id, r[idxkey], s[self.name], tcache)

			# Handle the special case of OUTER JOINed empty 's'
			if len(s) == 0 and len(idx2) != 0:
				assert m._ISNULL.all()
				s = ColGroup(dtype=s.dtype, size=1)

			# Perform the joins
			r = r[idx1]
			s = s[idx2]
			r.add_columns(s.items())
			assert np.all(idx2 == r[self.name])

			# Add all columns generated by the join
			for colname in m.keys():
				assert colname[0] == '_'

				r.add_column("%s.%s" % (self.name, colname), m[colname])

		# Perform spacetime cuts, if we have a time column
		# (and the JOIN didn't result in all NULLs)
		if hasBounds:
			tk = self.table.get_temporal_key()
			if tk is not None:
				t = tcache.load_column(cell_id,  tk, self.table)
				if len(t):
					r = self.filter_spacetime(r, t[r[self.name]], bounds)

		# Let children JOIN themselves onto us
		for ce in self.joins:
			r = ce.evaluate_join(cell_id, bounds, tcache, r, self.name, self.table)

		if self.relation is None:
			# Return None if the query yielded no rows
			if len(r) == 0:
				return None
			# Cleanup: if we are root, remove the _INBOUNDS helper column
			if '_INBOUNDS' in r:
				r.drop_column('_INBOUNDS')
			# Add a dummy _ISNULL column for root table (simplifies the code later on)
			r["%s._ISNULL" % (self.name)] = np.zeros(len(r), dtype='bool')
			# Sort the jmap to make the adjacent joined rows show up together in output
			#       and be sorted by neighbor rank (if it exists)
			# TODO: Do we really want to waste cycles on this, or is the user the one
			#       who should to the sorting if they need it?
			primkeys = [ key for key in r.keys() if key.endswith("._ID") ]
			sortkeys = []
			keys = r.keys()
			for k in primkeys:
				nrk = k[:-4] + "._NR"
				if nrk in keys:
					sortkeys.append(nrk)
				sortkeys.append(k)
			r.sort(sortkeys)
			# Drop the primary keys, they're not needed any more
			for k in primkeys:
				r.drop_column(k)

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
					#(x, y) = bhpix.proj_bhealpix(lon, lat)
					(x, y) = cached_proj_bhealpix(lon, lat)
					#assert np.prod([np.all(a == b) for a, b in zip((x, y), bhpix.proj_bhealpix(lon, lat))])
					#print "HERE! ", len(x), cached_proj_bhealpix.stats()

				#inbounds[:, i] &= bounds_xy.isInsideV(x, y)
				inbounds[:, i] &= cached_isInsideV(bounds_xy, x, y)
				#assert np.all(bounds_xy.isInsideV(x, y) == cached_isInsideV(bounds_xy, x, y))
				#print "HERE2!", len(x), cached_isInsideV.stats()

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

def native_join(id1, id2, kind, cg):
	"""
		Helper that performs a join of tables R and S on a common row
		whose values are given in id1 (for R) and id2 (for S).
			
		Join kind can be either 'inner' or 'outer'.

		The id1<->id2 mapping is given by (cg.m1, cg.m2) tuples.
			
		Returned are arrays idx1 and idx2, that can be used to constitute
		a new table by stacking columns produced by R[idx1] and S[idx2].

		Also returned is cg, that has and _ISNULL column == True for
		each row in the joined table where S's columns were NULL (because
		of the outer join). cg can also contain additional information,
		if given on input (e.g., _DIST or _NR).
	"""
	(idx1, idx2, idxLink, isnull) = native.table_join(id1, id2, cg.m1, cg.m2, kind)

	if len(cg) > 0:
		assert np.all(idxLink < len(cg))
	else:
		assert np.all(idxLink == 0)
		assert np.all(idx2 == 0)
		assert np.all(isnull)
		cg.resize(1)

	cg = cg[idxLink]
	cg._ISNULL = isnull
	del cg.m1, cg.m2

	# Null-out the columns that are NULL
	for colname in cg.keys():
		if colname == '_ISNULL':
			continue
		set_NULL(cg[colname], cg._ISNULL)

	return (idx1, idx2, cg)

class JoinRelation:
	kind   = 'inner'
	db     = None	# Controlling database instance
	tableR = None	# Right-hand side table of the relation
	tableS = None	# Left-hand side table of the relation
	
	def __init__(self, db, tableR, tableS, **joindef):
		self.db     = db
		self.kind   = 'inner' if 'outer' not in joindef else 'outer'
		self.tableR = tableR
		self.tableS = tableS

	def join_op(self):	# Returns 'and' if the relation has an inner join-like effect, and 'or' otherwise
		return 'and' if self.kind == 'inner' else 'or'

	def join(self, cell_id, table1, table2, idx1, idx2, tcache):	# Returns idx1, idx2, isnull
		raise NotImplementedError('You must override this method from a derived class')

class IndirectJoin(JoinRelation):
	m1_colspec = None	# (table, column) tuple giving the location of m1
	m2_colspec = None	# (table, column) tuple giving the location of m2

	def fetch_join_map(self, cell_id, m1_colspec, m2_colspec, tcache):
		"""
			Return a list of crossmatches corresponding to ids
		"""
		table1, column_from = m1_colspec
		table2, column_to   = m2_colspec
		
		# This allows tripple
		# joins of the form static-static-temporal, where the cell_id
		# will be temporal even when fetching the static-static join
		# table for the two other table.
		cell_id_from = table1.static_if_no_temporal(cell_id)
		cell_id_to   = table2.static_if_no_temporal(cell_id)

		cg = ColGroup()

		if    not table1.tablet_exists(cell_id_from) \
		   or not table2.tablet_exists(cell_id_to):
		   	cg.m1 = np.empty(0, dtype=np.uint64)
		   	cg.m2 = np.empty(0, dtype=np.uint64)
		   	return cg

		cg.m1 = tcache.load_column(cell_id_from, column_from, table1)
		cg.m2 = tcache.load_column(cell_id_to  , column_to  , table2)
		assert len(cg.m1) == len(cg.m2)

		# Fetch columns with extra join data, on which we might filter or which
		# the user may SELECT
		for colname in ['_NR', '_DIST']:
			try:
				cg[colname] = tcache.load_column(cell_id_from, colname, table1)
			except KeyError:
				pass

		# Apply cuts
		if getattr(cg, '_NR', None) is not None:
			cg.cut(cg._NR < self.n)
		elif self.n != 1:
			raise Exception("No _NR column in indirect join table, and nmax != 1")

		if getattr(cg, '_DIST', None) is not None:
			if self.d != 0:
				cg.cut(cg._DIST < self.d)
		elif self.d != 0:
			raise Exception("No _DIST column in indirect join table, and dmax != 0")

		return cg

	def join(self, cell_id, idx1, idx2, tcache):
		"""
		    Perform a JOIN on id1, id2, that were obtained by
		    indexing their origin tables with idx1, idx2.
		"""
		cg = self.fetch_join_map(cell_id, self.m1_colspec, self.m2_colspec, tcache)

		id1 = tcache.load_column(cell_id, self.tableR.get_primary_key(), self.tableR)[idx1]
		id2 = tcache.load_column(cell_id, self.tableS.get_primary_key(), self.tableS)[idx2]

		return native_join(id1, id2, self.kind, cg)

	def __init__(self, db, tableR, tableS, **joindef):
		JoinRelation.__init__(self, db, tableR, tableS, **joindef)

		m1_tabname, m1_colname = joindef['m1']
		m2_tabname, m2_colname = joindef['m2']

		# Number of neighbors, max distance (these are usually given as FROM clause args)
		self.n = int(joindef.get('nmax', 1))
		self.d = float(joindef.get('dmax', 0)) / 3600. # fetch and convert to degrees

		self.m1_colspec = (db.table(m1_tabname), m1_colname)
		self.m2_colspec = (db.table(m2_tabname), m2_colname)

	def __str__(self):
		return "%s indirect via [%s.%s, %s.%s]" % (
			self.kind,
			self.m1_colspec[0], self.m1_colspec[1].name,
			self.m2_colspec[0], self.m2_colspec[1].name,
		)

class CrossmatchJoin(JoinRelation):
	def __init__(self, db, tableR, tableS, **joindef):
		JoinRelation.__init__(self, db, tableR, tableS, **joindef)

		# Number of neighbors, max distance (these are usually given as FROM clause args)
		self.n = int(joindef.get('nmax', 1))
		self.d = float(joindef.get('dmax', 1.)) / 3600. # fetch and convert to degrees

	def join(self, cell_id, idx1, idx2, tcache):
		"""
		    Perform a JOIN on id1, id2, that were obtained by
		    indexing their origin tables with idx1, idx2.
		"""
		# Cross-match, R x S
		# Return objects (== rows) from S that are nearest neighbors of
		# objects (== rows) in R
		from scikits.ann import kdtree
		from utils import gnomonic, gc_dist

		join = ColGroup(dtype=[('m1', 'u8'), ('m2', 'u8'), ('_DIST', 'f4'), ('_NR', 'u1')])

		# Load spatial keys from S table
		rakey, deckey = self.tableS.get_spatial_keys()
		ra2, dec2 = tcache.load_column(cell_id, rakey, self.tableS), tcache.load_column(cell_id, deckey, self.tableS)

		if len(ra2) != 0:
			# Get all objects in R for which neighbors in S will be
			# looked up
			rakey, deckey = self.tableR.get_spatial_keys()
			uidx1 = np.unique(idx1)
			ra1, dec1 = \
				tcache.load_column(cell_id, rakey, self.tableR)[uidx1], \
				tcache.load_column(cell_id, deckey, self.tableR)[uidx1]

			# Project to tangent plane around the center of the cell. We
			# assume the cell is small enough for the distortions not to
			# matter and Euclidian distances apply
			bounds, _    = self.tableR.pix.cell_bounds(cell_id)
			(clon, clat) = bhpix.deproj_bhealpix(*bounds.center())
			xy1 = np.column_stack(gnomonic(ra1, dec1, clon, clat))
			xy2 = np.column_stack(gnomonic(ra2, dec2, clon, clat))

			# Construct kD-tree to find nearest neighbors from tableS
			# for every object in tableR
			tree = kdtree(xy2)
			match_idxs, match_d2 = tree.knn(xy1, min(self.n, len(xy2)))
			del tree

			# Expand the matches into a table, with one row per neighbor
			join.resize(match_idxs.size)
			for k in xrange(match_idxs.shape[1]):
				match_idx = match_idxs[:,k]
				join['m1'][k::match_idxs.shape[1]]   = uidx1
				join['m2'][k::match_idxs.shape[1]]   = match_idx
				join['_DIST'][k::match_idxs.shape[1]] = gc_dist(ra1, dec1, ra2[match_idx], dec2[match_idx])
				join['_NR'][k::match_idxs.shape[1]]   = k

			# Remove matches beyond the xmatch radius
			join = join[join['_DIST'] < self.d]

		# Perform the join
		assert idx1.dtype == idx2.dtype == np.int64
		id1, id2 = idx1.view(np.uint64), idx2.view(np.uint64) # Because native.table_join expects uint64 data
		return native_join(id1, id2, self.kind, join)

#class EquijoinJoin(IndirectJoin):
#	def __init__(self, db, tableR, tableS, kind, **joindef):
#		JoinRelation.__init__(self, db, tableR, tableS, kind, **joindef)
#
#		# Emulating direct join with indirect one
#		# TODO: Write a specialized direct join routine
#		self.m1_colspec = tableR, joindef['id1']
#		self.m2_colspec = tableS, joindef['id2']

def create_join(db, fn, jargs, tableR, tableS, jclass=None):
	if fn is not None:
		if 'j' in jargs:					# Allow the join file to be overridden
			j = jargs['j']
			if j.find('/') == '-1':
				fn = '%s/%s.join' % (db.path, j)	# j=myjoin
			else:
				fn = j					# j=bla/myjoin.join

		data = json.loads(file(fn).read())
	else:
		data = dict()

	# override the joindef with args from the FROM clause
	data.update(jargs)

	if jclass is None:
		assert 'type' in data
		jclass = globals()[data['type'].capitalize() + 'Join']

	return jclass(db, tableR, tableS, **data)

class iarray(np.ndarray):
	"""
	Subclass of ndarray allowing per-row indexing (with arr(ind)).

	All columns within a query are of this type. It allows the user to
	write queries such as:
	
	    SELECT chips(3) FROM ...
	   
	where chips is an array column (i.e., each row is an array). In the
	above query, the user expects to get the fourth element (remember,
	0-based indexing!) of chips within each row. But since columns are
	numpy arrays (vectors) where the row is the first index, we must
	(roughly) translate this internally into:
	
	    chips[:, 3]

	to work as expected. This is what the __call__ routine of this class
	does.
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
	""" A dict-like object for query_parser.resolve_wildcards()
	"""
	tables = None
	root_table = None

	def __init__(self, root_table, tables):
		self.root_table = root_table
		self.tables = tables

	def __getitem__(self, tabname):
		# Return a list of columns in table tabname, unless
		# they're pseudocolumns
		if tabname == '':
			tabname = self.root_table
		table = self.tables[tabname].table
		return [ name for (name, coldef) in table.columns.iteritems() if not table._is_pseudotablet(coldef.cgroup) ]

class QueryInstance(object):
	# Internal working state variables
	tcache   = None		# TabletCache() instance
	columns  = None		# Cache of already referenced and evaluated columns (dict)
	cell_id  = None		# cell_id on which we're operating
	jmap 	 = None		# index map used to materialize the JOINs
	bounds   = None

	# These will be filled in from a QueryEngine instance
	db       = None		# The controlling database instance
	tables   = None		# A name:Table dict() with all the tables listed on the FROM line.
	root	 = None		# TableEntry instance with the primary (root) table
	query_clauses = None	# Tuple with parsed query clauses
	pix      = None         # Pixelization object (TODO: this should be moved to class DB)
	locals   = None		# Extra local variables to be made available within the query

	def __init__(self, q, cell_id, bounds, include_cached):
		self.db            = q.db
		self.tables	   = q.tables
		self.root	   = q.root
		self.query_clauses = q.query_clauses
		self.pix           = q.root.table.pix
		self.locals        = q.locals

		self.cell_id	= cell_id
		self.bounds	= bounds

		self.tcache	= TabletCache(self.root.table.path, include_cached)
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

				if(in_.any()):
					if not in_.all():
						rows = rows[in_]

					# Attach metadata
					rows.info.cell_id = self.cell_id

					yield rows

		# We yield nothing if the result set is empty.

	def prep_globals(self):
		globals_ = globals()

		# Import packages of interest (numpy)
		for i in np.__all__:
			if len(i) >= 2 and i[:2] == '__':
				continue
			globals_[i] = np.__dict__[i]

		# Add implicit global objects present in queries
		globals_['_PIX'] = self.root.table.pix
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

		rows = ColGroup()
		for (asname, name) in select_clause:
#			col = self[name]	# For debugging
			col = eval(name, globals_, self)
#			exit()

			self[asname] = col
			rows.add_column(asname, col)

		return rows

	#################

	def load_column(self, name, tabname):
		# If we're just peeking, construct the column from schema
		if self.cell_id is None:
			assert self.bounds is None
			cdef = self.tables[tabname].table.columns[name]
			dtype = np.dtype(cdef.dtype)

			# Handle blobs
			if cdef.is_blob:
				col = np.empty(0, dtype=np.dtype(('O', dtype.shape)))
			else:
				col = np.empty(0, dtype=dtype)
		else:
			# Load the column from cgroup 'cgroup' of table 'tabname'
			# Also cache the loaded tablet, for future reuse
			table = self.tables[tabname].table

			# Load the column (via cache)
			col = self.tcache.load_column(self.cell_id, name, table)

			# Join/filter if needed
			if tabname in self.jmap:
				isnullKey = tabname + '._ISNULL'
				idx       = self.jmap[tabname]
				if len(col):
					# TODO: We can optimize here by remembering if np.all(isnull == 0) and if idx=arange(...)
					# and storing that into self.jmap.info
					col = col[idx]
					isnull = self.jmap[isnullKey]
					set_NULL(col, isnull)
				elif len(idx):
					# This tablet is empty, but columns show up here because of an OUTER join.
					# Just return NULLs of proper dtype
					assert self.jmap[isnullKey].all()
					assert (idx == 0).all()
					col = np.empty(shape=(len(idx),) + col.shape[1:], dtype=col.dtype)
					set_NULL(col)

			# Resolve blobs (if a blobref column)
			col = self.tcache.resolve_blobs(self.cell_id, col, name, table)

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
		elif name == '_CELLPATH':
			ret = np.empty(nrows, dtype='a128')
			ret[:] = self.pix.path_to_cell(self.cell_id)
			return ret
		else:
			raise Exception('Unknown pseudocolumn %s' % name)

	def __getitem__(self, name):
		# An already loaded column?
		if name in self.columns:
			return self.columns[name]

		# A yet unloaded column from one of the joined tables
		# (including the primary)? Try to find it in cgroups of
		# joined tables. It may be prefixed by the table name, in
		# which case we force the lookup of that table only.
		if name.find('.') == -1:
			# Do this to ensure the primary table is the first
			# to get looked up when resolving column names.
			tables = [ (self.root.name, self.tables[self.root.name]) ]
			tables.extend(( (name, table) for name, table in self.tables.iteritems() if name != self.root.name ))
			colname = name
		else:
			# Force lookup of a specific table + column ...
			ns = name.split('.')
			(tabname, colname) = '.'.join(ns[:-1]), ns[-1]
			try:
				tables = [ (tabname, self.tables[tabname]) ]
			except KeyError:
				# ... or dbdir + table
				if tabname == "db":
					colname = name
					tables = []
				else:
					raise
		for (tabname, e) in tables:
			colname = e.table.resolve_alias(colname)
			if colname in e.table.columns:
				self[name] = self.load_column(colname, tabname)
				return self.columns[name]

		# A name of a table? Return a proxy object
		if name in self.tables:
			return TableProxy(self, name)

		# A name of a dbdir? Also return a proxy
		if name == "db":
			return TableProxy(self, name)

		# A query pseudocolumn?
		if name in ['_ROWNUM', '_CELLID', '_CELLPATH']:
			col = self[name] = self.load_pseudocolumn(name)
			return col

		# A join-generated pseudocolumn (like _ISNULL, _NR, etc..)?
		# Note that these _must_ be prefixed by table name
		if name.find('.') != -1 and name in self.jmap:
			return self.jmap[name]

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

class IntoWriter(object):
	""" Implementation of the writing logic for '... INTO ...' query clause
	"""
	into_clause = None
	_tcache = None
	rows = None
	locals = None
	db = None
	pix = None
	table = None

	def __init__(self, db, into_clause, locals = {}):
		# This handles INTO clauses. Stores the data into
		# destination table, returning the IDs of stored rows.
		if isinstance(into_clause, str):
			query = "_ FROM _ INTO %s" % into_clause
			_, _, _, into_clause = qp.parse(query)

		self.db          = db
		self.into_clause = into_clause
		self.locals      = locals

	@property
	def tcache(self):
		# Auto-create a tablet cache if needed
		if self._tcache is None:
			into_table   = self.into_clause[0]
			self._tcache = TabletCache(self.db.table(into_table).path, False)
		return self._tcache

	#########
	def __getitem__(self, name):
		# Do we have this column in the query result rows?
		if name in self.rows:
			return self.rows[name]

		# Is this one of the local variables passed in by the user?
		if name in self.locals:
			return self.locals[name]

		# This object is unknown to us -- let it fall through, it may
		# be a global variable or function
		raise KeyError(name)

	#################

	def peek(self, rows):
		# We always return uint64 arrays
		return np.empty(0, dtype='u8')

	def write(self, cell_id, rows):
		assert isinstance(rows, ColGroup)
		self.rows = rows

		# The table may have been created on previous pass
		if self.table is None:
			self.table  = self.create_into_table()

		rows = self.eval_into(cell_id, rows)
		return (cell_id, rows)

	def _find_into_dest_rows(self, cell_id, table, into_col, vals):
		""" Return the keys of rows in table whose 'into_col' value
		    matches vals. Return zero for vals that have no match.
		"""
		into_col = table.resolve_alias(into_col)
		col = self.tcache.load_column(cell_id, into_col, table, autoexpand=False, resolve_blobs=True)
#		print "XX:", col, vals;

		if len(col) == 0:
			return np.zeros(len(vals), dtype=np.uint64)

		# Find corresponding rows
		ii = col.argsort()
		scol = col[ii]
		idx = np.searchsorted(scol, vals)
#		print "XX:", scol, ii

		idx[idx == len(col)] = 0
		app = scol[idx] != vals
#		print "XX:", idx, app

		# Reorder to original ordering
		idx = ii[idx]
#		print "XX:", idx, app

		# TODO: Verify (debugging, remove when happy)
		in2 = np.in1d(vals, col)
#		print "XX:", in2
		assert np.all(in2 == ~app)

		id = self.tcache.load_column(cell_id, table.primary_key.name, table, autoexpand=False)
		id = id[idx]		# Get the values of row indices that we have
		id[app] = 0		# Mark empty indices with 0
#		print "XX:", id;

		return id

	def prep_globals(self):
		globals_ = globals()

		# Add implicit global objects present in queries
		globals_['_PIX'] = self.table.pix
		globals_['_DB']  = self.db
		
		return globals_

	def eval_into(self, cell_id, rows):
		# Insert into the destination table
		(into_table, _, into_col, keyexpr, kind) = self.into_clause

		table = self.db.table(into_table)
		if kind == 'append':
			rows.add_column('_ID', cell_id, 'u8')
			ids = table.append(rows)
		elif kind in ['update/ignore', 'update/insert']:
			# Evaluate the key expression
			globals_ = self.prep_globals()
			vals = eval(keyexpr, globals_, self)
#			print rows['mjd_obs'], rows['mjdorig'], keyexpr, vals; exit()

			# Match rows
			id = self._find_into_dest_rows(cell_id, table, into_col, vals)
			if table.primary_key.name not in rows:
				rows.add_column('_ID', id)
				key = '_ID'
			else:
				key = table.primary_key.name
				assert np.all(rows[key][id != 0] == id[id != 0])
				#rows[key] = id

			if kind == 'update/ignore':
				# Remove rows that don't exist, and update existing
				rows = rows[id != 0]
			else:
				# Append the rows whose IDs are unspecified
				rows[key][ rows[key] == 0 ] = cell_id
			ids = table.append(rows, _update=True)
#			print rows['_ID'], ids

			assert np.all(id[id != 0] == ids)
		elif kind == 'insert':	# Insert/update new rows (the expression give the key)
			# Evaluate the key expression
			globals_ = self.prep_globals()
			id = eval(keyexpr, globals_, self)

			if table.primary_key.name not in rows:
				rows.add_column('_ID', id)
			else:
				assert np.all(rows[table.primary_key.name] == id)
				rows[table.primary_key.name] = id

			# Update and/or add new rows
			ids = table.append(rows, _update=True)

		# Return IDs only
		for col in rows.keys():
			rows.drop_column(col)
		rows.add_column('_ID', ids)
#		print "HERE", rows; exit()
		
		return rows

	def create_into_table(self):
		# called to auto-create the destination table
		(tabname, dtype, into_col, keyexpr) = self.into_clause[:4]
		assert dtype is None, "User-supplied dtype not supported yet."

		db = self.db
		dtype = self.rows.dtype
		schema = {
			'columns': [],
			'filters': { 'complevel': 1, 'complib': 'zlib', 'fletcher32': True }, # Enable compression and checksumming
		}
		with db.lock():
			if db.table_exists(tabname):
				# Must have a designated key for updating to work
#				if into_col is None:
#					raise Exception('If selecting into an existing table (%s), you must specify the column with IDs that will be updated (" ... INTO ... AT keycol") construct).' % tabname)

				table = db.table(tabname)

				# Find any new columns we'll need to create
				for name in dtype.names:
					rname = table.resolve_alias(name)
					if rname not in table.columns:
						schema['columns'].append((name, utils.str_dtype(dtype[name])))

				# Disallow creating new columns
				if into_col is None and schema['columns']:
					raise Exception('If selecting into an existing table (%s) with no INTO ... WHERE clause, all columns present in the query must already exist in the table' % tabname)
			else:
				# Creating a new table
				table = db.table(tabname, True)

				# Create all columns
				schema['columns'] = [ (name, utils.str_dtype(dtype[name])) for name in dtype.names ]
				
				# If key is specified, and is a column name from self.rows, name the primary
				# key after it
				#if keyexpr is not None and keyexpr in self.rows:
				#	schema['primary_key'] = keyexpr
				#else:
				#	schema['primary_key'] = '_id'
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
					if tname not in table._cgroups:
						break
				table.create_cgroup(tname, schema)

		##print "XXX:", tname, schema;# exit()
		return table

class QueryEngine(object):
	# These are a part of the mappers' public API
	db       = None		# The controlling database instance
	pix      = None         # Pixelization object (TODO: this should be moved to class DB)
	tables = None		# A name:Table dict() with all the tables listed on the FROM line.
	root	 = None		# TableEntry instance with the primary (root) table
	query_clauses  = None	# Parsed query clauses
	locals   = None		# Extra variables to be made local to the query

	def __init__(self, db, query, locals = {}):
		self.db = db

		# parse query
		(select_clause, where_clause, from_clause, into_clause) = qp.parse(query)

		self.root, self.tables = db.construct_join_tree(from_clause);
		select_clause            = qp.resolve_wildcards(select_clause, TableColsProxy(self.root.name, self.tables))

		self.query_clauses       = (select_clause, where_clause, from_clause, into_clause)

		self.locals = locals

		# Aux variables that mappers can access
		self.pix = self.root.table.pix

	def on_cell(self, cell_id, bounds=None, include_cached=False):
		return QueryInstance(self, cell_id, bounds, include_cached)

	def on_cells(self, partspecs, include_cached=False):
		# Set up the args for __iter__
		self._partspecs = partspecs
		self._include_cached = include_cached

		# Set the static cell
		if partspecs:
			cell_id = partspecs[0][0]
			self.static_cell = self.pix.static_cell_for_cell(cell_id)

		return self

	def __iter__(self):
		# Generate a single stream of row blocks for a list of cells+bounds
		partspecs, include_cached = self._partspecs, self._include_cached

		for cell_id, bounds in partspecs:
			for rows in QueryInstance(self, cell_id, bounds, include_cached):
				yield rows

	def peek(self):
		return QueryInstance(self, None, None, None).peek()

class Query(object):
	db      = None
	qengine = None
	qwriter = None
	query_string = None

	def __str__(self):
		"""
		Returns the query string.
		"""
		return self.query_string

	def __init__(self, db, query, locals = {}):
		"""
		Internal: Constructs the query.
		
		Do not attempt to instantiate Query objects directly; use
		DB.query() function instead.
		"""
		self.db		  = db
		self.query_string = query
		self.qengine = QueryEngine(db, query, locals=locals)

		(_, _, _, into_clause) = qp.parse(query)
		if into_clause:
			self.qwriter = IntoWriter(db, into_clause, locals)

	def execute(self, kernels, bounds=None, include_cached=False, cells=[], group_by_static_cell=False, testbounds=True, nworkers=None, progress_callback=None, _yield_empty=False):
		"""
		Map/Reduce a list of functions over query results
		
		Starts up a MapReduce job, where blocks of rows returned by
		a query in each cell are yielded to the first kernel
		(callable) in the list (the mapper), whose outputs are
		passed on to the reducers in accordance with the MapReduce
		programming model (see
		http://mwscience.net/trac/wiki/LargeSurveyDatabase for an
		introduction).

		This method is a generator, and should be called from within
		a loop, such as:
		
		   >>> for res in query.execute([kernel1, kernel2]):
		           ... do something with res ...

		NOTE: There are a number of parameters to this function that
		      haven't been documented yet. Assume those are internal
		      to the code and should not be used (their meaning may
		      change).

		Parameters
		----------
		kernels : list
		    A list of "kernels" to execute. Each kernel is either a
		    callable (e.g., a Python function or an object with
		    an overloaded __call__ method), or a tuple of the form:
		    
		        (callable, arg2, arg3, arg4, ...)

		    The MapReduce engine, running within each LSD cell, will
		    call the first callable ("the mapper") as follows:
		    
		        callable(qresult, arg2, arg3, arg4, ...)
		        
		    where arg2, ... will exist only if given in the tuple as
		    show above.
		    
		    The qresult object, passed as the first argument, is a
		    generator yielding blocks of rows that are the (partial)
		    results of the query. They are instances of ColGroup
		    class (with an interface compatible with that of a
		    structured ndarray).
		    
		    The mapper must process the yielded blocks and yield
		    back the outputs to be returned to the user, or
		    subsequent kernels. Note that results must always be
		    yielded, and never returned.
		    
		    If more than one kernel is present in the list, all but
		    the final one are required to yield tuples of the form
		    (key, value), which will be transformed in accordance
		    with the MapReduce model before being passed on to
		    subsequent kernels.
		    
		    All but the first kernel will be called as:

		        callable(kv, arg2, arg3, arg4, ...)
		        
		    where the first argument, kv, is a pair of the form:
		    
		        key, values = kv
		        
		    where the key is whatever was returned as key by the
		    previous kernel, and values is a generator that yields
		    values associated with that key, by the previous
		    kernels.
		    
		    The final kernel in the list may return any value; it
		    will be yielded back to the user, with no further
		    processing.

		bounds : list of (Polygon, intervalset) tuples
		    A list of space/time bounds to which to restrict the
		    query. The list is used to cull the list of all cells in
		    the database, to a subset that overlap the bounds, as
		    well as (unless testbounds=False) to cull the query
		    results within each cell before yielding them to the
		    kernel.

		testbounds : boolean
		    If set to false, the bounds will only be used to cull
		    the list of cells over which to launch the mappers, but
		    not to cull the query results within each cell. This is
		    useful if the mappers perform their own row culling
		    themselves (presumably, more efficient than the built-in
		    point-in-polygon test)

		include_cached : boolean
		    Whether to include the objects from the neighbor cache
		    into the results of a query. Unless you're performing
		    spatial correlations (e.g., nearest neighbor searches),
		    you likely want to leave this be False.
		
		group_by_static_cell : boolean
		    Each execution of a mapper by default operates on
		    exactly one table cell. If this flag is set to True, and
		    the table has a temporal component, the mapper will be
		    executed once for _all_ temporal cells within a spatial
		    cell (their data will be yielded to it with no
		    interruption). For more, see the discussion in
		    "Important notes"

		Important notes
		---------------
		    - Each execution of a mapper is guaranteed to operate on
		      one and only one cell, unless group_by_static=True in
		      which case rows from all temporal cells, belonging to
		      a common spatial cell, will be yielded to it.
		      
		      However, it is undefined, and mapper should make no
		      assumptions, on how many times blocks of results from within
		      a cell will be yielded to it. For smaller cells, it
		      the result may come in a single block: for larger
		      cells, there may be more than one block. To be
		      general, the mapper should always call qresult within
		      a loop, such as:
		      
		      	  for rows in qresult:
		      	  	... process the block of rows ...

		    - The mapper, the reducer, all of their arguments, and
		      all keys and values being yielded must be pickleable.
		      LSD internally pickles/unpickles these to transfer
		      them to nodes/processes/threads that do the actual
		      work on each cell.

		    - The keys must be comparable and hashable (nearly every
		      Python object is).
		"""
		partspecs = dict()

		# Add explicitly requested cells
		for cell_id in cells:
			partspecs[cell_id] = [(None, None)]

		# Add cells within bounds
		if len(cells) == 0 or bounds is not None:
			partspecs.update(self.qengine.root.get_cells(bounds, include_cached=include_cached))

		# Tell _mapper not to test spacetime boundaries if the user requested so
		if not testbounds:
			partspecs = dict([ (cell_id, [(None, None)]) for (cell_id, _) in partspecs.iteritems() ])

		# Reorganize cells to a per-static-cell basis, if requested
		if group_by_static_cell:
			pix = self.qengine.root.table.pix
			p2 = defaultdict(list)
			for cell_id, bounds in partspecs.iteritems():
				if pix.is_temporal_cell(cell_id):
					cell_id2 = pix.static_cell_for_cell(cell_id)
				else:
					cell_id2 = cell_id
				p2[cell_id2].append((cell_id, bounds))
			partspecs = p2

			# Resort by time, but make the static cell (if any) be at the end
			def order_by_time(part):
				cell_id, _ = part
				_, _, t = pix._xyt_from_cell_id(cell_id)
				if t == pix.t0:
					t = +np.inf
				return t
			for cell_id, parts in partspecs.iteritems():
				parts.sort(key=order_by_time)

			#for cell_id, parts in partspecs.iteritems():
			#	parts = [ pix._xyt_from_cell_id(cell_id)[2] for cell_id, _ in parts ]
			#	print cell_id, parts
			#exit()
		else:
			partspecs = dict([ (cell_id, [(cell_id, bounds)]) for (cell_id, bounds) in partspecs.iteritems() ])

		# Insert our feeder mapper into the kernel chain
		kernels = list(kernels)
		kernels[0] = (_mapper, kernels[0], self.qengine, include_cached)

		# Append a writer mapper if the query has an INTO clause
		if self.qwriter:
			kernels.append((_into_writer, self.qwriter))

		# start and run the workers
		peer_directory = os.getenv("PYMR", None)
		if peer_directory is None:
			pool = pool2.Pool(nworkers)
		else:
			pool = mr.Pool(peer_directory)
		yielded = False
		for result in pool.map_reduce_chain(partspecs.items(), kernels, progress_callback=progress_callback):
			yield result
			yielded = True

		# Yield an empty row, if requested
		# WARNING: This is NOT a flag designed for use by users -- it is only to be used from .fetch()!
		if not yielded and _yield_empty:
			if self.qwriter:
				yield 0, self.qwriter.peek()
			else:
				yield 0, self.qengine.peek()

		# Shut down the workers
		del pool

	def iterate(self, bounds=None, include_cached=False, cells=[], return_blocks=False, filter=None, testbounds=True, nworkers=None, progress_callback=None, _yield_empty=False):
		"""
		Yield query results row-by-row or in blocks

		Executes the query, and yields back the results as they
		become available. The results are yielded either row by row
		(if return_blocks=False, the default), or in blocks of rows
		(structured ndarrays, if return_blocks=True).  While the
		former seems natural and likely more convenient, the latter
		is significantly more efficient.

		Calls 'filter' callable (if given) to filter the returned
		rows.  The filter callable must respect the rules for a
		'mapper' kernel (the first kernel), as discussed in the
		documentation for Query.execute(). In particular, it should
		be a generator, and expect a qresult as first argument. If
		the filter expects extra argument, pass it and its arguments
		as a tuple (e.g., filter=(filtercallable, arg2, arg3, ...)).

		See the documentation of Query.execute() for a description of
		other parameters.

		Example
		-------
		An identity filter:

		    def identity(qresult):
		    	for rows in qresult:
		    		yield rows

		A kernel filtering on column 'r' may look like:

		    def r_filter(qresult):
		    	for rows in qresult:
		    		yield rows[rows['r'] < 21.5]
		   
		"""

		mapper = filter if filter is not None else _iterate_mapper

		for (cell_id, rows) in self.execute(
				[mapper], bounds, include_cached,
				cells=cells, testbounds=testbounds, nworkers=nworkers, progress_callback=progress_callback,
				_yield_empty=_yield_empty):
			if return_blocks:
				yield rows
			else:
				for row in rows:
					yield row

	def fetch(self, bounds=None, include_cached=False, cells=[], filter=None, testbounds=True, nworkers=None, progress_callback=None):
		"""
		Returns a table (a ColGroup instance) with query results.

		As opposed to Query.iterate and Query.execute, Query.fetch
		blocks until the query is completely executed and returns
		the results in a single ColGroup instance. This is
		convenient to collect results of smaller queries.

		See Query.iterate() and Query.execute() for descriptions of
		various parameters.
		"""

		return colgroup.fromiter(
				self.iterate(
					bounds, include_cached, cells=cells,
					return_blocks=True, filter=filter, _yield_empty=True,
					nworkers=nworkers, progress_callback=progress_callback
					),
				blocks=True
			)

	def fetch_cell(self, cell_id, include_cached=False):
		""" Internal: Execute the query on a given (single) cell.

		    Does not launch extra workers, nor does it show the
		    progress bar. Only to be used internally, not a part of
		    the public API.
		"""
		return self.fetch(cells=[cell_id], include_cached=include_cached, nworkers=1, progress_callback=pool2.progress_pass)

class DB(object):
	"""
	The interface to LSD databases

	DB objects represent LSD databases. Once instantiated, they can me
	used to create new queries, tables and joins, as well as perform
	house-keeping on the database.
	"""
	path = None	#: Path to the root directory of the database, where all tables reside

	def __init__(self, path):
		"""
		Opens an existing database.
		
		Opens a database residing in directory 'path'. The directory
		must exist (an exception is thrown if it doesn't).
		
		Tip: To create a new, empty, database, simply make a new
		     empty directory.
		"""
		self.path = path.split(':')

		for path in self.path:
			if not os.path.isdir(path):
				raise Exception('"%s" is not an acessible directory.' % (path))

		self.tables = dict()	# A cache of table instances

	@contextmanager
	def lock(self, retries=-1):
		"""
		Lock the entire database for table or join creation/removal
		operations. If more than one database directory exists in
		self.path, the lock is created in the first one.

		TODO: Only DB.create_into_tables() obeys this lock so far.
		      Make other ops that create tables obey it as well.
		"""
		# create a lock file
		lockfile = self.path[0] + '/dblock.lock'
		utils.shell('/usr/bin/lockfile -1 -r%d "%s"' % (retries, lockfile) )

		# yield self
		yield self

		# unlock
		os.unlink(lockfile)

	### URI (Uniform Resource Identifier) management routines
	def resolve_uri(self, uri):
		""" Resolve the given URI into something that can be
		    passed on to file() for loading.
		"""
		if uri[:4] != 'lsd:':
			return uri

		_, tabname, _ = uri.split(':', 2)
		return self.table(tabname).resolve_uri(uri)

	@contextmanager
	def open_uri(self, uri, mode='r', clobber=True):
		if uri[:4] != 'lsd:':
			# Interpret this as a general URL
			import urllib

			f = urllib.urlopen(uri)
			yield f

			f.close()
		else:
			# Pass on to the controlling table
			_, tabname, _ = uri.split(':', 2)
			with self.table(tabname).open_uri(uri, mode, clobber) as f:
				yield f

	def query(self, query, locals={}):
		"""
		Constructs and returns a Query object.
		
		Constructs a query object given an LSD query string, and
		potentially a dictionary of objects ('locals') that are to
		be made available to the code within the query. For example:

		>>> def Ar(ra, dec):
		       ... do something to compute extinction ...
		       return ext_r
		>>> db.query("SELECT mag_r + Ar(ra, dec) FROM sometable", {'Ar': Ar})
		
		"""
		return Query(self, query, locals=locals)

	def _aux_create_table(self, table, tname, schema):
		schema = copy.deepcopy(schema)

		# Remove any extra fields off schema['column']
		schema['columns'] = [ v[:2] for v in schema['columns'] ]

		table.create_cgroup(tname, schema)

	def create_table(self, tabname, tabdef):
		"""
		Creates a table, given the LSD Table Definition.

		The LSD Table Definition (LSD-TD) is a dictionary of entries
		describing the complete layout of a table. This is as
		close as LSD gets to a Data Definition Language (DDL).
		
		TODO: Fully describe LSD-TD in a separate document. For now,
		look into lsd.smf and lsd.sdss modules to see some examples
		of table definitions.
		"""
		table = self.table(tabname, create=True)

		# Add fgroups
		if 'fgroups' in tabdef:
			for fgroup, fgroupdef in tabdef['fgroups'].iteritems():
				table.define_fgroup(fgroup, fgroupdef)

		# Add filters
		table.set_default_filters(**tabdef.get('filters', {}))

		# Add column groups
		tschema = tabdef['schema']

		# Ensure we create the primary table first
		schemas = []
		for tname, schema in tschema.iteritems():
			schemas.insert(np.where('primary_key' in schema, 0, len(schemas)), (tname, schema))
		assert len(schemas) and 'primary_key' in schemas[0][1]
		for tname, schema in schemas:
			self._aux_create_table(table, tname, schema)

		# Add aliases (must do this last, as the aliased columns have to exist)
		if 'aliases' in tabdef:
			for alias, colname in tabdef['aliases'].iteritems():
				table.define_alias(alias, colname)

		return table

	def define_join(self, name, type, **joindef):
		"""
		Define how two tables are joined.

		Creates a .join file with the name '<name>.join', containing
		information from joindef on how to join two tables. The
		contents of joindef depends on the type of the join (the
		argument type).
		
		If name has the form '.<left_table>:<right_table>', this
		join will be automatically found by LSD when looking how to
		join two tables, left_table and right_table, found in the
		FROM clause of a query. NOTE: For forward compatibility,
		define such joins using DB.define_default_join()
	
		If more than one database directory exists in self.path,
		the .join file is defined in the first one.

		type=indirect
		-------------
		If type == 'indirect', the join being defined is an
		"indirect join". It is roughly equivalent to the following
		SQL statement:
		
		    SELECT ... FROM R
		    [OUTER] JOIN indir ON indir.m1 = R.id
		    [OUTER] JOIN S     ON indir.m2 = S.id
		
		where R and S are the two tables being joined, and indir is
		the indirection table. For this case, joindef specifys where
		to find indir.m1 and indir.m2 columns.

		For type=indirect, joindef must contain:
		
		    "m1" : ("indir1", "m1")
		    "m2" : ("indir2", "m2")

		Note that while m1 and m2 are not required to be in the same
		table (which SQL does require, and makes for a good practice
		anyways), they are required to be of the same length.

		Note that simpler JOINs are a subset of indirect joins. For
		example:
		
		    SELECT ... FROM R
		    JOIN S ON S.id = R.id

                is equivalent to:
                
                    "m1" : ("R", "id")
                    "m2" : ("R", "id")

                Or:
		
		    SELECT ... FROM R
		    JOIN S ON S.id = R.s_id

                is equivalent to:
                
                    "m1" : ("R", "id")
                    "m2" : ("R", "exp_id")

                assuming id are the primary keys of R and S, and (in the
                latter example), exp_id is the foreign key.
		"""
		#- .join file structure:
		#	- indirect joins:			Example: ps1_obj:ps1_det.join
		#		type:	indirect		"type": "indirect"
		#		m1:	(tab1, col1)		"m1:":	["ps1_obj2det", "id1"]
		#		m2:	(tab2, col2)		"m2:":	["ps1_obj2det", "id2"]
		#	- equijoins:				Example: ps1_det:ps1_exp.join		(!!!NOT IMPLEMENTED!!!)
		#		type:	equi			"type": "equijoin"
		#		id1:	colA			"id1":	"exp_id"
		#		id2:	colB			"id2":	"exp_id"
		#	- direct joins:				Example: ps1_obj:ps1_calib.join.json	(!!!NOT IMPLEMENTED!!!)
		#		type:	direct			"type": "direct"

		fname = '%s/%s.join' % (self.path[0], name)
		if os.access(fname, os.F_OK):
			raise Exception('Join relation %s already exist (in file %s)' % (name, fname))

		joindef['type'] = type
	
		f = open(fname, 'w')
		f.write(json.dumps(joindef, indent=4, sort_keys=True))
		f.close()

	def define_default_join(self, tableA, tableB, type, **joindef):
		"""
		Define a default join relation between two tables.
		
		See documentation for DB.define_join() for details on type
		and joindef arguments.
		"""
		return self.define_join('.%s:%s' % (tableA, tableB), type, **joindef)

	def table_exists(self, tabname):
		"""
		Test whether a table exists
		
		Returns:
		--------
		True or False
		"""
		try:
			self.table(tabname)
			return True
		except IOError:
			return False

	def table(self, tabname, create=False):
		"""
		Returns a Table instance for a given table name.

		If create=True, creates a new empty table. If the table
		being created already exists, throws an exception.
		
		NOTE: There should usually be no need to access tables
		      directly, though a Table instance (use queries for
		      that).
		"""
		if tabname not in self.tables:
			if create:
				path = '%s/%s' % (self.path[0], tabname)
				self.tables[tabname] = Table(path, name=tabname, mode='c')
			else:
				# Find the table. Allow the database to be specified
				# using db.tablename syntax
				if tabname.find('.') != -1:
					dbdir, tn = tabname.split('.')
				else:
					dbdir, tn = None, tabname

				for dbpath in self.path:
					if dbdir is not None:
						dbpath = '/'.join(dbpath.split('/')[:-1] + [ dbdir ])
					path = '%s/%s' % (dbpath, tn)
					if os.path.isdir(path):
						self.tables[tabname] = Table(path)
						break
				else:
					raise IOError("Table %s not found in %s" % (tabname, ':'.join(self.path)))
		else:
			assert not create

		return self.tables[tabname]

	def construct_join_tree(self, from_clause):
		"""
		Internal: Figure out how to join tables in a query

		This is an internal function; do not use it.
		"""
		tablist = []
		tables_by_path = {}

		# Find the tables involved in the query and instantiate them
		for tabname, tabpath, join_args in from_clause:
			entry = TableEntry(self.table(tabpath), tabname)
			te = ( tabname, (entry, join_args[0]) )

			tablist.append(te)
			tables_by_path[entry.table.path] = te[1]
		tables = dict(tablist)

		# Discover and set up JOIN links based on defined JOIN relations
		# TODO: Expand this to allow the 'joined to via' and related syntax, once it becomes available in the parser
		for tabname, (e, jargs) in tablist:
			# Check for 'matchedto' keys in join_args
			if 'matchedto' in jargs:
				# Test query:
				# NWORKERS=1 ./lsd-query "select ra, dec, galex_gr5.ra, galex_gr5.dec, sdss._NR, sdss._DIST*3600 from galex_gr5, sdss(matchedto=galex_gr5,nmax=5,dmax=15)" | head -n 20
				assert e.relation is None
				entry, _ = tables[jargs['matchedto']]
				e.relation = create_join(self, None, jargs, entry.table, e.table, jclass=CrossmatchJoin)
				##e.relation = create_join(self, "/n/pan/mjuric/db/.galex_gr5:sdss.join", jargs, entry.table, e.table)
				entry.joins.append(e)
			else:
				# Check for tables that can be joined onto this one (where this one is on the right hand side of the relation)
				# Look for default .join files named ".<tabname>:*.join"
				dbpath = os.path.dirname(e.table.path)	# Look only in the table's dbdir

				pattern = "%s/.%s:*.join" % (dbpath, e.table.name)
				for fn in glob.iglob(pattern):
					jtabname = fn[fn.rfind(':')+1:fn.rfind('.join')]
					jpath = "%s/%s" % (dbpath, jtabname)

					try:
						je, jargs = tables_by_path[jpath]
					except KeyError:
						continue

					if 'matchedto' in jargs:	# Explicitly joined
						continue
					if je.relation is not None:	# Already joined
						continue

					je.relation = create_join(self, fn, jargs, e.table, je.table)
					e.joins.append(je)

		# Discover the root (the one and only one table that has no links pointing to it)
		root = None
		for _, (e, jargs) in tables.iteritems():
			if e.relation is None:
				assert root is None		# Can't have more than one roots
				assert 'outer' not in jargs	# Can't have something like 'ps1_obj(outer)' on root table
				root = e
		assert root is not None			# Can't have zero roots
	
		# TODO: Verify all tables are reachable from the root (== there are no circular JOINs)
	
		# Return the root of the tree, and a dict of all the Table instances
		return root, dict((  (tabname, table) for (tabname, (table, _)) in tables.iteritems() ))

	def build_neighbor_cache(self, tabname, margin_x_arcsec=30):
		""" 
		(Re)Build the neighbor cache in a given table.
		
		Cache the objects found within margin_x (arcsecs) of each
		cell into neighboring cells, to support efficient
		nearest-neighbor lookups.

		Parameters
		----------
		tabname : string
		    The name of the table
		margin_x_arcsec : number
		    The margin (in arcseconds) which to cache into
		    neighboring cells
		    
		TODO: The implementation of this is pretty and inefficient.
		      It should be nearly completely rewritten at some
		      point.
		"""
		# This routine works in tandem with _cache_maker_mapper and
		# _cache_maker_reducer auxilliary routines.
		margin_x = np.sqrt(2.) / 180. * (margin_x_arcsec/3600.)

		ntotal = 0
		ncells = 0
		query = "_ID, _LON, _LAT FROM %s" % (tabname)
		for (_, ncached) in self.query(query).execute([
						(_cache_maker_mapper,  margin_x, self, tabname),
						(_cache_maker_reducer, self, tabname)
					]):
			ntotal = ntotal + ncached
			ncells = ncells + 1
			#print self._cell_prefix(cell_id), ": ", ncached, " cached objects"
		print "Total %d cached objects in %d cells" % (ntotal, ncells)

	def compute_summary_stats(self, *tables):
		"""
		Compute summary statistics.
		
		Compute and store frequently used summary statistics. This
		should be called to refresh the stats after insertions or
		larger modifications to tables.
		
		It:
		    - counts and stores the number of rows in a table
		    - constructs and stores a cache to speed up query
		      startup (TabletTreeCache objects)

		Parameters
		----------
		*tables: strings
		    One or more names of tables to operate on.

		TODO: Better document *when* this function should be caller
		      (or better yet, do away with it alltogether).
		"""
		from tasks import compute_counts
		from fcache import TabletTreeCache
		for tabname in tables:
			table = self.table(tabname)

			# Compute the tablet tree cache
			data_path = table._get_cgroup_data_path(table.primary_cgroup)
			pattern   = table._tablet_filename(table.primary_cgroup)
			table.tablet_tree = TabletTreeCache().create_cache(table.pix, data_path, pattern, table.path + '/tablet_tree.pkl')

			# Scan the number of rows
			table._nrows = compute_counts(self, tabname)
			table._store_schema()

###################################################################
## Auxilliary functions implementing DB.build_neighbor_cache
## functionallity
def _cache_maker_mapper(qresult, margin_x, db, tabname):
	# Map: fetch all rows to be copied to adjacent cells,
	# yield them keyed by destination cell ID
	for rows in qresult:
		cell_id = rows.info.cell_id
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
		q           = db.query("* FROM '%s' WHERE np.in1d(_ID, idsInMargin, assume_unique=True)" % tabname, {'idsInMargin': idsInMargin} )
		data        = q.fetch_cell(cell_id)

		# Send these to neighbors
		if data is not None:
			for neighbor in qresult.pix.neighboring_cells(cell_id):
				yield (neighbor, data)

		##print "Scanned margins of %s*.h5 (%d objects)" % (db.table(tabname)._cell_prefix(cell_id), len(data))

def _cache_maker_reducer(kv, db, tabname):
	# Cache all rows to be cached in this cell
	cell_id, rowblocks = kv

	# Delete existing neighbors
	table = db.table(tabname)
	table.drop_row_group(cell_id, 'cached')

	# Add to cache
	ncached = 0
	for rows in rowblocks:
		table.append(rows, cell_id=cell_id, group='cached')
		ncached += len(rows)
		##print cell_id, len(rows), table.pix.path_to_cell(cell_id)

	# Return the number of new rows cached into this cell
	yield cell_id, ncached

###############################################################
# Aux. functions implementing Query.iterate() and
# Query.fetch()
class TableProxy:
	coldict = None
	prefix = None

	def __init__(self, coldict, prefix):
		self.coldict = coldict
		self.prefix = prefix

	def __getattr__(self, name):
		return self.coldict[self.prefix + '.' + name]

def _mapper(partspec, mapper, qengine, include_cached):
	(group_cell_id, cell_list) = partspec
	mapper, mapper_args = utils.unpack_callable(mapper)

	# Pass on to mapper (and yield its results)
	qresult = qengine.on_cells(cell_list, include_cached)
	for result in mapper(qresult, *mapper_args):
		yield result

def _iterate_mapper(qresult):
	for rows in qresult:
		if len(rows):	# Don't return empty sets. TODO: Do we need this???
			yield (rows.info.cell_id, rows)

def _into_writer(kw, qwriter):
	cell_id, irows = kw
	for rows in irows:
		rows = qwriter.write(cell_id, rows)
		yield rows

###############################

def _fitskw_dumb(hdrs, kw):
	# Easy way
	import StringIO
	res = []
	for ahdr in hdrs:
		hdr = pyfits.Header( txtfile=StringIO(ahdr) )
		res.append(hdr[kw])
	return res

def fits_quickparse(header):
	"""
	An ultra-simple FITS header parser.
	
	Does not support CONTINUE statements, HIERARCH, or anything of the
	sort; just plain vanilla:
	
	    	key = value / comment

	one-liners. The upshot is that it's fast, much faster than the
	PyFITS equivalent.

	NOTE: Assumes each 80-column line has a '\n' at the end (which is
	      how we store FITS headers internally.)
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
	"""
	Intelligently extract a keyword kw from an arbitrarely
	shaped object ndarray of FITS headers.
	
	Designed to be called from within LSD queries.
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
		global _DB
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

def bin(v):
	"""
	Similar to __builtin__.bin but works on ndarrays.
	
	Useful in queries for converting flags to bit strings.

	FIXME: The current implementation is painfully slow.
	"""
	import __builtin__
	if not isinstance(v, np.ndarray):
		return __builtin__.bin(v)

	# Must be some kind of integer
	assert v.dtype.kind in ['i', 'u']

	# Create compatible string array
	l = v.dtype.itemsize*8
	ss = np.empty(v.shape, dtype=('a', v.dtype.itemsize*9))
	s = ss.reshape(-1)
	for i, n in enumerate(v.flat):
		c = __builtin__.bin(n)[2:]
		c = '0'*(l-len(c)) + c
		ll = [ c[k:k+8] for k in xrange(0, l, 8) ]
		s[i] = ','.join(ll)
		#s[i] = c
	return ss

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

		import bounds
		query = "_ID, ps1_det._ID, ps1_exp._ID, sdss._ID FROM '../../../ps1/sdss'(outer) as sdss, '../../../ps1/ps1_obj' as ps1_obj, '../../../ps1/ps1_det' as ps1_det, '../../../ps1/ps1_exp'(outer) as ps1_exp WHERE True"
		query = "_ID, ps1_det._ID, ps1_exp._ID, sdss._ID FROM sdss(outer), ps1_obj, ps1_det, ps1_exp(outer)"
		cell_id = np.uint64(6496868600547115008 + 0xFFFFFFFF)
		include_cached = False
		bounds = [(bounds.rectangle(120, 20, 160, 60), None)]
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
