#!/usr/bin/env python

import StringIO
import tokenize

valid_keys_from = frozenset(['nmax', 'dmax', 'inner', 'outer', 'xmatch', 'matchedto'])
valid_keys_into = frozenset(['spatial_keys', 'temporal_key', 'dtype', 'no_neighbor_cache'])

def unquote(s):
	# Unquote if quoted
	if s and s[0] in frozenset(['"', "'"]):
		return s[1:-1]
	return s

def parse_args(g, token, valid_keys):
	args = dict()
	while token != ')':
		key = next(g)[1].lower()	# key=value or key or )
		if key == ')':
			break
		if key not in valid_keys:
			raise Exception("Unknown keyword '%s' found in table arguments" % key)
		token = next(g)[1].lower()
		if token == '=':
			# Slurp up the value
			val = next(g)[1]
			token = next(g)[1]

			# Allow constructs such as: spatial_keys=(ra, dec)
			# and parse these into a list
			if val in ['(', '[']:
				# List of values
				end_token = ')' if val == '(' else ']'
				val = []
				v = []
				while token != end_token:
					if token != ',':
						v.append(token)
					else:
						val.append(' '.join(v))
						v = []
					token = next(g)[1]
				if v:
					val.append(' '.join(v))

				token = next(g)[1]
				assert token in [',', ')'], token
			else:
				while token not in set([')', ',', '']):
					val += token
					token = next(g)[1]

		else:
			val = None
			assert token in [',', ')']

		args[key] = unquote(val) if not isinstance(val, list) else [ unquote(v) for v in val ]

	return args, token

def parse(query):
	""" 
	    Parse query of the form:

	    ra, dec, u , g, r, sdss.u, sdss.r, tmass.*, func(ra,dec) as xx WHERE (expr)
	"""
	g = tokenize.generate_tokens(StringIO.StringIO(query).readline)
	where_clause = 'True'
	select_clause = []
	from_clause = []
	into_clause = None
	first = True
	try:
		for (id, token, _, _, _) in g:
			if first: # Optional "SELECT"
				first = False
				if token.lower() == "select":
					continue

			if id == tokenize.ENDMARKER:
				break

			col = ''
			while token.lower() not in ['', ',', 'where', 'as', 'from']:
				col = col + token
				if token == '(':
					# find matching ')'
					pt = 1
					while pt:
						(_, token, _, _, _) = next(g)
						if token == '(': pt = pt + 1
						if token == ')': pt = pt - 1
						col = col + token
				(_, token, _, _, _) = next(g)

			if col == '':
				raise Exception('Syntax error')

			# Parse column for the simple case of col='*' or col='table.*'
			if col == '*' or len(col) > 2 and col[-2:] == '.*':
				# wildcards
				tbl = col[:-2]
				newcols = [ (col, col) ]
			else:
				# as token is disallowed after wildcards
				if token.lower() == 'as':
					# expect:
					# ... as COLNAME
					# ... as (COL1, COL2, ...)
					(_, name, _, _, _) = next(g)
					if name == '(':
						token = ','
						names = []
						while token != ')':
							assert token == ','
							(_, name, _, _, _) = next(g)	# Expect the column name
							names.append(name)
							(_, token, _, _, _) = next(g)	# Expect , or ')'
					else:
						names = [ name ]
					(_, token, _, _, _) = next(g)
				else:
					names = [ col ]
				newcols = [(names, col)]

			# Column delimiter or end of SELECT clause
			if token.lower() in ['', ',', 'from']:
				select_clause += newcols
				if token.lower() == "from":
					# FROM clause
					while token.lower() not in ['', 'where', 'into']:
						# Slurp the table path, allowing for db.tabname constructs
						(_, table, _, _, _) = next(g)				# table path
						token = next(g)[1]
						if token == '.':
							table += '.' + next(g)[1]
							token = next(g)[1]
						table = unquote(table)

						# At this point we expect:
						# ... [EOL] # <-- end of line
						# ... WHERE
						# ... (inner/outer)
						# ... AS asname
						# ... (inner/outer) AS asname
						join_args = []
						astable = table
						for _ in xrange(2):
							if token == '(':
								args, token = parse_args(g, token, valid_keys_from)
								if 'inner' in args and 'outer' in args:
									raise Exception('Cannot simultaneously have both "inner" and "outer" as join type')
								if len(args):
									join_args.append(args)
							elif token.lower() == 'as':			# table rename
								(_, astable, _, _, _) = next(g)
								(_, token, _, _, _) = next(g)		# next token
								break
							elif token.lower() in ['', ',', 'where', 'into']:
								break

							(_, token, _, _, _) = next(g)

						if not join_args:
							join_args.append(dict())

						from_clause += [ (astable, table, join_args) ]

					# WHERE clause (optional)
					if token.lower() == 'where':
						# WHERE clause
						where_clause = ''
						(_, token, _, _, _) = next(g)
						while token.lower() not in ['', 'into']:
							where_clause = where_clause + token
							(_, token, _, _, _) = next(g)

					# INTO clause (optional)
					if token.lower() == 'into':
						(_, table, _, _, _) = next(g)
						(_, token, _, _, _) = next(g)
						into_col = keyexpr = None
						into_args = {}
						kind = 'append'

						# Look for explicit into_args in parenthesis
						if token == '(':
							into_args, token = parse_args(g, token, valid_keys_into)
							#dtype = ''
							#(_, token, _, _, _) = next(g)
							#while token not in [')']:
							#	dtype += token
							#	(_, token, _, _, _) = next(g)

							(_, token, _, _, _) = next(g)

						# Look for WHERE xx = expr clause (update key specification)
						# or for AT idexpr clause (insert with given IDs)
						if token.lower() in ['where', 'at']:
							if token.lower() == 'where':
								# WHERE xx = expr construct
								(_, into_col, _, _, _) = next(g)	# column against which to mach in the INTO table
								(_, token, _, _, _) = next(g)	# must be '='
								if token == '==':
									kind = 'update/ignore'	# update if exists, ignore otherwise
								elif token == '|=':
									kind = 'update/insert' # update if exists, insert otherwise
								else:
									raise Exception('Syntax error in INTO clause near "%s" (expected "==")', token)
							else:
								# AT expr construct
								into_col = '_ID'
								kind = 'insert'

							# slurp up everything to the end -- this will be the expr giving the keys
							tokens = []
							while token != '':
								(_, token, _, _, _) = next(g)
								tokens.append(token)
							keyexpr = ''.join(tokens)

						into_clause = (table, into_args, into_col, keyexpr, kind)
					
					if token != '':
						raise Exception('Syntax error near "%s"', token)

					break
	except list as dummy:
	#except StopIteration:
		pass

	return (select_clause, where_clause, from_clause, into_clause)

def resolve_wildcards(select_clause, tablecols):
	# Resolve all .* columns, given a dict-like variable
	# tablecols that should return a list of columns
	# that exist in a given table. tablecols['']
	# must return the list of columns from the root table
	ret = []
	for ascol, col in select_clause:
		if col == '*':
			# Construct a list of all columns from all tables, prefixing
			# some with their table name to avoid ambiguity.
			# Note that the first table returned by tablecols.keys() is
			# assumed to be the root table (its columns will never get
			# prefixed)
			selcols = set()
			for tbl in tablecols.keys():
				for col in tablecols[tbl]:
					if col in selcols:
						c = tbl+'.'+col
					else:
						c = col
						selcols.add(c)
					ret.append(([c],c))
		elif len(col) > 2 and col[-2:] == '.*':
			tbl = col[:-2]
			ret.extend(( ([tbl+'.'+col], tbl+'.'+col) for col in tablecols[tbl] ))
		else:
			ret.append((ascol, col))
	return ret

if __name__ == '__main__':
	class VerboseDict:
		def __getitem__(self, key):
			print "Requested key: " + key
			return None

	tablecols = {
		'': ['a', 'b', 'c'],
		'sdss': ['haha', 'hihi']
	}
#	print parse("sdss.ra as ra, sdss.dec FROM sdss AS s")
#	exit()
	(select_clause, where_clause, from_clause, into_clause) = parse("_ from _ into exp2 where aa |= bb");
#	(select_clause, where_clause, from_clause, into_clause) = parse("* from exp where _TIME < 55248.25 into exp2");
#	(select_clause, where_clause, from_clause, into_clause) = parse("*, sdss.* FROM '/w sdss' as sx WHERE aa == bb INTO blabar(i4,f8) WHERE _ID == sdss._ID");
#	(select_clause, where_clause, from_clause, into_clause) = parse("*, sdss.* FROM '/w sdss' as sx WHERE aa == bb INTO blabar(i4,f8)");
	print (select_clause, where_clause, from_clause, into_clause)
	print resolve_wildcards(select_clause, tablecols)
	exit()
	print parse("ra, dec");
	exit()
	print parse("sdss.*, ra, sdss.gic as a, dec as DEC, (where(sdss.dec,where,ra,dec)) as blarg FROM ps1, sdss(outer), tmass WHERE 0.1 < g-r < 0.5")
	exit()

	vd = VerboseDict()
	print eval('2+2*sdss.gic', {}, vd)
