#!/usr/bin/env python

import StringIO
import tokenize

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
	try:
		for (id, token, _, _, _) in g:
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
					(_, name, _, _, _) = next(g)
					(_, token, _, _, _) = next(g)
				else:
					name = col
				newcols = [(name, col)]

			# Column delimiter or end of SELECT clause
			if token.lower() in ['', ',', 'from']:
				select_clause += newcols
				if token.lower() == "from":
					# FROM clause
					while token.lower() not in ['', 'where', 'into']:
						# Slurp the table path
						(_, table, _, _, _) = next(g)				# table path

						if table and table[0] in ['"', "'"]:			# Unquote if quoted
							table = table[1:-1]

						# At this point we expect:
						# ... [EOL] # <-- end of line
						# ... WHERE
						# ... (inner/outer)
						# ... AS asname
						# ... (inner/outer) AS asname
						join_type = 'inner'					# default JOIN type
						astable = table
						for _ in xrange(2):
							(_, token, _, _, _) = next(g)
							if token == '(':
								(_, join_type, _, _, _) = next(g)	# inner/outer
								(_, token, _, _, _) = next(g)		# )
							elif token.lower() == 'as':			# table rename
								(_, astable, _, _, _) = next(g)
								(_, token, _, _, _) = next(g)		# next token
								break
							elif token.lower() in ['', ',', 'where']:
								break

						from_clause += [ (astable, table, join_type) ]

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
						dtype = into_col = keyexpr = None
						kind = 'append'

						# Look for explicit dtype in parenthesis
						if token == '(':
							dtype = ''
							(_, token, _, _, _) = next(g)
							while token not in [')']:
								dtype += token
								(_, token, _, _, _) = next(g)

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

						into_clause = (table, dtype, into_col, keyexpr, kind)
					
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
			ret.extend(( (col, col) for col in tablecols[''] ))
		elif len(col) > 2 and col[-2:] == '.*':
			tbl = col[:-2]
			ret.extend(( (tbl+'.'+col, tbl+'.'+col) for col in tablecols[tbl] ))
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
#	(select_clause, where_clause, from_clause, into_clause) = parse("* from exp where _TIME < 55248.25 into exp2");
#	(select_clause, where_clause, from_clause, into_clause) = parse("*, sdss.* FROM '/w sdss' as sx WHERE aa == bb INTO blabar(i4,f8) WHERE _ID == sdss._ID");
	(select_clause, where_clause, from_clause, into_clause) = parse("*, sdss.* FROM '/w sdss' as sx WHERE aa == bb INTO blabar(i4,f8)");
	print (select_clause, where_clause, from_clause, into_clause)
	print resolve_wildcards(select_clause, tablecols)
	exit()
	print parse("ra, dec");
	exit()
	print parse("sdss.*, ra, sdss.gic as a, dec as DEC, (where(sdss.dec,where,ra,dec)) as blarg FROM ps1, sdss(outer), tmass WHERE 0.1 < g-r < 0.5")
	exit()

	vd = VerboseDict()
	print eval('2+2*sdss.gic', {}, vd)
