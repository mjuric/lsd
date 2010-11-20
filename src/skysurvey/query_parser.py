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
			if token.lower() in ['', ',', 'where', 'from']:
				select_clause += newcols
				if token.lower() == "from":
					# FROM clause
					while token.lower() not in ['', 'where']:
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
						for _dummy in xrange(2):
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

				if token.lower() == 'where':
					# WHERE clause
					where_clause = ''
					while token != '':
						(_, token, _, _, _) = next(g)
						where_clause = where_clause + token
					break
				else:
					continue

			raise Exception('Syntax error')
	except list as dummy:
	#except StopIteration:
		pass

	return (select_clause, where_clause, from_clause)

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
	(select_clause, where_clause, from_clause) = parse("*, sdss.* FROM '/w sdss' as sx WHERE aa == bb");
	print (select_clause, where_clause, from_clause)
	print resolve_wildcards(select_clause, tablecols)
	exit()
	print parse("ra, dec");
	exit()
	print parse("sdss.*, ra, sdss.gic as a, dec as DEC, (where(sdss.dec,where,ra,dec)) as blarg FROM ps1, sdss(outer), tmass WHERE 0.1 < g-r < 0.5")
	exit()

	vd = VerboseDict()
	print eval('2+2*sdss.gic', {}, vd)
