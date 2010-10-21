#!/usr/bin/env python

import StringIO
import tokenize

def parse(query, tablecols):
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
			while token.lower() not in ['', ',', 'where', 'as']:
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
			if col == '*':
				newcols = [ (col, col) for col in tablecols[''] ]
			elif len(col) > 2 and col[-2:] == '.*':
				tbl = col[:-2]
				newcols = [ (tbl+'.'+col, tbl+'.'+col) for col in tablecols[tbl] ]
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
					# JOIN clause
					while token.lower() not in ['', 'where']:
						(_, table, _, _, _) = next(g)			# table name

						(_, token, _, _, _) = next(g)
						if token == '(':
							(_, join_type, _, _, _) = next(g)	# inner/outer
							(_, token, _, _, _) = next(g)		# )
							(_, token, _, _, _) = next(g)
						else:
							join_type = 'inner'			# default JOIN type

						from_clause += [ (table, join_type) ]

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

if __name__ == '__main__':
	class VerboseDict:
		def __getitem__(self, key):
			print "Requested key: " + key
			return None

	tablecols = {
		'': ['a', 'b', 'c'],
		'sdss': ['haha', 'hihi']
	}
	print parse("*, sdss.*", tablecols);
	exit()
	print parse("ra, dec", tablecols);
	exit()
	print parse("sdss.*, ra, sdss.gic as a, dec as DEC, (where(sdss.dec,where,ra,dec)) as blarg FROM ps1, sdss(outer), tmass WHERE 0.1 < g-r < 0.5", tablecols)
	exit()

	vd = VerboseDict()
	print eval('2+2*sdss.gic', {}, vd)
