#!/usr/bin/env python
"""
A simple (incomplete) parser to parse tables from SDSS' .par files

See http://www.sdss.org/dr7/data/parfiles.html for file format description.

Limitations:
	- All files must have exactly one typedef struct {} definition
	- All records must be of the same type
	- enums are not supported
"""

import numpy as np
import re
from itertools import izip

EXPECT_TYPEDEF = 0
DEFINING_STRUCT = 1
EXPECT_DATA = 2

def str_conv(s):
	if s[:1] in [ '"', "'"] :
		return s[1:-1]
	return s

type_map = { 'int': 'i8', 'double': 'f8', 'char': 'a', 'float': 'f4', 'short': 'i2' }
type_conv = { 'int': int, 'double': float, 'char': str_conv, 'float': float, 'short': int }
pat = r'''("[^"]*"|'[^']*'|[^ \t]+)'''	# Pattern matching a whitespace-separated string, potentially enclosed in single or double quotes

def read(fp):
	"""
		Read an SDSS .par file. See the documentation of this module
		for the limitations of this function.
	"""
	state = EXPECT_TYPEDEF
	struct_name = None
	dtype = None
	fields = []
	field_splits = []
	rows = []

	line_cont = []
	for line in fp:
		line = line.split('#')[0].strip()
		if line == "":
			continue
		if line[-1] == '\\':
			line_cont.append(line[:-1])
			continue
		if line_cont:
			line_cont.append(line)
			line = ''.join(line_cont)
			line_cont = []

		if state == EXPECT_TYPEDEF:
			cannonical = ' '.join(s.lower() for s in line.split())
			if cannonical != 'typedef struct {':
				raise Exception("Only know how to parse files that have definitions of the form 'typedef struct {'")
			state = DEFINING_STRUCT
		elif state == DEFINING_STRUCT:
			typestr, name = line.split(';')[0].split()

			if typestr == '}':
				struct_name = name
				dtype = np.dtype([(name, dtypestr) for (name, dtypestr, _, _) in fields])

				pats = [ r'\{\s*' + '\s+'.join([pat]*n) + r'\s*\}' if n > 1 else pat for (_, _, _, n) in fields ]
				restr = r'\s+'.join([pat] + pats)
				r = re.compile(restr)

				field_lens = [ n for (_, _, _, n) in fields ]
				field_offs = np.cumsum([0] + field_lens)
				field_splits = zip(field_offs, field_lens)
				n_fields = np.sum(field_lens)

				state = EXPECT_DATA
				continue

			# Map type to dtype string
			typ = type_map[typestr]

			# Check if it's an array
			if name[-1] == ']':
				name, tmp = name[:-1].split('[', 1)
				tmp = tmp.split('][')

				if tmp[-1] == '':
					if typ != 'a':
						raise Exception("Can't make an unbounded array of %s" % typestr)

					tmp[-1] = '256'

				tmp = [ int(x) for x in tmp ]
				if typ == 'a':
					typ = 'a%d' % tmp[-1]
					tmp = tmp[:-1]

				assert len(tmp) in [0, 1]
				n = int(tmp[0]) if tmp else 1
			else:
				n = 1

			dtypestr = str(n) + typ
			fields.append((name, dtypestr, type_conv[typestr], n))
		elif state == EXPECT_DATA:
			# Parse the line
			row = r.match(line).groups()

			# Some sanity checking
			if len(row) != n_fields + 1 or row[0] != struct_name:
				raise Exception("Error at line '%s'" % line)
			else:
				row = row[1:]

			# Reshape it (for array items)
			row = [ row[a:a+l] if l != 1 else row[a] for a, l in field_splits ]

			# Conversion (for array items)
			row2 = []
			for col, (name, _, conv_fun, n) in izip(row, fields):
				if n == 1:
					row2.append(conv_fun(col))
				else:
					row2.append(tuple(conv_fun(item) for item in col))

			rows.append(row2)

	# Convert to numpy array
	ret = np.zeros(len(rows), dtype=dtype)
	for k, row in enumerate(rows):
		for c, (name, _, _, n) in enumerate(fields):
			if n == 1:
				ret[k][name] = row[c]
			else:
				for j, item in enumerate(row[c]):
					ret[k][name][j] = item
	return ret

if __name__ == "__main__":
	#ret = read(open('opRunlist.par'))
	ret = read(open('/n/home06/mjuric/sources/opRunlist.par'))
	print ret[:5]
