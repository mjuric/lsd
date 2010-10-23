#!/usr/bin/env python
#
# Example: import_sdss.py -c sdss /raid14/sweeps/sdss3/2009-11-16.v2/301
#

import sys
import getopt
import skysurvey.sdss  as sdss
import skysurvey as lsd
from   skysurvey.utils import *

def usage():
	print "Usage: %s [-c|--create] <cat_dir> <sweep_file_dir>" % sys.argv[0]

try:
	optlist, args = getopt.getopt(sys.argv[1:], 'c', ['create'])
except getopt.GetoptError, err:
	print str(err)
	usage()
	exit(-1)

if len(args) != 2:
	print "Error: Not enough command line arguments."
	usage()
	exit(-1)

cat_dir, sweep_dir = args

create = False
for (o, a) in optlist:
	if o == '-c' or o == '--create':
		create = True

#
# Actual work
#
files = shell('find "' + sweep_dir + '" -name "*star*.fits.gz" -o -name "*gal*.fits.gz" ').splitlines()

print "Importing SDSS catalog ",
sdss.import_from_sweeps(cat_dir, files, create)
print " done."

cat = lsd.Catalog(cat_dir)
print "Building neighbor cache ",
cat.build_neighbor_cache()
print " done."

print "Computing summary statistics ",
cat.compute_summary_stats()
print " done."
print "Total number of objects in catalog: ", cat.nrows()
