#!/usr/bin/env python
#
# Example: import_dvo.py -c ps1 /raid14/panstarrs/dvo-201008
#

import sys
import getopt
import skysurvey.dvo  as dvo
import skysurvey as lsd
from   skysurvey.utils import *

def usage():
	print "Usage: %s [-c|--create] <cat_dir> <dvo_file_dir>" % sys.argv[0]

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

cat_dir, dvo_dir = args

create = False
for (o, a) in optlist:
	if o == '-c' or o == '--create':
		create = True

#
# Actual work
#
files = shell('find "' + dvo_dir + '" -name "*.cpt"').splitlines();
#files = files[:10]

print "Importing PS1 catalog ",
dvo.import_from_dvo(cat_dir, files, create)
print " done."
#exit()

cat = lsd.Catalog(cat_dir)
print "Building neighbor cache ",
cat.build_neighbor_cache()
print " done."

print "Computing summary statistics ",
cat.compute_summary_stats()
print " done."
