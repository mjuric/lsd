#!/usr/bin/env python

import sys
import getopt
import skysurvey.sdss  as sdss
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
	print args
	print "Error: Not enough command line arguments."
	usage();
	exit(-1)

cat_dir, sweep_dir = args
#sweep_dir = '/data/sdss/sdss3/2009-11-16.v2/301/'
#cat_dir = 'sdss';

create = False
for (o, a) in optlist:
	if o == '-c' or o == '--create':
		create = True

#
# Actual work
#
files = shell('find "' + sweep_dir + '" -name "*star*.fits.gz"').splitlines();

sdss.import_from_sweeps(cat_dir, files, create)
