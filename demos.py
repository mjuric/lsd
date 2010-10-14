#!/usr/bin/env python

#
# Examples of using skysurvey.Catalog
#

import skysurvey as ss
import numpy as np
from itertools import izip
import pyfits

# xmatch PS1 to SDSS
#ps1 = ss.Catalog('ps1')
#sdss = ss.Catalog('sdss3')
#ss.xmatch(ps1, sdss, 'sdss')
#exit()


# Open the catalog
#cat = ss.Catalog('sdss3')
cat = ss.Catalog('ps1')

#rows = cat.fetch('ra dec g r i z y')
#exit()


# Compute and store the sky coverage at a given resolution (see skysurvey/tasks.py on how this is implemented)
print "Computing sky coverage map: ",
sky = ss.compute_coverage(cat, dx=0.05, include_cached=True)
pyfits.writeto('foot.0.05.fits', sky.astype(float).transpose()[::-1,], clobber=True)
exit()
#########################################################


# Show sky coverage of cached objects
def show_cached_only(rows):
	return rows[rows['cached'] != 0]

print "Computing cached sky coverage map: ",
sky = ss.compute_coverage(cat, dx=0.1, filter=show_cached_only)
pyfits.writeto('foot.0.1.fits', sky.astype(float).transpose()[::-1,], clobber=True)
exit()
#########################################################


# Custom MapReduce example: create a histogram of counts vs. declination
def deccount_mapper(rows):
	# Mapper: compute the histogram for objects in this cell
	hist, edges = np.histogram(rows["dec"], bins=18, range=(-90, 90))
	bins = edges[0:-1] + 0.5*np.diff(edges)

	# Return only nonzero bins
	res = [ (bin, ct) for (bin, ct) in izip(bins, hist) if ct > 0 ]
	return res

def deccount_reducer(bin, counts):
	# Reducer: sum up the counts for each declination bin
	return (bin, sum(counts))

print "Computing dec. distribution:",
for (k, v) in sorted(cat.map_reduce(deccount_mapper, deccount_reducer)):
	print k, v
#########################################################


# Count the total number of objects in the catalog (see skysurvey/tasks.py on how this is implemented)
# Ofcourse, this information should/will be cached
print "Counting the number of objects in catalog:",
ntotal = ss.compute_counts(cat)
print "  ==> Total of %d objects in catalog." % ntotal
#########################################################
