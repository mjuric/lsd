#!/usr/bin/env python

#
# Examples of using skysurvey.Catalog
#

import skysurvey as ss
import numpy as np
from itertools import izip
import pyfits

# Open the catalog
cat = ss.Catalog('sdss')

# Compute and store the sky coverage at a given resolution (see skysurvey/tasks.py on how this is implemented)
print "Computing sky coverage: ",
sky = ss.compute_coverage(cat, dx=0.25)
pyfits.writeto('foot.fits', sky.astype(float).transpose()[::-1,], clobber=True)

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

# Count the total number of objects in the catalog (see skysurvey/tasks.py on how this is implemented)
# Ofcourse, this information should/will be cached
print "Counting the number of objects in catalog:",
ntotal = ss.compute_counts(cat)
print "  ==> Total of %d objects in catalog." % ntotal
