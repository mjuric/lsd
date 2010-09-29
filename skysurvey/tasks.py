"""
	Common tasks needed when dealing with survey datasets.
"""

import pool2
import numpy as np
from itertools import izip

###################################################################
## Sky-coverage computation
def _coverage_mapper(rows, dx = 1.):
	self = _coverage_mapper

	i = (rows['ra'] / dx).astype(int)
	j = ((90 - rows['dec']) / dx).astype(int)

	(imin, imax, jmin, jmax) = (i.min(), i.max(), j.min(), j.max())
	w = imax - imin + 1
	h = jmax - jmin + 1
	sky = np.zeros((w, h))

	i -= imin; j -= jmin
	for (ii, jj) in izip(i, j):
		sky[ii, jj] += 1

	return (sky, imin, jmin, self.CELL_FN)

def compute_coverage(cat, dx = 0.5):
	width  = int(round(360/dx))
	height = int(round(180/dx))

	sky = np.zeros((width, height))

	for (patch, imin, jmin, fn) in cat.map_reduce(_coverage_mapper, mapper_args=(dx,)):
		sky[imin:imin + patch.shape[0], jmin:jmin + patch.shape[1]] += patch
	
	return sky
###################################################################

###################################################################
## Mapper-only example: count the number of objects in each file ##
def ls_mapper(rows):
	# return the number of rows in this chunk, keyed by the filename
	self = ls_mapper
	return (self.CELL_FN, len(rows))

def compute_counts(cat):
	ntotal = 0
	for (file, nobjects) in cat.map_reduce(ls_mapper, include_cached=False, nworkers=4):
		ntotal = ntotal + nobjects
	return ntotal
###################################################################
