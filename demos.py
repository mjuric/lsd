#!/usr/bin/env python

#
# Examples of using skysurvey.Catalog
#

import skysurvey as ss
import skysurvey.footprint as ssfoot
import skysurvey.utils as ssutils
import skysurvey.bhpix as bhpix
import numpy as np
from itertools import izip
import pyfits
#########################################################

if False:
	# Reprocess the calibration residuals image
	img = pyfits.getdata('rCalib.fits')
	pimg = np.zeros_like(img)
	aimg = np.zeros(shape=img.shape[1:3])
	bins = np.arange(-2-0.05/2, 2.0001+0.05/2, 0.05)
	for j in xrange(img.shape[1]):
		print j
		for i in xrange(img.shape[2]):
			hist = img[:,j,i]
			n = sum(hist)
			if n == 0: continue
			pimg[:,j,i] = hist/n
			aimg[j,i] = sum(hist*bins)/n
		pyfits.writeto('rCalib_norm.fits', pimg.astype(float), clobber=True)
		pyfits.writeto('rCalib_mean.fits', aimg.astype(float), clobber=True)
	exit()
#########################################################

if False:
	# xmatch PS1 to SDSS
	ps1 = ss.Catalog('ps1')
	sdss = ss.Catalog('sdss3')
	ss.xmatch(ps1, sdss, 'sdss')
	exit()
#########################################################

if True:
	# Compute and store the sky coverage at a given resolution (see skysurvey/tasks.py on how this is implemented)
	cat = ss.Catalog('ps1')
	print "Computing sky coverage map: ",
	sky = ss.compute_coverage(cat, dx=0.025, include_cached=True)
	pyfits.writeto('foot.0.025.fits', sky.astype(float).transpose()[::-1,], clobber=True)
	exit()
#########################################################



if False:
	# A simple query with reference to an x-matched catalog
	cat = ss.Catalog('ps1')
#	for row in cat.iterate('ra dec g r i z y sdss3.ra sdss3.dec sdss3.u sdss3.g sdss3.r', join_type='outer', foot=ssfoot.rectangle(180, 10, 181, 11)):
	for row in cat.iterate('* sdss3.*', join_type='outer', foot=ssfoot.rectangle(180, 10, 181, 11)):
		ssutils.extract_row(row)
		isnull = sdss3_g == cat.NULL
		d = ss.gc_dist(ra, dec, sdss3_ra, sdss3_dec)*3600 if not isnull else 0
		print "%12.8f %12.8f %6.3f %6.3f %6.3f %6.3f %6.3f %12.8f %12.8f %6.3f %6.3f %6.3f   %6.3f %1d" % (ra, dec, g, r, i, z, y, sdss3_ra, sdss3_dec, sdss3_u, sdss3_g, sdss3_r, d, isnull)
	exit()
#########################################################

if True:
	# MapReduce: Compute SDSS vs. PS1 r-mag distribution across the entire sky

	# Mapper: compute histograms of mag. offsets, keyed by healpix pixels
	def calib_check_mapper(rows, level, edges):
		rows = rows[np.isnan(rows["r"]) == False]			# Throw away rows where there's no r-band magnitude measurement
		if len(rows) == 0: return []

		# Compute the healpix pixel for each object
		(x, y)   = bhpix.proj_bhealpix(rows["ra"], rows["dec"])
		pixels   = bhpix.pix_idx(x, y, level)

		# For all objects in the same pixel, compute the histogram
		# of magnitude deviations
		ret = []
		for pix in set(pixels):
			rows2 = rows[pixels == pix]
			dm = rows2["r"] - rows2["sdss3.r"]
			hist = ssutils.xhistogram(dm, edges)
			ret += [ (pix, hist) ]

		return ret

	# Reducer: add together the histograms for the given pixel
	def calib_check_reducer(pix, hists):
		total = hists.pop()
		for hist in hists:
			total += hist

		return (pix, total)

	# Main program
	cat = ss.Catalog('ps1')

	level = 10						# healpix level
	nside = bhpix.nside(level)				# number of pixels
	edges = np.arange(-2, 2.0001, 0.05)			# edges of mag. offset histogram
	img = np.zeros(shape=[len(edges)+1, nside, nside])	# output image

	print "Computing magnitude distributions (%d x %d): " % (nside, nside),
	#for (pix, dist) in sorted(cat.map_reduce(calib_check_mapper, calib_check_reducer, mapper_args=(level, edges), cols='ra dec r sdss3.r', join_type='inner', foot=ssfoot.rectangle(180, 10, 181, 11))):
	#for (pix, dist) in sorted(cat.map_reduce(calib_check_mapper, calib_check_reducer, mapper_args=(level, edges), cols='ra dec r sdss3.r', join_type='inner', foot=ssfoot.rectangle(175, 35, 185, 85))):
	for (pix, dist) in sorted(cat.map_reduce(calib_check_mapper, calib_check_reducer, mapper_args=(level, edges), cols='ra dec r sdss3.r', join_type='inner')):
		j = pix // nside
		i = pix % nside
		img[:, j, i] = dist

	pyfits.writeto('rCalib2.fits', img.astype(float), clobber=True)
	exit()
#########################################################

if False:
	# Show sky coverage of cached objects
	def show_cached_only(rows):
		rows = rows[rows['cached'] != 0]
		return rows

	cat = ss.Catalog('ps1')
	print "Computing cached sky coverage map: ",
	sky = ss.compute_coverage(cat, dx=0.1, include_cached=True, filter=show_cached_only)
	pyfits.writeto('foot.0.1.fits', sky.astype(float).transpose()[::-1,], clobber=True)
	exit()
#########################################################

if False:
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

	cat = ss.Catalog('ps1')
	print "Computing dec. distribution:",
	for (k, v) in sorted(cat.map_reduce(deccount_mapper, deccount_reducer)):
		print k, v
#########################################################

if False:
	# Count the total number of objects in the catalog (see skysurvey/tasks.py on how this is implemented)
	# Ofcourse, this information should/will be cached
	cat = ss.Catalog('ps1')
	print "Counting the number of objects in catalog:",
	ntotal = ss.compute_counts(cat)
	print "  ==> Total of %d objects in catalog." % ntotal
#########################################################
