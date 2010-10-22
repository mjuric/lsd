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

if True:
	# Extract a block of rows and store it in a FITS file
	cat = ss.Catalog('sdss')
	query = 'ra, dec, l, b, type, flags, flags2, resolve_status, u, uErr, uExt, uCalib, g, gErr, gExt, gCalib, r, rErr, rExt, rCalib, i, iErr, iExt, iCalib, z, zErr, zExt, zCalib, run, camcol, field, objid'
	rows = cat.fetch(query=query, foot=ssfoot.rectangle(15, 15, 28, 75))
	print rows.dtype.names
	pyfits.writeto('sdss3-subset.fits', rows, clobber=True)
	exit()

if False:
	# Execute a query using both fetch() and iterate(), and compare the results.
	query = 'ra, dec, g, r, i, z, y'
	foot  = ssfoot.rectangle(180, 10, 181, 11)

	cat = ss.Catalog('ps1')
	rows1 =                 cat.fetch(query=query, foot=foot)
	rows2 = np.fromiter( (cat.iterate(query=query, foot=foot)), dtype=rows1.dtype)

	# get rid of NaNs, as they mess up sorting and comparison (eg., NaN == NaN evaluates to False)
	for col in rows1.dtype.names: rows1[col][np.isnan(rows1[col])] = 0
	for col in rows2.dtype.names: rows2[col][np.isnan(rows2[col])] = 0

	rows1.sort()
	rows2.sort()
	assert (rows1 == rows2).all()

	exit()

#########################################################

if False:
	# xmatch PS1 to SDSS
	ps1 = ss.Catalog('ps1')
	sdss = ss.Catalog('sdss')
	ss.xmatch(ps1, sdss)
	exit()
#########################################################

if False:
	# Compute and store the sky coverage at a given resolution (see skysurvey/tasks.py on how this is implemented)
	cat = ss.Catalog('ps1')
	print "Computing sky coverage map: ",
	sky = ss.compute_coverage(cat, dx=0.025, include_cached=False, query='ra, abs(dec), sdss.ra, sdss.dec, sdss.id XMATCH WITH sdss')
	#sky = ss.compute_coverage(cat, dx=0.025, include_cached=False, query='ra, dec')
	pyfits.writeto('foot.0.025.fits', sky.astype(float).transpose()[::-1,], clobber=True)
	exit()
#########################################################



if False:
	# A simple query with reference to an x-matched catalog
	cat = ss.Catalog('ps1')
	for row in cat.iterate('ra, dec, g, r, i, z, y, sdss.ra, sdss.dec, sdss.u, sdss.g, sdss.r XMATCH sdss(outer)', foot=ssfoot.rectangle(180, 10, 181, 11)):
		ssutils.extract_row(row)
		d = ss.gc_dist(ra, dec, sdss_ra, sdss_dec)*3600 if sdss_g != 0 else 0
		print "%12.8f %12.8f %6.3f %6.3f %6.3f %6.3f %6.3f %12.8f %12.8f %6.3f %6.3f %6.3f   %6.3f" % (ra, dec, g, r, i, z, y, sdss_ra, sdss_dec, sdss_u, sdss_g, sdss_r, d)
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
			dm = rows2["r"] - rows2["sdss.r"]
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
	for (pix, dist) in sorted(cat.map_reduce(calib_check_mapper, calib_check_reducer, mapper_args=(level, edges), query='ra, dec, r, sdss.r XMATCH sdss')):
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

if True:
	# Mapper: compute the distribution of counts within a cell, and
	#         return the result as an array of (bin, count) tuples
	def deccount_mapper(rows, bins):
		counts, _ = np.histogram(rows['dec'], bins)
		return zip(bins[:-1], counts)

	# Reducer: sum up the counts for a given bin
	def deccount_reducer(bin, counts):
		return (bin, sum(counts))

	ddec = 10
	bins = np.arange(-90, 90.01, ddec)	# bin edges
	cat  = ss.Catalog('ps1')
	hist = cat.map_reduce(deccount_mapper, deccount_reducer, mapper_args=(bins,), query='dec xmatch with sdss')
	for (bin, count) in sorted(hist):
		print "%+05.1f %10d" % (bin + ddec/2, count)
#########################################################

if False:
	# Count the total number of objects in the catalog (see skysurvey/tasks.py on how this is implemented)
	# Ofcourse, this information should/will be cached
	cat = ss.Catalog('ps1')
	print "Counting the number of objects in catalog:",
	ntotal = ss.compute_counts(cat)
	print "  ==> Total of %d objects in catalog." % ntotal
#########################################################
