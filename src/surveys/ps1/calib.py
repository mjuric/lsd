#!/usr/bin/env python

import pyfits
import numpy as np
import os
import logging

logger = logging.getLogger('surveys.ps1.calib')

def ps_xy2cell(x, y, chip_id = None):
	"""
	Convert chip x,y pixel numbers to cell numbers within the chip.

	Python port of IDL routine ps_xy2cell by Finkbeiner & Schlafly.

	Parameters
	----------
	x : ndarray or scalar
	    x pixel number
	y : ndarray or scalar
	    y pixel number
	chip_id : ndarray, scalar, or None
	    Scalar id of the chip, where chip_id = X + Y*8.
	    By default, we assign cell numbers so that x,y = (0,0)
	    is always assigned to cell zero.  If chip is set to the chip 
	    number, we assign cell numbers so that cell 0 is most in the
	    direction of chip XY01, while chip XY63 is most in the direction of
	    chip XY76.  These differ in that chips with chip X coordinate <= 3
	    are 180 degree rotated.

	Returns
	-------
	cell : ndarray
	    The cell number; 0 indexed, 0-63.
	"""
	xpad = 8
	ypad = 10
	base = 600

	ix = (x+xpad) // (base+xpad)
	iy = (y+ypad) // (base+ypad)

	cell = np.array(ix + 8 * iy, dtype='i4', copy=False)

	if chip_id is not None:
		chipx = chip_id % 8
		flip = chipx <= 3
		cell = flip*63 + (1-2*flip)*cell

	return cell

def load_flat(fn):
	""" Read a FITS file with flats stored in "Finkbeiner Format v0"
	    and produce a (nx, nx, N) sized ndarray of flats, where the
	    first two dimensions refer to chips+cells, while the last one is
	    the index of a MJD computed as floor(mjd-mjd0). N is computed
	    from the largest MJD found in the file.

	    Given ndarrays of chip coordinates x, y, and a ndarray of
	    exposure dates mjd, the offsets are obtained as:
	    
	    	offs = flats[x, y, floor(mjd-mjd0)]
	"""
	f = pyfits.getdata(fn)[0]

	# Load and fix the shapes
	band   = f['BAND']
	mjds   = f['MJD']
	season = f['SEASON']
	nstack = f['NSTACK']
	nx     = f['NX']
	ny     = f['NY']
	nmean  = f['NMEAN']
	nmjd   = len(mjds)
	stack  = f['STACK'].reshape((nx, ny, nstack), order='F')
	mask   = f['MASK'].reshape((nx, ny), order='F')
	wts    = f['WEIGHTS'].reshape((nstack, nmjd), order='F')

	# Create the output array
	mjd0=int(mjds.min())
	mjd1=int(mjds.max())
	flats = np.zeros((nx, ny, np.ceil(mjd1-mjd0)+1))

	# Flat computation
	for k, mjd in enumerate(mjds):
		idx = int(mjd-mjd0)
		# Multiply the weights for a given night with the stacks,
		# and sum the result along the last axis. This gives a
		# (nx, ny) array. Multiply that with the mask, and store
		# it into idx position in the flats 3D array.
		flats[...,idx] = mask*np.sum(wts[:, k]*stack, axis=2)

	return band, (mjd0, mjd1), flats

def load_flats(pattern = 'flats/calib-%s.fits'):
	""" Returns a (flats4d, mjd0) tuple, where flats4d is
	    a 4-dimensonal array of "flats", to be indexed as:

	    	offs = flats[x, y, floor(mjd-mjd0), bandidx]

	    where bandidx is 0..4 for grizy, respectively.
	    
	    See load_flat() for more information.
	"""
	flats = dict()
	mmin = 1e10
	mmax = 0
	for i in 'grizy':
		band, (mjd0, mjd1), f = load_flat(pattern % i)

		assert band == i
		flats[band] = (mjd0, mjd1, f)
		mmin = min(mmin, mjd0)
		mmax = max(mmax, mjd1)

		(nx, ny) = f.shape[:2]

	nmjd = mmax-mmin+1
	flats4d = np.zeros((nx, ny, nmjd, len(flats)))

	for k, i in enumerate('grizy'):
		mjd0, mjd1, f = flats[i]
		flats4d[:,:,mjd0-mmin:mjd1-mmin+1, k] = f

	return flats4d, mjd0

def cell2flat_ij(chip_id, cell, nx = 64, ny = 64):
	"""
	Aux function for flat_offs.

	Compute 2D index into flat field offset arrays.
	
	Given a chip_id and an OTA cell_id (computed using ps_xy2cell),
	compute 2D index into dfink's flat field offset files.
	"""
	chipx, chipy = chip_id % 8, chip_id // 8
	ncx, ncy     = nx // 8, ny // 8

	return (chipx*ncx + cell % ncx, chipy*ncy + cell // ncx)

def flat_offs(chip_id, x, y, mjd, filterid, flatsmjd = None):
	"""
	Return the flat-field correction offset for the given object.

	Returns the photometry offsets given positions chip_id, the on-chip
	coordinates x, y, the MJD of the exposure, and the filterid of the
	filter (must be one of g.0000 through y.0000).

	Can be used from within LSD queryes.

	If flatsmjd is not given, auto-loads flat files from a pattern given
	by PS1_FLATS environment variable (one such as
	'flats/calib-%s.fits', where %s will be replaced by g, r, i, z, y). 
	"""
	
	self = flat_offs

	# Cast to arrays
	#x = np.array(x, copy=False, ndmin=1)
	#y = np.array(y, copy=False, ndmin=1)
	#chip_id = np.array(chip_id, copy=False, ndmin=1)

	bidx = np.zeros(x.shape, dtype=int)
	for k, i in enumerate('grizy'):
		in_ = filterid == '%s.0000' % i
		bidx[in_] = k

	# Auto-load and cache flats
	if flatsmjd is None:
		if getattr(flat_offs, 'flats', None) is None:
			flat_pat = os.getenv('PS1_FLATS', None)
			if flat_pat is None:
				raise Exception('Please set the PS1_FLATS environment'
				 'variable to a pattern matching PS1 flat files'
				 '(e.g., "flats/calib-%s.fits")')
			self.flats, self.mjd0 = load_flats(flat_pat)
	else:
		self.flats, self.mjd0 = flatsmjd

	# Compute cell coordinates x, y
	cell = ps_xy2cell(x, y, chip_id)
	i, j = cell2flat_ij(chip_id, cell, nx=self.flats.shape[0], ny=self.flats.shape[1])
	#print chip_id, cell, x, y, i, j

	k = np.floor(mjd - self.mjd0).astype(int)
	notfound = (k < 0) | (k >= self.flats.shape[2]) | (i > 63) | (j > 63) | (i < 0) | (j < 0)
	k[notfound] = 0
	i[notfound] = 0
	j[notfound] = 0
	offs = self.flats[i, j, k, bidx]
	if np.any(notfound):
		offs[notfound] = 0
		umjd = np.unique(mjd[notfound])
		logger.warning("Corrections not found for %d out of %d exposures." % (len(umjd), len(np.unique(mjd))))
	return offs

	# clf(); imshow(ff.T, interpolation='nearest', cmap='hot'); colorbar(); xlabel('X'); ylabel('Y')

if __name__ == '__main__':
	# Q/A code
	#band, flats, nmjd = load_flat('flats/calib-r.fits')
	#print "Loaded", band, flats.shape, "nmjd=", nmjd
	print "HERE!"
