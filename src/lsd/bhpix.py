#!/usr/bin/env python
"""
BHpix and HEALPix projection functions
"""

import numpy as np
from numpy import cos, radians, fmod, sqrt, pi, arcsin, degrees, abs, floor, fabs

def proj_healpix(l, b):
	"""
	Project (l, b) to HealPix projection coordinates

	Parameters
	----------
	l, b: numpy array or scalar
	    The longitude and lattitude, in degrees
	    
	Returns
	-------
	x, y: np.ndarrays
	    The x and y coordinates in HealPix projection.
	"""
	z   = cos(radians(90.-b))
	phi = radians(fmod(fmod(l, 360.)+360, 360)) # wrap to [0,360)

	phit = fmod(phi, .5*pi)
	sigz = np.where(z > 0, 2 - sqrt(3*(1-z)), -(2 - sqrt(3*(1+z))) )

	x = np.where(abs(z) < 2./3, 	phi, 		phi - (abs(sigz) - 1) * (phit - 0.25 * pi))
	y = np.where(abs(z) < 2./3, 	3./8*pi*z,	0.25 * pi * sigz)

	return (x, y)

# Constants used in proj_bhealpix and related functions
_c = np.array([-1., -1.,  1., 1.])
_s = np.array([ 1., -1., -1., 1.])

def proj_bhealpix(l, b):
	"""
	Project (l, b) to BHpix ("Butterfly HealPix") projection

	Parameters
	----------
	l, b: numpy array or scalar
	    The longitude and lattitude, in degrees
	    
	Returns
	-------
	x, y: np.ndarrays
	    The x and y coordinates in BHpix projection. The
	    projected coordinates range from [-1, 1].
	"""
	(hxx, hyy) = proj_healpix(l, b)

	l = (hxx / (0.5*pi)).astype(int)
	h1x = (hxx - pi/2*(0.5 + l))
	h1y = hyy - pi/2

	h1 = (h1x * _c[l] - h1y * _s[l]) / pi
	h2 = (h1x * _s[l] + h1y * _c[l]) / pi

	return (h1, h2)

def deproj_healpix(x, y):
	"""
	Deproject from HealPix to (lon, lat)

	Parameters
	----------
	x, y: numpy array or scalars
	    HealPix coordinates of the point(s)
	    
	Returns
	-------
	lon, lat: np.ndarrays
	    The deprojected longitude and lattitude
	    corresponding to (x, y), in degrees.
	"""
	l = np.empty_like(x)
	b = np.empty_like(x)

	# Equations from Calabretta and Roukema, MNRAS, 381, 865
	K = 3; H = 4;

	equ = fabs(y) <= 0.5*pi * (K-1)/H

	# Equatorial regions
	l[equ] = x[equ]
	b[equ] = arcsin(y[equ]*H / (0.5*pi*K))

	# Polar regions
	pol = ~equ
	w = 1 if fmod(K, 2) == 1 else 0
	s = 0.5*(K+1) - abs(y[pol]*H)/pi
	xc = -pi + (2.*floor( (x[pol] + pi)*H/(2*pi) + 0.5*(1-w)) + w)*pi/H
	l[pol] = xc + (x[pol] - xc) / s
	b[pol] = arcsin(1 - s*s/K) * np.where(y[pol] > 0, 1, -1)

	return degrees(l), degrees(b)

def _deproj_bhealpix_scalar(x, y):
	""" Aux. function for deproj_bhealpix. Do not use directly. """
	(l, b) = deproj_bhealpix(np.array([x]), np.array([y]))
	return l[0], b[0]

def deproj_bhealpix(x, y):
	"""
	Deproject from BHpix ("butterfly HealPix") to (lon, lat)

	Parameters
	----------
	x, y: numpy array or scalars
	    BHpix coordinates of the point(s)
	    
	Returns
	-------
	lon, lat: np.ndarrays
	    The deprojected longitude and lattitude
	    corresponding to (x, y), in degrees.
	"""
	if not isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
		return _deproj_bhealpix_scalar(x, y)

	# Compute to which of the four healpix slices
	# this point belongs to
	l = degrees(np.arctan2(y, x))
	l = np.where(l > 0, l, l + 360)
	l = (l / 90).astype(int)

	h1x = pi/2. * (_c[l]*x + _s[l]*y)
	h1y = pi/2. * (_c[l]*y - _s[l]*x)
	
	hxx = h1x + pi/2.*(0.5 + l)
	hyy = h1y + pi/2.

	return deproj_healpix(hxx, hyy)

def ij_center(i, j, level):
	"""
	BHpix pixel center from BHpix integers (internal)
	
	Return (x, y) coordinates of a pixel with integer
	coordinates (i, j), at level 'level'. The integer
	coordinates range from to [-2**level/2, 2**level/2).

	TODO: ij should be redefined to span from
	      [0, 2**level)
	"""
	dx = pix_size(level)
	return ((i + 0.5)*dx, (j + 0.5)*dx)

def xy_to_ij(x, y, level):
	"""
	Compute BHpix integer coordinates (internal)

	Return (i, j) integers corresponding to (x, y)
	where (i, j) span [-2**level/2, 2**level/2).
	
	TODO: ij should be redefined to span from
	      [0, 2**level)
	"""
	dx = pix_size(level)
	return (x // dx, y // dx)

def xy_center(x, y, level):
	"""
	Return the center of a pixel with point (x, y)
	"""
	(i, j) = xy_to_ij(x, y, level)
	return ij_center(i, j, level)

def pix_size(level):
	"""
	The size (in BHpix projected units) of a pixel
	at level 'level'.
	"""
	return 2.**(1-level)

def NSIDE(level):
	"""
	The HealPix NSIDE parameter corresponding to 'level'
	"""
	return width(level) / 4

def pix_area(lev):
	"""
	The area (in degrees) of a pixel at level 'lev'
	"""
	return (129600. / np.pi) / (3./4. * width(lev)**2)

def pix_idx(x, y, level):
	"""
	The one-dimensional BHpix index I of a pixel containing
	the point (x, y).
	
	The index is computed as j * wh + i, where (i, j) are
	integer indices shifted so that
	(x,y) = (-1,-1) <=> (i, j) <=> (0, 0)
	"""
	i, j = xy_to_ij(x, y, level)
	wh = width(level)
	i = i + wh/2; j = j + wh/2;
	return j * wh + i

def width(level):
	'''
	The width on BHpix map (in pixels) at a given level.
	
	The number of pixels in the map equals width*width.
	Note that not all of them are valid pixels on the sphere.
	'''
	return 2**level

def get_pixel_level(x, y):
	"""
	Deduce the level, given a center (x, y) of a pixel
	
	Returns
	-------
	level: integer
	    BHpix level
	"""
	for levx in xrange(32):
		p = x*(1<<levx)
		if int(p) == p: break;

	for levy in xrange(32):
		p = y*(1<<levy)
		if int(p) == p: break;

	if(levx != levy or levx == 31):
		raise Exception('Invalid pixel center coordinate ' + str((x,y)))

	return levx

def testview():
	vl = np.arange(0., 360., 5.)
	vb = np.arange(-90., 90., 5.)
	(l, b) = zip(*[ (l, b) for b in vb for l in vl ])
	(x, y) = toBHealpix(np.array(l), np.array(b))

	scatter(x, y); show();

def testpix():
	#x = -0.5; y = 0.721; k = 6;
	k = 6;
	for ctr in xrange(100000):
		(x, y) = ran.uniform(-1, 1), ran.uniform(-1, 1)
		i, j = xy_center(x, y, k)
		ns   = width(k)
		dx   = pix_size(k)
		if not (i-dx/2 <= x < i+dx/2) or not (j-dx/2 <= y < j+dx/2):
			print x, y, '->', i, j, ns, dx, get_path(x, y, k)
			raise Exception('Bug bug bug -- test failed.!!')
	print x, y, '->', i, j, ns, dx, get_path(x, y, k)
	print "Passed."

################################################################
def map_to_valid_pixels(x, y, dx, assume_adjacent=True):
	"""
	Map pixel coordinates (x, y) to a valid pixel.

	Map (x, y) to a valid BHpix pixel, if they're outside
	of the valid BHpix projection boundaries. This is useful
	to enumerate all neighbors of a given pixel (see
	bhpix.neighbors).
	"""
	if not assume_adjacent:
		raise Exception('Not implemented.')

	# Map into a pixel within range
	ax = fabs(x)
	ay = fabs(y)
	if ax > 1 or ay > 1:
		# Fallen off the rectangle
		if y > 1.:
			if x > 1:
				pix = (-(x-dx), -(y-dx))
			elif x < -1:
				pix = (-(x+dx), -(y-dx))
			else:
				pix = (-x, y-dx)
		elif y < -1.:
			if x > 1:
				pix = (-(x-dx), -(y+dx))
			elif x < -1:
				pix = (-(x+dx), -(y+dx))
			else:
				pix = (-x, y+dx)
		elif x > 1:
			pix = (x-dx, -y)
		else:
			pix = (x+dx, -y)
	elif fabs(ax - ay) > 0.5:
		if ax - 0.5 > ay:
			# left and right triangles
			if   x > 0 and y > 0:
				pix = (0.5 + y,   0.5 - x)
			elif x > 0 and y < 0:
				pix = (0.5 - y,  -0.5 + x)
			elif x < 0 and y > 0:
				pix = (-0.5 - y,  0.5 + x)
			elif x < 0 and y < 0:
				pix = (-0.5 + y, -0.5 - x)
		else:
			# top and bottom triangles
			if   y > 0 and x > 0:
				pix = (0.5 - y,  0.5 + x)
			elif y > 0 and x < 0:
				pix = (-0.5 + y, 0.5 - x)
			elif y < 0 and x > 0:
				pix = (0.5 + y, -0.5 - x)
			elif y < 0 and x < 0:
				pix = (-0.5 - y, -0.5 + x)
	else:
		# OK.
		pix = (x, y)

	# Now check if this is a "halfpixel" and add its
	# second piece
	if fabs(fabs(pix[0]) - fabs(pix[1])) == 0.5:
		(x, y) = pix
		if fabs(x) - 0.5 == fabs(y):
			pix2 = (x, -y)
		else:
			pix2 = (-x, y)
		ret = (pix, pix2)
	else:
		ret = (pix,)

	for (cx, cy) in ret:
		if fabs(fabs(cx) - fabs(cy)) > 0.5 or fabs(cx) >= 1 or fabs(cy) >= 1:
			raise Exception("map_to_valid_pixels: ", x, y, " mapped to illegal value: ", cx, cy)

	return ret

def neighbors(x, y, level, include_self=False):
	"""
	Return a set of neighbors of a given pixel
	
	Returns
	-------
	neighbors: set
	    A set of (x, y) center coordinates of pixels
	    sharing an edge or a corner with the input
	    pixel.
	"""
	# Return a set of immediate neighbors of the pixel
	# into which (x,y) falls
	result = set()
	dx = pix_size(level)
	(ox, oy) = xy_center(x, y, level)
	#(ox, oy, dx) = (x, y, level)
	for (cx, cy) in map_to_valid_pixels(ox, oy, dx):
#		print (cx, cy)
		for di in xrange(-1, 2):
			for dj in xrange(-1, 2):
				(x, y) = (cx + di*dx, cy + dj*dx)
				pix = map_to_valid_pixels(x, y, dx)
#				print '  n=', di, dj, pix
				result.update(pix)

	if not include_self and (ox, oy) in result:
		result.remove((ox, oy))

	return result
################################################################


#testview()
#from matplotlib.pyplot import *
#import random as ran
#testpix()

if __name__ == '__main__':
	print proj_bhealpix(np.array([10, 10]), np.array([10, 80]))
	print proj_bhealpix(10, 80)
	print deproj_bhealpix(*proj_bhealpix(np.array([10, 10]), np.array([10, 80])))
	exit()

if __name__ == '__main__':
	clon = 212; clat = 12;
	lon1 = 212.2; lat1 = 15.1;
	lon2 = 212; lat2 = 15.11;
	(x1, y1) = gnomonic(lon1, lat1, clon, clat)
	(x2, y2) = gnomonic(lon2, lat2, clon, clat)
	dx = x2 - x1
	dy = y2 - y1
	d = (dx**2 + dy**2)**.5
	d0 = gc_dist(lon1, lat1, lon2, lat2)
	print d, d0, d/d0 - 1
	#print x, y, (x*x + y*y)**.5, gc_dist(lon1)
	exit()

if __name__ == '__main__':
	from numpy.random import random
	l = random(100000)*360
	b = random(100000)*180 - 90
	print len(l), ' items'
	x, y = proj_bhealpix(l, b)
	l1, b1 = deproj_bhealpix(x, y)
	print np.allclose(l, l1, 1e-10, 1e-10), np.allclose(b, b1, 1e-10, 1e-10)
	exit()

