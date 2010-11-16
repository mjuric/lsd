#!/usr/bin/env python

import Polygon
import bhpix
import numpy as np
from slalib import sla_galeq
from numpy import sin, cos, radians, degrees, sqrt, arcsin, arctan2, pi

# Full sky in bhpix projection
ALLSKY = Polygon.Polygon([
		(.5, 0), (1, .5), (1, 1), (.5, 1), (0, .5),
		(-.5, 1), (-1, 1), (-1, .5), (-.5, 0),
		(-1, -.5), (-1, -1), (-.5, -1), (0, -.5),
		(.5, -1), (1, -1), (1, -.5)
	])

def lambert_proj(l, phi, l0, phi1):
	""" Return the Lambert Equal Area Projection coordinates
	    x, y given input lon,lat == l,phi and projection pole
	    l0, phi1

	    All input quantities are in degrees.
	"""
	l    = radians(l)
	phi  = radians(phi)
	l0   = radians(l0)
	phi1 = radians(phi1)

	cosphi1 = cos(phi1)
	sinphi1 = sin(phi1)

	denom = 1. + sinphi1 * sin(phi) + cosphi1 * cos(phi) * cos(l - l0)

	kp = sqrt(2. / denom);
	x = kp * cos(phi) * sin(l - l0)
	y = kp * (cosphi1 * sin(phi) - sinphi1 * cos(phi) * cos(l-l0))

	return x, y

def lambert_deproj(x, y, l0, phi1):
	""" Deproject from Lambert Equal Area coordinates x, y
	    to lon, lat, given projection pole l0, phi1

	    All angular quantities are in degrees.
	"""
	r = sqrt(x * x + y * y);
	c = 2 * arcsin(0.5 * r);

	phi1    = radians(phi1)
	cosphi1 = cos(phi1)
	sinphi1 = sin(phi1)

	phi = np.where(r != 0., arcsin(cos(c) * sinphi1 + y * sin(c) * cosphi1 / r), phi1)
	l   = np.where(r != 0., arctan2(x * sin(c), r * cosphi1 * cos(c) - y * sinphi1 * sin(c)), 0.)

	return l0 + degrees(l), degrees(phi)

def make_healpix_poly(lon, lat):
	# Project to bhpix and construct the polygon, taking care of the south pole
	hrect = bhpix.proj_bhealpix(lon, lat)
	poly = Polygon.Polygon(zip(*hrect))
	if(poly.orientation()[0] == -1):
		poly = Polygon.Polygon([(1, 1), (-1, 1), (-1, -1), (1, -1)]) - poly
	return poly

def beam(lon0, lat0, radius=1., coordsys='equ'):
	""" Return a polygon (in bhpix projection) corresponding
	    to the requested lon/lat/radius beam.
	    
	    TODO: Properly handle lon=(0,90,180,270) transitions
	    in bhpix projection
	"""
	# transform from the desired coordinate system to equatorial
	if coordsys == 'gal':
		lon0, lat0 = degrees(sla_galeq(radians(lon0), radians(lat0)))
	elif coordsys == 'equ':
		pass
	else:
		raise Exception('Unknown coordinate system')

	r, _ = lambert_proj(0, 90-radius, 270, 90)		# Obtain beam radius in Lambert coordinates
	phi  = np.linspace(0, 2*pi, 360, endpoint=False)	# Construct a finely sampled circle
	x, y = r*cos(phi), r*sin(phi)				#
	l, b = lambert_deproj(x, y, lon0, lat0)			# Deproject to circle on the sky

	return make_healpix_poly(l, b)

if __name__ == '__main__':
	print beam(143.01142277, 39.00018678).center()
	exit()

	l = np.array([3, 7, 12])
	b = np.array([44, 22, 12])
	x, y = lambert_proj(l, b, -22, -11)
	lb = lambert_deproj(x, y, -22, -11)
	print l, b, x, y, lb
	exit()

def rectangle(lon0, lat0, lon1, lat1, coordsys='equ'):
	""" Return a polygon (in bhpix projection) corresponding
	    to the requested lon/lat rectangle.
	    
	    TODO: Properly handle lon=(0,90,180,270) transitions
	    in bhpix projection
	"""
	if lon0 > lon1:
		lon1 = lon1 + 360

	# Generate CCW rectangle bounded by great circles
	#nsplit = int(round((lon1 - lon0) / 0.1) + 1)
	#lon = np.linspace(lon0, lon1, nsplit)
	#lon = np.concatenate((lon, lon[::-1]))
	#lat = [lat0] * nsplit + [lat1]*nsplit

	# Generate a well sampled CCW rectangle
	nlon = int(round((lon1 - lon0) / 0.1) + 1)
	nlat = int(round((lat1 - lat0) / 0.1) + 1)
	vlon0 = np.linspace(lon0, lon1, nlon); vlon1 = vlon0[::-1]
	vlat0 = np.linspace(lat0, lat1, nlat); vlat1 = vlat0[::-1]
	lon = np.concatenate((vlon0, [lon1]*nlat, list(vlon1), [lon0]*nlat))
	lat = np.concatenate(([lat0]*nlon, list(vlat0), [lat1]*nlon, list(vlat1)))

	# transform it from the desired coordinate system
	if coordsys == 'gal':
		for i in xrange(len(lon)):
			(lon[i], lat[i]) = np.degrees(sla_galeq(np.radians(lon[i]), np.radians(lat[i])))
	elif coordsys == 'equ':
		pass
	else:
		raise Exception('Unknown coordinate system')

	return make_healpix_poly(lon, lat)
