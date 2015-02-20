#!/usr/bin/env python

import Polygon
import bhpix
import numpy as np
import astropy.coordinates
from numpy import sin, cos, radians, degrees, sqrt, arcsin, arctan2, pi
from interval import intervalset
from collections import defaultdict

# Full sky in bhpix projection
ALLSKY = Polygon.Polygon([
		(.5, 0), (1, .5), (1, 1), (.5, 1), (0, .5),
		(-.5, 1), (-1, 1), (-1, .5), (-.5, 0),
		(-1, -.5), (-1, -1), (-.5, -1), (0, -.5),
		(.5, -1), (1, -1), (1, -.5)
	])

def lambert_proj(l, phi, l0, phi1):
	"""
	Project l, phi to Lambert Equal Area Projection
	
	Return the Lambert Equal Area Projection coordinates x, y given
	input lon,lat == l,phi and projection pole l0, phi1

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
	"""
	Deproject from Lambert Equal Area Projection to lon, lat
	
	Deproject from Lambert Equal Area coordinates x, y to lon, lat,
	given projection pole l0, phi1

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
	"""
	Transform a polygon on the sphere to a Polygon in BHpix

	Given the vertices (lon, lat; ndarrays), transform them to BHpix
	projection space and construct a Polygon.Polygon instance
	corresponding to the input Polygon on the sphere.

	The function knows how to handle a case when the south pole is
	included in the polygon on the sphere.
	
	TODO: We currently don't take care to add vertices when crossing
	l=(0, 90, 180, 270) meridians. We should do that (otherwise small
	pieces of the sky will be effectively cut out of the resulting BHpix
	polygon).
	"""
	hrect = bhpix.proj_bhealpix(lon, lat)

	poly = Polygon.Polygon(zip(*hrect))

	if(poly.orientation()[0] == -1):
		poly = Polygon.Polygon([(1, 1), (-1, 1), (-1, -1), (1, -1)]) - poly

	return poly

def beam(lon0, lat0, radius=1., coordsys='equ', npts=360):
	"""
	Return a polygon (in bhpix projection) corresponding to the
	requested lon/lat/radius beam.
	"""
	# transform from the desired coordinate system to equatorial
	if coordsys == 'gal':
		_c = astropy.coordinates.SkyCoord(lon0, lat0, astropy.coordinates.Galactic, unit='deg').icrs
                lon0, lat0 = _c.ra.value, _c.dec.value
	elif coordsys == 'equ':
		pass
	else:
		raise Exception('Unknown coordinate system')

	r, _ = lambert_proj(0, 90-radius, 270, 90)		# Obtain beam radius in Lambert coordinates
	phi  = np.linspace(0, 2*pi, npts, endpoint=False)	# Construct the polygon (approximating a circle, when npts is large)
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
	"""
	Return a polygon (in bhpix projection) corresponding to the
	requested lon/lat rectangle.
	"""
	if lon0 > lon1:
		lon1 = lon1 + 360

	# Generate CCW rectangle bounded by great circles
	#nsplit = int(round((lon1 - lon0) / 0.1) + 1)
	#lon = np.linspace(lon0, lon1, nsplit)
	#lon = np.concatenate((lon, lon[::-1]))
	#lat = [lat0] * nsplit + [lat1]*nsplit

	# Generate a well sampled CCW rectangle
	nlon = max(int(round((lon1 - lon0) / 0.1) + 1), 10)
	nlat = max(int(round((lat1 - lat0) / 0.1) + 1), 10)
	vlon0 = np.linspace(lon0, lon1, nlon); vlon1 = vlon0[::-1]
	vlat0 = np.linspace(lat0, lat1, nlat); vlat1 = vlat0[::-1]
	lon = np.concatenate((vlon0, [lon1]*nlat, list(vlon1), [lon0]*nlat))
	lat = np.concatenate(([lat0]*nlon, list(vlat0), [lat1]*nlon, list(vlat1)))

	# transform it from the desired coordinate system
	if coordsys == 'gal':
		_c = astropy.coordinates.SkyCoord(lon, lat, frame=astropy.coordinates.Galactic, unit='deg').icrs
                lon, lat = _c.ra.value, _c.dec.value
	elif coordsys == 'equ':
		pass
	else:
		raise Exception('Unknown coordinate system')

	return make_healpix_poly(lon, lat)

def __part_to_xy_t(part):
	""" Internal: Used by canonicize_bounds """
	bounds_xy = Polygon.Polygon()
	bounds_t = intervalset()
	for v in part:
		if   isinstance(v, Polygon.Polygon):
			bounds_xy |= v
		elif isinstance(v, intervalset):
			bounds_t |= v
		else:
			raise Exception('Incorrect part specification')

	if not bounds_xy:		bounds_xy =  ALLSKY
	if len(bounds_t) == 0:	bounds_t = intervalset((-np.inf, np.inf))

	return (bounds_xy, bounds_t)

def make_canonical(bounds):
	"""
	Convert a "free-form" bounds specification to canonical form

	Used to allow some flexibility when inputing the bounds from the
	command line.

	Parameters
	----------
	bounds: Polygon, tuple, integer or an array of these
	    The bounds can either be a Polyon, a specific cell_id integer, a
	    (t0, t1) time tuple, or a list of these. If bounds is an empty
	    Python list (i.e., []) or None, we return None (== all sky)

	Returns a list of (Polygon, intervalset) tuples, or None (signifying
	all sky), both suitable to be passed as bounds parameter of
	Query.execute()/iterate()/fetch() calls.
	"""

	# Handle all-sky requests, and scalars
	if bounds == [] or bounds is None:
		return None
	elif not isinstance(bounds, list):
		bounds = [ bounds ]

	# Now our input is a list that consists of one or more of:
	# a) Polygon instances
	# b) interval instances
	# c) tuples or lists of (Polygon, interval, Polygon, interval, ...)
	# e) cell_ids (integers)
	#
	# Do the following:
	# - extract a) and b) and form a single ([Polygon], interval), add it to the list
	# - append e) to the output list
	# - for each tuple:
	#	form a single ([Polygon], interval)
	# 	call _get_cell_recursive to obtain cell_ids, append them to output

	#print "***INPUT:", bounds;

	# cell->bound_xy->times
	cells = defaultdict(dict)
	part0 = []
	parts = []
	for v in bounds:
		if   isinstance(v, Polygon.Polygon) or isinstance(v, intervalset):
			part0.append(v)
		elif isinstance(v, tuple) or isinstance(v, list):
			parts.append(v)
		else:
			cells[int(v)][None] = None	# Fetch entire cell (space and time)

	#print "****A1:", cells, part0, parts;
	if part0:
		parts.insert(0, part0)

	bounds = [ __part_to_xy_t(part) for part in parts]

	return bounds
