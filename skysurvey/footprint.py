import Polygon
import bhpix
import numpy as np
from slalib import sla_galeq

# Full sky in bhpix projection
ALLSKY = Polygon.Polygon([
		(.5, 0), (1, .5), (1, 1), (.5, 1), (0, .5),
		(-.5, 1), (-1, 1), (-1, .5), (-.5, 0),
		(-1, -.5), (-1, -1), (-.5, -1), (0, -.5),
		(.5, -1), (1, -1), (1, -.5)
	])

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

	# Project to bhpix and construct the polygon, taking care of the south pole
	hrect = bhpix.proj_bhealpix(lon, lat)
	poly = Polygon.Polygon(zip(*hrect))
	if(poly.orientation()[0] == -1):
		poly = Polygon.Polygon([(1, 1), (-1, 1), (-1, -1), (1, -1)]) - poly
	return poly
