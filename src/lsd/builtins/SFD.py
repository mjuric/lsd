import os
import pywcs
try:
	import astropy.io.fits as pyfits
except ImportError:
	import pyfits
import numpy
from scipy.ndimage import map_coordinates

class DustMap(object):
	"""
	A generalized dust map class

	Author: Eddie Schlafly <schlafly@cfa...>
	Modified by: Mario Juric <mjuric@cfa...>
	
	Return SFD (or other maps) at the Galactic coordinates l, b.

	Example usage:
	EBV = DustMap(map='sfd')
	h, w = 1000, 4000
	b, l = numpy.mgrid[0:h,0:w]
	l = 180.-(l+0.5) / float(w) * 360.
	b = 90. - (b+0.5) / float(h) * 180.
	ebv = EBV(l, b)
	imshow(ebv, aspect='auto', norm=matplotlib.colors.LogNorm())
	"""

	def __init__(self, map='sfd', size=None, dir=None):
		self.__data = {}

		if map == 'sfd':
			map = 'dust'
		if map in ['dust', 'd100', 'i100', 'i60', 'mask', 'temp', 'xmap']:
			fname = 'SFD_'+map
		else:
			fname = map

		maxsize = { 'd100':1024, 'dust':4096, 'i100':4096, 'i60':4096, 'mask':4096 }
		if size is None and map in maxsize:
			size = maxsize[map]
		if size is not None:
			fname = fname + '_%d' % size

		if dir is None:
			from .. import config
			dir = os.getenv('DUST_DIR', os.path.join(config.data_dir, 'sfd-dust-maps'))

		fname = os.path.join(dir, fname)
		if not os.access(fname+'_ngp.fits', os.F_OK):
			raise Exception('Map file %s not found. Check your $DUST_DIR or download it from http://www.astro.princeton.edu/~schlegel/dust/data/data.html.' % (fname+'_ngp.fits[.gz]'))

		# Load the maps
		self.data = {}
		for pole in ['ngp', 'sgp']:
			fn = fname+'_%s.fits' % (pole,)
			with pyfits.open(fn, memmap=True) as hdulist:
				self.data[pole] = hdulist[0].header, hdulist[0].data

	def __call__(self, l, b, order=0):
		l = numpy.asarray(l)
		b = numpy.asarray(b)

		if l.shape != b.shape:
			raise ValueError('l.shape must equal b.shape')

		out = numpy.zeros_like(l, dtype='f4')

		for pole in ['ngp', 'sgp']:
			m = (b >= 0) if pole == 'ngp' else b < 0
			if numpy.any(m):
				if not m.shape:	# Support for 0-dimensional arrays (scalars). Otherwise it barfs on l[m], b[m]
					m = ()
				header, data = self.data[pole]
				wcs = pywcs.WCS(header)
				x, y = wcs.wcs_sky2pix(l[m], b[m], 0)
				out[m] = map_coordinates(data, [y, x], order=order)

		return out

## Define the functions for most commonly used maps
from ..utils import LazyCreate
EBV  = LazyCreate(DustMap, map='sfd')
temp = LazyCreate(DustMap, map='temp')
mask = LazyCreate(DustMap, map='mask')
xmap = LazyCreate(DustMap, map='xmap')
d100 = LazyCreate(DustMap, map='d100')
i100 = LazyCreate(DustMap, map='i100')
i60  = LazyCreate(DustMap, map='i60')
