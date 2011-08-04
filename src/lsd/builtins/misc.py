# LSD-defined functions available within queries

import numpy
import numpy as np
from .. import colgroup
from ..utils import NamedList

# Appendix of Reid et al. (http://adsabs.harvard.edu/cgi-bin/bib_query?2004ApJ...616..872R)
# This convention is also used by LAMBDA/WMAP (http://lambda.gsfc.nasa.gov/toolbox/tb_coordconv.cfm)
_angp = np.radians(192.859508333) #  12h 51m 26.282s (J2000)
_dngp = np.radians(27.128336111)  # +27d 07' 42.01" (J2000) 
_l0   = np.radians(32.932)
_ce   = np.cos(_dngp)
_se   = np.sin(_dngp)

def equgal(ra, dec):
	ra = np.radians(ra)
	dec = np.radians(dec)

	cd, sd = np.cos(dec), np.sin(dec)
	ca, sa = np.cos(ra - _angp), np.sin(ra - _angp)

	sb = cd*_ce*ca + sd*_se
	l = np.arctan2(sd - sb*_se, cd*sa*_ce) + _l0
	b = np.arcsin(sb)

	l = np.where(l < 0, l + 2.*np.pi, l)

	l = np.degrees(l)
	b = np.degrees(b)

	return NamedList(('l', l), ('b', b))

def galequ(l, b):
	l = np.radians(l)
	b = np.radians(b)

	cb, sb = np.cos(b), np.sin(b)
	cl, sl = np.cos(l-_l0), np.sin(l-_l0)

	ra  = np.arctan2(cb*cl, sb*_ce-cb*_se*sl) + _angp
	dec = np.arcsin(cb*_ce*sl + sb*_se)

	ra = np.where(ra < 0, ra + 2.*np.pi, ra)

	ra = np.degrees(ra)
	dec = np.degrees(dec)

	return NamedList(('ra', ra), ('dec', dec))

def _fits_quickparse(header):
	"""
	An ultra-simple FITS header parser.
	
	Does not support CONTINUE statements, HIERARCH, or anything of the
	sort; just plain vanilla:
	
	    	key = value / comment

	one-liners. The upshot is that it's fast, much faster than the
	PyFITS equivalent.

	NOTE: Assumes each 80-column line has a '\n' at the end (which is
	      how we store FITS headers internally.)
	"""
	res = {}
	for line in header.split('\n'):
		at = line.find('=')
		if at == -1 or at > 8:
			continue

		# get key
		key = line[0:at].strip()

		# parse value (string vs number, remove comment)
		val = line[at+1:].strip()
		if val[0] == "'":
			# string
			val = val[1:val[1:].find("'")]
		else:
			# number or T/F
			at = val.find('/')
			if at == -1: at = len(val)
			val = val[0:at].strip()
			if val.lower() in ['t', 'f']:
				# T/F
				val = val.lower() == 't'
			else:
				# Number
				val = float(val)
				if int(val) == val:
					val = int(val)
		res[key] = val
	return res;

def fitskw(hdrs, kw, default=0):
	"""
	Intelligently extract a keyword kw from an arbitrarely
	shaped object ndarray of FITS headers.
	
	Designed to be called from within LSD queries.
	"""
	shape = hdrs.shape
	hdrs = hdrs.reshape(hdrs.size)

	res = []
	cache = dict()
	for ahdr in hdrs:
		ident = id(ahdr)
		if ident not in cache:
			if ahdr is not None:
				#hdr = pyfits.Header( txtfile=StringIO(ahdr) )
				hdr = _fits_quickparse(ahdr)
				cache[ident] = hdr.get(kw, default)
			else:
				cache[ident] = default
		res.append(cache[ident])

	res = np.array(res).reshape(shape)
	return res

def ffitskw(uris, kw, default = False, db=None):
	""" Intelligently load FITS headers stored in
	    <uris> ndarray, and fetch the requested
	    keyword from them.
	"""

	if len(uris) == 0:
		return np.empty(0)

	uuris, idx = np.unique(uris, return_inverse=True)
	idx = idx.reshape(uris.shape)

	if db is None:
		# _DB is implicitly defined inside queries
		global _DB
		db = _DB

	ret = []
	for uri in uuris:
		if uri is not None:
			with db.open_uri(uri) as f:
				hdr_str = f.read()
			hdr = _fits_quickparse(hdr_str)
			ret.append(hdr.get(kw, default))
		else:
			ret.append(default)

	# Broadcast
	ret = np.array(ret)[idx]

	assert ret.shape == uris.shape, '%s %s %s' % (ret.shape, uris.shape, idx.shape)

	return ret

def OBJECT(uris, db=None):
	""" Dereference blobs referred to by URIs,
	    assuming they're pickled Python objects.
	"""
	return _deref(uris, db, True)

def BLOB(uris, db=None):
	""" Dereference blobs referred to by URIs,
	    loading them as plain files
	"""
	return _deref(uris, db, False)

def _deref(uris, db=None, unpickle=False):
	""" Dereference blobs referred to by URIs,
	    either as BLOBs or Python objects
	"""
	if len(uris) == 0:
		return np.empty(0, dtype=object)

	uuris, idx = np.unique(uris, return_inverse=True)
	idx = idx.reshape(uris.shape)

	if db is None:
		# _DB is implicitly defined inside queries
		db = _DB

	ret = np.empty(len(uuris), dtype=object)
	for i, uri in enumerate(uuris):
		if uri is not None:
			with db.open_uri(uri) as f:
				if unpickle:
					ret[i] = cPickle.load(f)
				else:
					ret[i] = f.read()
		else:
			ret[i] = None

	# Broadcast
	ret = np.array(ret)[idx]

	assert ret.shape == uris.shape, '%s %s %s' % (ret.shape, uris.shape, idx.shape)

	return ret

def bin(v):
	"""
	Similar to __builtin__.bin but works on ndarrays.
	
	Useful in queries for converting flags to bit strings.

	FIXME: The current implementation is painfully slow.
	"""
	import __builtin__
	if not isinstance(v, np.ndarray):
		return __builtin__.bin(v)

	# Must be some kind of integer
	assert v.dtype.kind in ['i', 'u']

	# Create compatible string array
	l = v.dtype.itemsize*8
	ss = np.empty(v.shape, dtype=('a', v.dtype.itemsize*9))
	s = ss.reshape(-1)
	for i, n in enumerate(v.flat):
		c = __builtin__.bin(n)[2:]
		c = '0'*(l-len(c)) + c
		ll = [ c[k:k+8] for k in xrange(0, l, 8) ]
		s[i] = ','.join(ll)
	return ss

class Map(object):
	def __init__(self, k, v, missing=None):
		import numpy as np

		i = np.argsort(k)
		self.k = k[i]
		self.v = v[i]

		self.missing = missing

	def __call__(self, x):
		i = np.searchsorted(self.k, x)
		i[i == len(self.k)] = 0

		v = self.v[i]
		wipe = self.k[i] != x
		if np.any(wipe):
			if self.missing is not None:
				v[wipe] = [ self.missing ]
			else:
				# Have the missing records be set to zero
				tmp = np.zeros(v.shape, v.dtype)
				tmp[~wipe] = v[~wipe]

		# If this is a single-column return, return only the
		# single column, to allow expressions such as x + y(z)
		# in queries
		if len(v.dtype.names) == 1:
			return v[v.dtype.names[0]]

		return v

class FileTable(object):
	""" FileTable - load a FITS/pkl/text key/value table in queries
	"""
	__key, __vals, __map = 0, [1], None

	def __init__(self, fn, **kwargs):
		import os.path
		import numpy as np

		# Look for short-hand syntax
		if fn.find(':') != -1:
			fn, self.__key, vals = ( s.strip() for s in fn.split(':') )
			self.__vals = [ s.strip() for s in vals.split(',') ]

		# Load the file
		basename, ext = os.path.splitext(fn)
		ext = ext.lower()

		if ext == '.fits':
			# Assume fits
			import pyfits
			self.data = np.array(pyfits.getdata(fn, **kwargs))
		elif ext == '.pkl':
			# Assume pickled
			import cPickle
			from . import colgroup
			with open(fn) as fp:
				self.data = cPickle.load(fp)
				if isinstance(self.data, colgroup.ColGroup):
					self.data = self.data.as_ndarray()
				assert isinstance(self.data, np.ndarray)
		else:
			# Assume text
			from . import utils
			self.data = np.genfromtxt(utils.open_ex(fn), **kwargs)

	def __call__(self, x):
		if self.__map is None:
			self.__map = self.map(self.__key, *self.__vals)

		return self.__map(x)

	def map(self, key=0, val=1, *extra_vals, **kwargs):
		if not isinstance(key, str):
			key = self.data.dtype.names[key]
		if not isinstance(val, str):
			val = self.data.dtype.names[val]

		missing = kwargs.get('missing', None)

		if len(extra_vals) == 0:
			# Attach a name
			dt = [ tuple((val,) + d[1:]) for d in self.data[val].dtype.descr ]
			val = self.data[val].view(dtype=dt)
			if missing is not None:
				missing = (missing,)
		else:
			# Extract a subset of the structured array
			extra_vals = (val,) + extra_vals	# List of column names (or indices)
			val = colgroup.ColGroup()
			for v in extra_vals:
				if not isinstance(v, str):
					v = self.data.dtype.names[v]
				val[v] = self.data[v]

		return Map(self.data[key], val, missing=missing)

def filetable(x, fn, key, val, *extra_vals, **kwargs):
	"""
	# NOTE: In retrospect, I think this function is a bad idea that
	#       encourages bad user behavior. I've disabled it until a
	#       convincing argument changes my mind
	"""
	assert 0, "This function has been deprecated. Use the FileTable object"

	# Keep the three most recently loaded tables available
	# Makes it possible to avoid explicit FileTable initialization, 
	# at the cost of keeping things cached in memory
	self = filetable
	try:
		cache = self.__cache__
	except AttributeError:
		import collections
		cache = self.__cache__ = collections.OrderedDict()

	ckey = fn, key, val, extra_vals, tuple(sorted(kwargs.items()))
	if ckey not in cache:
		# Load table and construct mapping
		m = FileTable(fn, **kwargs).map(key, val, *extra_vals, **kwargs)
	else:
		m = cache[ckey]
		del cache[ckey]

	# Cache and trim cache if needed
	cache[ckey] = m
	if len(cache) > 3:
		del cache[next(iter(cache))]

	# Return
	return m(x)
