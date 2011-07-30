# LSD-defined functions available within queries

import numpy as np

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
