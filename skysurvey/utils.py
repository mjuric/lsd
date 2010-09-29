import subprocess, os, errno
import numpy as np

def shell(cmd):
	p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	(out, err) = p.communicate();
	if p.returncode != 0:
		err = subprocess.CalledProcessError(p.returncode, cmd)
		raise err
	return out;

def mkdir_p(path):
	''' Recursively create a directory, but don't fail if it already exists. '''
	try:
		os.makedirs(path)
	except OSError as exc: # Python >2.5
		if exc.errno == errno.EEXIST:
			pass
		else:
			raise

def chunks(l, n):
	""" Yield successive n-sized chunks from l.
	    From http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
	"""
	for i in xrange(0, len(l), n):
		yield l[i:i+n]

def astype(v, t):
	""" Typecasting that works for arrays as well as scalars.
	    Note: arrays not being 1st order types in python is truly
	          annoying for scientific applications....
	"""
	if type(v) == np.ndarray:
		return v.astype(t)
	return t(v)

