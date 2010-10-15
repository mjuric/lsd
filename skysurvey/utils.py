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

# extract/compact functions by David Zaslavsky from 
# http://stackoverflow.com/questions/783781/python-equivalent-of-phps-compact-and-extract
#
# -- mjuric: modification to extract to ensure variable names are legal
import inspect

legal_variable_characters = ''
for i in xrange(256):
	c = chr(i)
	legal_variable_characters = legal_variable_characters + (c if c.isalnum() else '_')

def compact(*names):
	caller = inspect.stack()[1][0] # caller of compact()
	vars = {}
	for n in names:
		if n in caller.f_locals:
			vars[n] = caller.f_locals[n]
		elif n in caller.f_globals:
			vars[n] = caller.f_globals[n]
	return vars

def extract(vars, level=1):
	caller = inspect.stack()[level][0] # caller of extract()
	for n, v in vars.items():
		n = n.translate(legal_variable_characters)
		caller.f_locals[n] = v   # NEVER DO THIS ;-)

def extract_row(row, level=1):
	caller = inspect.stack()[level][0] # caller of extract()
	for n in row.dtype.names:
		v = row[n]
		n = n.translate(legal_variable_characters)
		caller.f_locals[n] = v   # NEVER DO THIS ;-)

