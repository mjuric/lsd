# Note: In the installed package, this file will be regenerated
# by setup.py with the correct version information.

__all__ = ['__version__', '__version_info__']

# Return the git revision as a string
def git_describe():
	""" Partly lifted from numpy's setup.py """
	def _minimal_ext_cmd(cmd):
		# construct minimal environment
		env = {}
		for k in ['SYSTEMROOT', 'PATH']:
			v = os.environ.get(k)
			if v is not None:
				env[k] = v
		# LANGUAGE is used on win32
		env['LANGUAGE'] = 'C'
		env['LANG'] = 'C'
		env['LC_ALL'] = 'C'
		out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
		return out

	import os, subprocess
	base, _ = os.path.split(__file__)

	tag = "<<unknown>>"
	try:
		cmd = ['git', '--work-tree', '%s/../..' % base, 'describe', '--long', '--abbrev=8']
		tag = _minimal_ext_cmd(cmd)
	except OSError:
		pass

	return tag.strip()

tag = git_describe()
if tag[:1] == 'v':
	tag = tag[1:]
	# Parse the tag into components
	(version, additional_commits, hash) = tag.split('-')
	ver_tuple = tuple( int(v) for v in version.split('.') )
	additional_commits = int(additional_commits)
	hash = hash[1:]

	if additional_commits == 0:
		__version__ = version
	else:
		__version__ = tag

	__version_info__ = (ver_tuple, additional_commits, hash)
else:
	__version__ = tag
	__version_info__ = ((99, 99, 99), 99, '0'*8)
