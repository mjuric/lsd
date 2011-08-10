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
		out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=open('/dev/null', 'w'), env=env).communicate()[0]
		return out

	import os, subprocess
	base, _ = os.path.split(__file__)

	tag = "<<unknown>>"
	try:
		# Have to call 'git status' first, because of a git bug where it incorrectly detects
		# a tree as dirty under conditions triggered by things like './setup.py sdist'
		cmd = ['git', '--work-tree', '%s/../..' % base, 'status']
		_minimal_ext_cmd(cmd)

		# Try with the --dirty flag (newer versions of git support this)
		cmd = ['git', '--work-tree', '%s/../..' % base, 'describe', '--long', '--abbrev=8', '--dirty']
		tag = _minimal_ext_cmd(cmd)

		if tag == "":
			# Try without the --dirty flag (for older versions of git)
			cmd = ['git', '--work-tree', '%s/../..' % base, 'describe', '--long', '--abbrev=8']
			tag = _minimal_ext_cmd(cmd)
	except OSError:
		pass

	return tag.strip()

tag = git_describe()
if tag[:1] == 'v':
	tag = tag[1:]
	# Parse the tag into components
	tagparts = tag.split('-')
	(version, additional_commits, hash) = tagparts[:3]
	dirty = len(tagparts) > 3
	ver_tuple = tuple( int(v) for v in version.split('.') )
	additional_commits = int(additional_commits)
	hash = hash[1:]

	if additional_commits == 0 and not dirty:
		__version__ = version
	else:
		__version__ = tag

	__version_info__ = (ver_tuple, additional_commits, hash, dirty)
else:
	__version__ = tag
	__version_info__ = ((99, 99, 99), 99, '0'*8, True)
