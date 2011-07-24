try:
	from urlgrabber.grabber import URLGrabber, URLGrabError
	import os

	class Fetcher(object):
		def __init__(self, remote):
			self.remote = remote
			self.g = URLGrabber(prefix=self.remote)

		def fetch_to_file(self, src, dest):
			tmp = dest + '.part'
			try:
				self.g.urlgrab(src, filename=tmp, copy_local=1, user_agent='lsd-fetch/1.0')
			except URLGrabError as e:
				raise IOError(str(e))
			os.rename(tmp, dest)

		def fetch(self, src='/'):
			try:
				contents = self.g.urlread(src).strip()
			except URLGrabError as e:
				raise IOError(str(e))
			return contents

		def listdir(self, dir='/'):
			lfn = os.path.join(dir, '.listing')

			contents = self.fetch(lfn)

			return [ s.strip() for s in contents.split() if s.strip() != '' ]

		# Pickling support -- only pickle the remote URL
		def __getstate__(self):
			return self.remote
		def __setstate__(self, remote):
			self.__init__(remote)

except ImportError:
	# If no urlgrabber installed

	class Fetcher(object):
		def __init__(self, remote):
			raise Exception("Install urlgrabber module to use remote-access capabilities")
