try:
	import requests
	import os

	class Fetcher(object):
		def __init__(self, remote):
			self.remote = remote

		def fetch_to_file(self, src, dest):
			url = self.remote + '/' + src
			tmp = dest + '.part'

			r = requests.get(url, stream=True, headers={'user-agent': 'lsd-fetch/1.0'})
			with open(tmp, "wb") as fp:
				for chunk in r.iter_content(chunk_size=50*(1<<20)):
					fp.write(chunk)

			os.rename(tmp, dest)

		def fetch(self, src='/'):
			url = self.remote + '/' + src
			r = requests.get(url, stream=True, headers={'user-agent': 'lsd-fetch/1.0'})
			return r.text.encode('ascii', 'ignore').strip()

		def listdir(self, dir='/'):
			lfn = os.path.join(dir, '.listing')

			contents = self.fetch(lfn)

			return [ s.strip() for s in contents.split() if s.strip() != '' ]

except ImportError:
	# If no requests module is installed, try with urlgrabber
	# WARNING: THIS CODE IS DEPRECATED AND WILL BE REMOVED IN THE FUTURE
	# PLEASE BEGIN USING THE REQUESTS LIBRARY

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
		# If no urlgrabber installed, use dummy

		class Fetcher(object):
			def __init__(self, remote):
				raise Exception("Install urlgrabber module to use remote-access capabilities")
