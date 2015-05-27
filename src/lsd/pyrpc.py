#!/usr/bin/env python

import socket
import threading
import SocketServer as socketserver
import json
import re
import logging
import os
import time

from . import utils

logger = logging.getLogger('lsd.pyrpc')

def escape_nl(s):
	r""" Escape any newlines with \n, and backslashes with \\ """
	def repl(match):
		match, = match.groups()
		if match == '\\': return r'\\'
		if match == '\n': return r'\n'
	return re.sub(r'(\n|\\)', repl, s)

def unescape_nl(s):
	""" Unescape a string escaped with escape_nl() """
	def repl(match):
		match, = match.groups()
		if match == r'\\': return '\\'
		if match == r'\n': return '\n'
	return re.sub(r'(\\n|\\\\)', repl, s)

def _unicode_to_str(val):
	"""
		Recursively convert unicode objects to str objects given
		a Python list or dict.
	"""
	if isinstance(val, unicode):
		return val.encode('utf-8')
	elif isinstance(val, list):
		return [ _unicode_to_str(v) for v in val ]
	elif isinstance(val, dict):
		return val.__class__((k, _unicode_to_str(v)) for k, v in val.iteritems())
	else:
		return val


class RPCError(Exception):
	def __init__(self, msg, *args, **kwargs):
		Exception.__init__(self, msg, *args, **kwargs)

class Credentials(utils.Namespace):
	user = None
	access = False

	def __nonzero__(self):
		return bool(self.access)

class AllowAllAccessControl:	# Allow access to everyone, recording their username if they login
	def login(self, user):
		return Credentials(user=user, access=True)

	def __auth__(self, creds, func):
		return True

class SimpleAccessControl:
	users = None		# User->Password dictionary

	def __init__(self, users = dict()):
		self.users = users

	def add_user(self, user, password):
		self.users[user] = password

	def login(self, user, password):
		""" Authorize access of user 'user' to the entire server """
		if user in self.users and self.users[user] == password:
			return Credentials(user=user, access=True)
		else:
			return Credentials()

	def __auth__(self, creds, func):
		"""
		Authorize access of user with credentials 'creds' to function 'func'
		
		Returns:
			- True, if the access is positively granted
			- False, if the access is explicitly denied
			- None, if this object cannot decide whether to grant
			  or deny access
		"""
		if func == "add_user":
			return False

		if func == "login":
			return True

		if creds:
			return True

		return None

class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
	funcs = None		# Function name->Callable dictionary
	instances = None	# Object instances with functions

	timeout = float(os.getenv("PYRPCTIMEOUT", "30."))		# Timeout before the connection is presumed dead and dropped

	def __init__(self, host, port, *args, **kwargs):
		socketserver.TCPServer.__init__(self, (host, port), Handler, *args, **kwargs)
		self.funcs = dict()
		self.instances = set()

	def _get_func(self, func):
		""" Get a pointer to the named function """
		if func in self.funcs:
			return self.funcs[func]
		else:
			f = []
			for instance in self.instances:
				try:
					f.append(getattr(instance, func))
				except AttributeError:
					pass

			# Detect if there are multiple functions registered to the same name,
			# saving the programmer from suffering bugs that'd be otherwise
			# caused by this.
			if len(f) == 1:
				return f[0]
			elif len(f) > 1:
				raise RPCError('Ambiguous function "%s" (%d functions registered under that name)' % (func, len(f)))

		raise RPCError('Unknown function %s' % func)

	def register_instance(self, instance):
		try:
			instance._server = self
		except AttributeError:
			pass

		self.instances.add(instance)

		for hook_dispatcher in dir(self):
			if hook_dispatcher[:3] != "on_": continue
			try:
				name = "__%s__" % hook_dispatcher[3:]
				hook = getattr(instance, name)

				tmp = "_%s_callbacks" % hook_dispatcher[3:]
				callbacks = getattr(self, tmp, [])
				setattr(self, tmp, callbacks)

				callbacks.append(hook)
			except AttributeError:
				pass

	def register_function(self, func, name=None):
		try:
			func._server = self
		except AttributeError:
			pass

		if name is None:
			name = func.__name__

		if name in self.funcs:
			raise Exception("Function names '%s' already registered." % name)

		self.funcs[name] = func

		for hook_dispatcher in dir(self):
			if hook_dispatcher[:3] != "on_": continue
			if name == '__%s__' % hook_dispatcher[3:]:
				tmp = "_%s_callbacks"
				callbacks = getattr(self, tmp, [])
				setattr(self, tmp, callbacks)

				callbacks.append(func)

	def _dispatch(self, func, args, client_address, creds):
		try:
			# Try passing extra context information
			context = utils.Namespace(client_address=client_address, creds=creds)
			return self._get_func(func)(*args, _context=context)
		except TypeError as err:
			return self._get_func(func)(*args)

	#
	# Hooks, with callbacks filled by register_instance/register_function
	#

	def on_auth(self, creds, func):
		""" Authorization function dispatcher """
		for auth in getattr(self, '_auth_callbacks', ()):
			r = auth(creds, func)
			if r is not None:
				return r

		return False

	def on_connect(self, client_address, creds):
		""" On-connect event dispatcher """
		context = utils.Namespace(client_address=client_address, creds=creds)
		for fun in getattr(self, '_connect_callbacks', ()):
			fun(_context=context)

	def on_disconnect(self, client_address, creds):
		context = utils.Namespace(client_address=client_address, creds=creds)
		for fun in getattr(self, '_disconnect_callbacks', ()):
			fun(_context=context)

class Handler(socketserver.StreamRequestHandler):
	creds = Credentials()		# The currently logged-in users' credentials

	def encode(self, v):
		return escape_nl(json.dumps(v))

	def decode(self, v):
		return _unicode_to_str(json.loads(unescape_nl(v)))

	def rpc_return(self, code, result):
		self.wfile.write(code + " " + self.encode(result) + "\n")

	def setup(self):
		self.formatted_addr = "%s:%s" % self.client_address
		logger.info("[%s %s] Connection opened" % (self.formatted_addr, self.creds.user))

		# Call __connect__ hook in server
		self.server.on_connect(self.client_address, self.creds)

		self.timeout = self.server.timeout

		return socketserver.StreamRequestHandler.setup(self)

	def finish(self):
		# Call __disconnect__ hook in server
		self.server.on_disconnect(self.client_address, self.creds)

		logger.info("[%s %s] Connection closed" % (self.formatted_addr, self.creds.user))
		return socketserver.StreamRequestHandler.finish(self)

	def handle(self):
		# Wait for procedure call requests
		while True:
			try:
				line = self.rfile.readline()
				if line == '': break			# Client hung up

				line = line.strip()
				if len(line) == 0:
#					logger.info("[%s %s] HEARTBEAT" % (self.formatted_addr, self.creds.user))
					continue			# Empty lines/heartbeats

				tokens = line.split(None, 1) + [ "[]" ]
				func, args = tokens[:2]
#				logger.info("[%s %s] CALL %s %s" % (self.formatted_addr, self.creds.user, func, args))

				# Check for authorization
				if func[0] == '_' or not self.server.on_auth(self.creds, func):
					raise RPCError("Access denied")

				# Parse the arguments and call the function
				args = self.decode(args)
				result = self.server._dispatch(func, args, self.client_address, self.creds)

				# Check for special results
				if isinstance(result, Credentials):
					if result:
						self.creds = result
					result = bool(result)

				# Return the result
				self.rpc_return("RESULT", result)

#				logger.info("[%s %s] RESULT %s" % (self.formatted_addr, self.creds.user, self.encode(result)))
			except socket.timeout:
				logger.info("[%s %s] Lost heartbeat - connection timed out." % (self.formatted_addr, self.creds.user))
				break
			except Exception as e:
				logger.info("[%s %s] RESULT %s" % (self.formatted_addr, self.creds.user, e))
				self.rpc_return("FAULT", str(e))

class Proxy(object):
	_sock = None		# Socket connection to server
	_rfile = None		# file-like object for reading from socket
	_wfile = None		# file-like object for writing from socket

	_lock = None		# lock protecting rfile/wfile/sock

	_debug_stop_heartbeat = False	# For debugging: set to True to stop the heartbeat thread

	def __init__(self, host, port, heartbeat_interval=5.):
		# Set up the call lock
		self._lock = threading.Lock()
		self._addr = (host, port)

		# Connect
		self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self._sock.connect(self._addr)

		self._rfile = self._sock.makefile('rb', -1)
		self._wfile = self._sock.makefile('wb', 0)

		# Start the hearbeat thread
		th = threading.Thread(target=self._heartbeat_thread, args=(heartbeat_interval,))
		th.daemon = True
		th.start()

	def _heartbeat_thread(self, interval):
		try:
			while not self._debug_stop_heartbeat:
				time.sleep(interval)
				with self._lock:
					if self._sock is None:
						break
					self._wfile.write("\n")
		except Exception as e:
			logger.warning("Lost connection to %s:%s (%s)" % (self._addr[0], self._addr[1], e))
			self._close()
		finally:
			self._debug_stop_heartbeat = False

	def _close(self):
		with self._lock:
			for obj in (self._rfile, self.wfile, self._sock):
				try:
					if obj is not None:
						self.obj.close()
				except:
					pass
			self._rfile = self._wfile = self._sock = None

	def __del__(self):
		self._close()

	# Remote procedure caller
	class _Method:
		def __init__(self, call, func):
			self.__call = call
			self.__func = func

		def __call__(self, *args):
			return self.__call(self.__func, args)

	def _call(self, func, args):
		a = escape_nl(json.dumps(args))

		try:
			with self._lock:
				if not self._sock:
					raise RPCError("Calling a method on a closed connection.")

				self._wfile.write("%s %s\n" % (func, a))	# Request
				line = self._rfile.readline().strip()		# Response
		except socket.error as e:
			# Pass any socket error as RPCError
			raise RPCError(str(e))

		# Check if the server hung up on us
		if line == '':
			self._close()
			raise RPCError("Connection closed by server")

		# Parse the return
		status, data = line.split(None, 1)
		if status == "RESULT":
			return _unicode_to_str(json.loads(unescape_nl(data)))

		raise RPCError(data)

	def __getattr__(self, func):
		return self._Method(self._call, func)

class StatelessProxy(object):
	""" A proxy that acts as if it's connection-less. It will
	    also transparently handle being duplicated by a fork().
	"""
	_proxy = None
	_pid = None

	def __init__(self, *args, **kwargs):
		self._init_args = (args, kwargs)
		self._pid = os.getpid()

	def _startup(self):
		# To be overridden by the user to do any initialization
		# when the connection to the RPC server has been established
		# (typically, logging in)
		pass

	def __getattr__(self, func):
		return Proxy._Method(self._call, func)

	def _call(self, func, args):
		try:
			# Test if we got forked while not looking
			if self._pid != os.getpid():
				self._close()

			if self._proxy is None:
				self._proxy = Proxy(*self._init_args[0], **self._init_args[1])
				self._pid = os.getpid()
				self._startup()

			return self._proxy._call(func, args)
		except RPCError as e:
			# If anything went wrong, delete the proxy
			self._proxy = None
			raise

	def _close(self):
		if self._proxy is not None:
			self._proxy._close()
			self._proxy = None

	def __del__(self):
		self._close()

class CachingProxy(StatelessProxy):
	_defaults_obj = None	# Object whose methods will be called if RPC call fails
	_retry_interval = 0	# Interval during which we won't retry RPC calls after one fails

	_next_try = None
	_cache = None

	def __init__(self, host, port, defaults=None, retry_interval=0, cache_for=0, *args, **kwargs):
		self._defaults_obj = defaults
		self._retry_interval = retry_interval
		self._cache_for = cache_for
		self._cache = {}

		StatelessProxy.__init__(self, host, port, *args, **kwargs)

	def _startup(self):
		if self._defaults_obj:
			try:
				f = self._defaults_obj._startup
			except AttributeError as e:
				return

			return f(self)

	def _call(self, func, args):
		# See if it's cached and unexpired
		try:
			(expires, result) = self._cache[func]
			if expires > time.time():
				return result
		except KeyError:
			expires = None

		try:
			# Invoke the RPC on the remote
			result = StatelessProxy._call(self, func, args)
			cache_for = self._cache_for
		except (RPCError, socket.error) as e:
			# Problem connecting to remote; get a result from defaults
			# unless an old cached value exists
			if expires is None:
				result = getattr(self._defaults_obj, func)(*args)
			cache_for = self._retry_interval

		expires = time.time() + cache_for
		self._cache[func] = (expires, result)

		return result

########### Unit tests

def test_unicode_to_str():
	"_unicode_to_str"
	s = 'foo'
	j = json.dumps(s)
	s2 = _unicode_to_str(json.loads(j))
	assert type(s2) == str and s == s2

def test_escape_nl():
	"(un)escape_nl"
	for s in ['true', '\n', '\\n', '[ a, "foo\nbar"]\nSomething else', 'B\\\nla\nGla\\\n']:
		es = escape_nl(s)
		us = unescape_nl(es)

		print '==\n' + s + '\n== -> ==\n' + es + '\n== -> ==\n' + us + '\n==\n'

		assert us == s

class TestProxy:
	host, port = "localhost", 0

	@classmethod
	def setUpClass(cls):
		# Activate logging
		import os
		format = '%(asctime)s.%(msecs)03d %(name)s[%(process)d] %(levelname)-8s {%(module)s:%(funcName)s}: %(message)s'
		datefmt = '%a, %d %b %Y %H:%M:%S'
		level = logging.DEBUG if (os.getenv("DEBUG", 0) or os.getenv("LOGLEVEL", "info") == "debug") else logging.INFO
		logging.basicConfig(format=format, datefmt=datefmt, level=level)

		# Instantiate a server
		server = Server(cls.host, cls.port)
		cls.host, cls.port = server.server_address

		server.register_instance(SimpleAccessControl({'mjuric': 'bla'}))
		server.register_function(time.time)
		server.register_function(foo)
		server.register_function(bla)
		server.register_instance(A())
		ip, port = server.server_address
		threading.Thread(target=server.serve_forever).start()
		print "Listening on %s:%d" % (ip, port)

	@classmethod
	def tearDownClass(cls):
		svr = Proxy(cls.host, cls.port)
		svr.login("mjuric", "bla")
		svr.shutdown()
		svr._close()

	def test_connect(self):
		""" Proxy: Connect and run """
		# Run a number of calls against the server
		svr = Proxy(self.host, self.port)

		assert svr.login("mjuric", "bla")
		assert svr.foo(10) == 52
		assert svr.bar() == "bar"
		for s in ['true', '\n', '\\n', '[ a, "foo\nbar"]\nSomething else', 'B\\\nla\nGla\\\n']:
			assert svr.echo(s) == s

		svr._close()

	def test_defaults(self):
		""" CachingProxy: Test if defaults work """
		# Run a number of calls against the server

		class Defaults(object):
			def login(self, user, passwd):
				return True

			def echo(self, x):
				return "FB: %s" % x

		svr = CachingProxy(self.host, self.port+1, defaults=Defaults())
		assert svr.login("mjuric", "bla")
		assert svr.echo("aa") == "FB: aa"

		svr._close()

	def test_caching(self):
		""" CachingProxy: Test if caching works """
		# Run a number of calls against the server

		class Defaults(object):
			def _startup(self, proxy):
				proxy.login("mjuric", "bla")

		svr = CachingProxy(self.host, self.port, defaults=Defaults(), cache_for=4)
		assert svr.echo("aa") == "aa"
		t0 = svr.time()
		time.sleep(2)
		t1 = svr.time()
		assert t0 == t1
		time.sleep(3)
		t2 = svr.time()
		assert t0 != t2

		svr._close()

	def test_inexistent(self):
		""" Proxy: Invoke a function that does not exist """
		# Run a number of calls against the server
		svr = Proxy(self.host, self.port)

		assert svr.login("mjuric", "bla")

		try:
			svr.not_there()
			assert 0, "Should have failed"
		except RPCError as e:
			pass

		svr._close()

	def test_timeout(self):
		""" StatelessProxy: Timeouts """
		import time

		# Connect once to reset the timeout to 1 (the new timeout does not apply until
		# the next connection)
		print "Reducing timeout to 1 second"
		svr = Proxy(self.host, self.port)
		assert svr.login("mjuric", "bla")
		old_timeout = svr.set_timeout(1.)
		svr._close()

		class MyProxy(StatelessProxy):
			def _startup(self):
				print "In startup, logging in"
				assert self.login("mjuric", "bla")

		print "Sleeping for 3 sec (heartbeats should keep us alive)"
		svr = MyProxy(self.host, self.port, heartbeat_interval=0.5)
		assert svr.echo("All OK") == "All OK"
		time.sleep(3)

		print "Stopping heartbeats. We should get disconnected."
		svr._proxy._debug_stop_heartbeat = True
		while svr._proxy._debug_stop_heartbeat:	# Wait for the heartbeat thread to exit
			time.sleep(0.2)
		time.sleep(2)				# Wait for the server to kick us out
		try:
			svr.bar()
			assert 0, "Should have failed"
		except RPCError as e:
			pass

		print "Reconnecting and checking if heartbeats will restart."
		assert svr.echo("All OK") == "All OK"
		svr.bar()
		time.sleep(2)
		try:
			svr.bar()
		except RPCError as e:
			assert 0, "Failed to keep connection"
		svr._close()

		print "Returning timeout to old value"
		svr = Proxy(self.host, self.port, heartbeat_interval=0.5)
		svr.login("mjuric", "bla")
		svr.set_timeout(old_timeout)
		svr._close()

	####
	# Launch a number of concurrent connection requests towards the server
	class mt_helper:
		def __init__(self, c):
			self.c = c
			self.e = None

		def run(self):
			try:
				self.c.test_connect()
			except Exception as e:
				self.e = e

	def test_mt_connect(self):
		""" Proxy: Connection flood """

		objs = [ self.mt_helper(self) for _ in xrange(10) ]
		threads = [ threading.Thread(target=o.run) for o in objs ]
		[ th.start() for th in threads ]
		[ th.join() for th in threads ]

		# Test if there was an exception in any of the threads
		for o in objs:
			if o.e is not None:
				raise o.e

def foo(a):
	return 42+a

def bla():
	return "Bla!"

class A(object):
	def bar(self):
		return "bar"

	def echo(self, s):
		return s

	def shutdown(self):
		self._server.shutdown()

	def set_timeout(self, timeout):
		old = self._server.timeout
		self._server.timeout = timeout
		return old

############

if __name__ == "__main__":
	#x = TestProxy()
	#x.test_timeout()
	#exit()

	import os
	format = '%(asctime)s.%(msecs)03d %(name)s[%(process)d] %(levelname)-8s {%(module)s:%(funcName)s}: %(message)s'
	datefmt = '%a, %d %b %Y %H:%M:%S'
	level = logging.DEBUG if (os.getenv("DEBUG", 0) or os.getenv("LOGLEVEL", "info") == "debug") else logging.INFO
	logging.basicConfig(format=format, datefmt=datefmt, level=level)

	host, port = "localhost", 1234
	server = Server(host, port)
	server.register_instance(SimpleAccessControl({'mjuric': 'bla'}))
	server.register_function(foo)
	server.register_function(bla)
	server.register_instance(A())
	ip, port = server.server_address
	print "Listening on %s:%d" % (ip, port)

	server.serve_forever()
