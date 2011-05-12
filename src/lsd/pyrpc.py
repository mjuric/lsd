#!/usr/bin/env python

import socket
import threading
import SocketServer as socketserver
import json
import re
import logging
import os

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

class Credentials(dict):
	pass

class SimpleAccessControl:
	users = None		# User->Password dictionary

	def __init__(self, users = dict()):
		self.users = users

	def add_user(self, user, password):
		self.users[user] = password

	def login(self, user, password):
		""" Authorize access of user 'user' to the entire server """
		if user in self.users and self.users[user] == password:
			return Credentials([('user', user)])
		else:
			return Credentials()

	def __auth__(self, creds, func):
		"""
		Authorize access of user with credentials 'creds' to function 'func'
		"""
		if func == "add_user":
			return False
		
		if func == "login":
			return True

		if creds:
			return True

class PyRPCServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
	funcs = None		# Function name->Callable dictionary
	instances = None	# Object instances with functions

	timeout = float(os.getenv("PYRPCTIMEOUT", "30."))		# Timeout before the connection is presumed dead and dropped

	def __init__(self, host, port, *args, **kwargs):
		socketserver.TCPServer.__init__(self, (host, port), PyRPCHandler, *args, **kwargs)
		self.funcs = dict()
		self.instances = set()

	def _get_func(self, func):
		""" Get a pointer to the named function """
		if func in self.funcs:
			return self.funcs[func]
		else:
			for instance in self.instances:
				try:
					return getattr(instance, func)
				except AttributeError:
					pass

		raise RPCError('Unknown function %s' % func)

	def register_instance(self, instance):
		instance._server = self
		self.instances.add(instance)

		for hook in dir(self):
			if hook[:3] != "on_": continue
			try:
				name = "__%s__" % hook[3:]
				setattr(self, hook, getattr(instance, name))
			except AttributeError:
				pass

	def register_function(self, func, name=None):
		func._server = self
		if name is None:
			name = func.__name__
		self.funcs[name] = func

		for hook in dir(self):
			if hook[:3] != "on_": continue
			if name == '__%s__' % hook[3:]:
				setattr(self, hook, func)

	def _dispatch(self, func, args):
		return self._get_func(func)(*args)

	#
	# Hooks, (re)set by register_instance/register_function
	#

	def on_auth(self, creds, func):
		""" Default authorization function """
		return True

	def on_connect(self, client_address):
		return True

	def on_disconnect(self, client_address):
		return True

class PyRPCHandler(socketserver.StreamRequestHandler):
	creds = Credentials()		# The currently logged-in users' credentials

	def encode(self, v):
		return escape_nl(json.dumps(v))

	def decode(self, v):
		return _unicode_to_str(json.loads(unescape_nl(v)))

	def rpc_return(self, code, result):
		self.wfile.write(code + " " + self.encode(result) + "\n")

	def setup(self):
		self.formatted_addr = "%s:%s" % self.client_address
		logger.info("[%s] Connection opened" % self.formatted_addr)

		# Call __connect__ hook in server
		self.server.on_connect(self.client_address)

		self.timeout = self.server.timeout

		return socketserver.StreamRequestHandler.setup(self)

	def finish(self):
		logger.info("[%s] Connection closed" % self.formatted_addr)

		# Call __disconnect_ hook in server
		self.server.on_disconnect(self.client_address)

		return socketserver.StreamRequestHandler.finish(self)

	def handle(self):
		# Wait for procedure call requests
		while True:
			try:
				line = self.rfile.readline()
				if line == '': break			# Client hung up

				line = line.strip()
				if len(line) == 0:
					logger.info("[%s] HEARTBEAT" % (self.formatted_addr,))
					continue			# Empty lines/heartbeats

				tokens = line.split(None, 1) + [ "[]" ]
				func, args = tokens[:2]
				logger.info("[%s] CALL %s %s" % (self.formatted_addr, func, args))

				# Check for authorization
				if func[0] == '_' or not self.server.on_auth(self.creds, func):
					raise RPCError("Access denied")

				# Parse the arguments and call the function
				args = self.decode(args)
				result = self.server._dispatch(func, args)

				# Check for special results
				if isinstance(result, Credentials):
					self.creds, result = result, len(result) != 0

				# Return the result
				self.rpc_return("RESULT", result)

				logger.info("[%s] RESULT %s" % (self.formatted_addr, self.encode(result)))
			except socket.timeout:
				logger.info("[%s] Lost heartbeat - connection timed out." % (self.formatted_addr))
				break
			except Exception as e:
				logger.info("[%s] RESULT %s" % (self.formatted_addr, e))
				self.rpc_return("FAULT", str(e))

class PyRPCProxy(object):
	_sock = None	# Socket connection to server
	_rfile = None	# file-like object for reading from socket
	_wfile = None	# file-like object for writing from socket
	_lock = None	# lock protecting rfile/wfile/sock

	_heatbeat_interval = None	# The interval in which to send heartbeats to the server
	_hbeat = None			# The heatbeat timer thread

	def __init__(self, host, port, heartbeat_interval=10.):
		# Set up the call lock
		self._lock = threading.Lock()
		self._addr = (host, port)
		self._heartbeat_interval = heartbeat_interval

		#self._connect()

	def _connect(self):
		if self._wfile is None:
			self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self._sock.connect(self._addr)

			self._rfile = self._sock.makefile('rb', -1)
			self._wfile = self._sock.makefile('wb', 0)

			# Start the hearbeat thread
			self._heartbeat()

	# Deletion/closure of proxy object
	def close(self):
		self._stop_heartbeat()

		with self._lock:
			try:
				if self._rfile is not None: self._rfile.close()
			except:
				pass

			try:
				if self._wfile is not None: self._wfile.close()
			except:
				pass

			try:
				if self._sock  is not None: self._sock.close()
			except:
				pass

			self._rfile = self._wfile = self._sock = None

	def __del__(self):
		self.close()

	## Heatbeat methods
	def _heartbeat(self):
		try:
			with self._lock:
				self._wfile.write("\n")

				# Schedule next firing
				i = self._heartbeat_interval
				if i is not None:
					self._hbeat = threading.Timer(i, self._heartbeat)
					self._hbeat.daemon = True
					self._hbeat.start()
		except Exception as e:
			print e
			pass

	def _stop_heartbeat(self):
		if self._heartbeat_interval and self._hbeat:
			self._hbeat.cancel()
			self._heartbeat_interval = None
			self._hbeat = None

	# Remote procedure caller
	class _Method:
		def __init__(self, call, func):
			self.__call = call
			self.__func = func

		def __call__(self, *args):
			return self.__call(self.__func, args)

	def __call(self, func, args):
		a = escape_nl(json.dumps(args))

		try:
			self._connect()
			with self._lock:
				self._wfile.write("%s %s\n" % (func, a))	# Request
				line = self._rfile.readline().strip()	# Response
		except socket.error as e:
			# Pass any socket error as RPCError
			raise RPCError(str(e))

		# Check if the server hung up on us
		if line == '':
			self.close()
			raise RPCError("Connection closed")

		# Parse the return
		status, data = line.split(None, 1)
		if status == "RESULT":
			return _unicode_to_str(json.loads(unescape_nl(data)))

		raise RPCError(data)

	def __getattr__(self, func):
		return self._Method(self.__call, func)

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

class TestPyRPCProxy:
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
		server = PyRPCServer(cls.host, cls.port)
		cls.host, cls.port = server.server_address

		server.register_instance(SimpleAccessControl({'mjuric': 'bla'}))
		server.register_function(foo)
		server.register_function(bla)
		server.register_instance(A())
		ip, port = server.server_address
		threading.Thread(target=server.serve_forever).start()
		print "Listening on %s:%d" % (ip, port)

	@classmethod
	def tearDownClass(cls):
		svr = PyRPCProxy(cls.host, cls.port)
		svr.login("mjuric", "bla")
		svr.shutdown()
		svr.close()

	def test_connect(self):
		""" PyRPCProxy: Connect and run """
		# Run a number of calls against the server
		svr = PyRPCProxy(self.host, self.port)

		assert svr.login("mjuric", "bla")
		assert svr.foo(10) == 52
		assert svr.bar() == "bar"
		for s in ['true', '\n', '\\n', '[ a, "foo\nbar"]\nSomething else', 'B\\\nla\nGla\\\n']:
			assert svr.echo(s) == s

		svr.close()


	def test_timeout(self):
		""" PyRPCProxy: Timeouts """
		import time

		# Connect once to reset the timeout to 1 (the new timeout does not apply until
		# the next connection)
		print "Reducing timeout to 1 second"
		svr = PyRPCProxy(self.host, self.port)
		assert svr.login("mjuric", "bla")
		old_timeout = svr.set_timeout(1.)
		svr.close()

		print "Sleeping for 5 sec (heartbeats should keep us alive)"
		svr = PyRPCProxy(self.host, self.port, heartbeat_interval=0.5)
		assert svr.login("mjuric", "bla")
		time.sleep(5)

		print "Stopping heartbeats. We should get disconnected."
		svr._stop_heartbeat()
		time.sleep(1)
		try:
			svr.bar()
			assert 0 == "Should have failed"
		except RPCError as e:
			pass
		svr.close()

		print "Returning timeout to old value"
		svr = PyRPCProxy(self.host, self.port, heartbeat_interval=0.5)
		svr.login("mjuric", "bla")
		svr.set_timeout(old_timeout)
		svr.close()

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
		""" PyRPCProxy: Connection flood """

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
	#x = TestPyRPCProxy()
	#x.test_timeout()
	#exit()

	import os
	format = '%(asctime)s.%(msecs)03d %(name)s[%(process)d] %(levelname)-8s {%(module)s:%(funcName)s}: %(message)s'
	datefmt = '%a, %d %b %Y %H:%M:%S'
	level = logging.DEBUG if (os.getenv("DEBUG", 0) or os.getenv("LOGLEVEL", "info") == "debug") else logging.INFO
	logging.basicConfig(format=format, datefmt=datefmt, level=level)

	host, port = "localhost", 1234
	server = PyRPCServer(host, port)
	server.register_instance(SimpleAccessControl({'mjuric': 'bla'}))
	server.register_function(foo)
	server.register_function(bla)
	server.register_instance(A())
	ip, port = server.server_address
	print "Listening on %s:%d" % (ip, port)

	server.serve_forever()
