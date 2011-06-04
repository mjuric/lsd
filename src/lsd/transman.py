#!/usr/bin/env python

"""
A distributed lock manager (DLM) with auto-instantiation.

Example client usage:

	lm = LockManager("db_lms.url")		# 1)
	lm.acquire("mylock")			# 2)
	lm.acquire("mylock", blocking=False)	# 3)
	lm.release("mylock")			# 4)

Line #1 connects to an already running DLM at the location
specified by the "db_lms.url" file. If there's no DLM running, it is
instantiated on the local host and a db_lms.url file is created. An
auto-instantiated DLM will shut down as soon as its last client exits.

Line #2 acquires a lock named "mylock". Any future attempt to acquire a lock
of the same name will block/fail, *including* an attempt by the same client
to do so (i.e., the lock is not recursive).

Line #3 attempts to acquire a lock named "mylock", but does not block if the
lock is not available. Given line #2, it fails to acquire a lock in this
case, returning False.

Line #4 releases the lock named "mylock".

Notes:

The auto-instantiation code assumes a shared filesystem, but the actual
lock/unlock operation are performed directly via TCP (XMLRPC), and do not
depend on the availability of a shared filesystem.

TODO:
- Allow atomic variables/operations on the lock manager
- Allow connections to DLM without a .url file (given url and auth token)

"""

from utils import shell
from multiprocessing import Process, Queue
from contextlib import contextmanager
import os, socket, errno
import mr
import mr.core
import SimpleXMLRPCServer
import threading
import xmlrpclib
import time

class LockManagerServer(object):
	url = None		# URL of the XMLRPC LockManager service
	token = None		# Authentification token (password) for the session
	server = None		# SimpleXMLRPCServer instance

	lock  = None		# Lock for the entire LockManagerServer object (syncronizes access to member variables)
	locks = None		# The locks the server is managing, dict of name -> [threading.Lock(), nclients]
	cli_lock = None		# Synchronization of access to cli_id and clients variables (Lock instance)
	cli_id = None		# Next available client ID (an ever-increasing integer)
	clients = None		# Dict of client_id -> True (for now, later it may map to something more complex)
	
	variables = None	# dict() of shared integer variables (accessible via atomic_get, atomic_set and atomic_add)

	def __init__(self, url, token, server):
		self.url = url
		self.token = token
		self.server = server

		self.lock = threading.Lock()
		self.locks = dict()
		self.cli_id = 0
		self.clients = dict()
		self.cli_lock = threading.Lock()

		self.variables = dict()

	##### Shared variables facility
	def atomic_set(self, cred, var, val):
		with self._lock(cred, "__var_" + var):
			self.variables[var] = val

	def atomic_get(self, cred, var):
		with self._lock(cred, "__var_" + var):
			try:
				return self.variables[var]
			except KeyError:
				return 0

	def atomic_del(self, cred, var):
		with self._lock(cred, "__var_" + var):
			try:
				del self.variables[var]
			except KeyError:
				pass

	def atomic_add(self, cred, var, n):
		""" Set var += n atomically, returning the old value of var """
		with self._lock(cred, "__var_" + var):
			try:
				val = self.variables[var]
			except KeyError:
				val = 0

			self.variables[var] += n

		return val
	#####

	@contextmanager
	def _lock(self, cred, name):
		self.acquire(cred, name)
		yield
		self.release(cred, name)

	def _decref(self, name):
		""" Decrease the reference count of lock 'name' """
		with self.lock:
			lockarr = self.locks[name]
			lock = lockarr[0]

			lockarr[1] -= 1
			if lockarr[1] == 0:
				del self.locks[name]

		return lockarr[0]

	def _addref(self, name):
		""" Increase the reference count of lock 'name' """
		with self.lock:
			try:
				lockarr = self.locks[name]
				lockarr[1] += 1
			except KeyError:
				lockarr = self.locks[name] = [ threading.Lock(), 1 ]

		return lockarr[0]

	def _check_cred(self, cred, token_only=False):
		if token_only:
			token = cred
		else:
			token, _ = cred

		return token == self.token

	def acquire(self, cred, name, blocking=True):
		if not self._check_cred(cred): return None

		lock = self._addref(name)

		if lock.acquire(blocking):
			print "Acquired lock %s" % (name,)
			return True
		else:
			print "Failed to acquire lock %s" % (name,)
			self._decref(name)
			return False

	def release(self, cred, name):
		if not self._check_cred(cred): return None

		lock = self._decref(name)
		print "Released lock %s" % (name,)
		return lock.release()

	def register(self, token):
		if not self._check_cred(token, token_only=True): return None

		with self.cli_lock:
			cli_id = self.cli_id
			self.clients[cli_id] = True
			self.cli_id += 1

		return cli_id

	def unregister(self, cred):
		if not self._check_cred(cred): return None

		_, cli_id = cred
		with self.cli_lock:
			del self.clients[cli_id]
			if len(self.clients) == 0:
				print "Shutting down."
				assert len(self.locks) == 0
				#th = threading.Thread(target=self._delayed_shutdown)
				#th.daemon = True
				#th.start()
				#print threading.current_thread().daemon
				#print "Exiting unregister()."
				self.server.shutdown()

	#def _delayed_shutdown(self):
	#	print "Waiting"
	#	time.sleep(1)
	#	print "Shutting down"
	#	self.server.shutdown()
	#	print "Shut down"

def _spawn_lms(urlfile, queue):
	try:
		print "SPAWNED ", urlfile
		print threading.current_thread()
		hostname = socket.gethostname()

		server, port = mr.core.start_threaded_xmlrpc_server(SimpleXMLRPCServer.SimpleXMLRPCRequestHandler, 1023, hostname, daemon=True)

		token = "blabla" # generate_crypto_safe_hash()
		url   = "http://%s:%s" % (hostname, port)

		lms = LockManagerServer(url, token, server)

		# Write the location and token into urlfile
		with open(urlfile, 'w') as fpw:
			os.chmod(urlfile, 0700)
			fpw.write("%s\n%s\n" % (url, token))

		id = lms.register(token)
		queue.put((url, token, id))

		server.register_instance(lms)
		server.register_introspection_functions()
		print "ENTERING"
		server.serve_forever()
		print "DONE"
	finally:
		try:
			os.unlink(urlfile)
		except OSError as e:
			if e.errno != errno.ENOENT: raise

class LockManager(object):
	lms = None
	cred = None

	def __init__(self, urlfile):
		lfile = urlfile + ".lock"
		shell('lockfile -1 "%s"' % (lfile) )
		try:
			try:
				url, token = [ line.strip() for line in open(urlfile).xreadlines() ]
				lms = xmlrpclib.ServerProxy(url)
				id = self.lms.register(token)
			except:
				# Spawn a new LMS and wait for it to become active
				queue = Queue()
				p = Process(target=_spawn_lms, args=(urlfile, queue,))
				p.daemon = True
				p.start()
				print "Waiting"
				url, token, id = queue.get()
				print "OK"
				lms = xmlrpclib.ServerProxy(url, allow_none=True)

			self.lms = lms
			self.cred = (token, id)
		finally:
			os.unlink(lfile)

	def __del__(self):
		if self.lms is not None:
			self.lms.unregister(self.cred)

	def acquire(self, name, blocking=True):
		return self.lms.acquire(self.cred, name, blocking)

	def release(self, name):
		return self.lms.release(self.cred, name)

	def atomic_get(self, name):
		return self.lms.atomic_get(self.cred, name)

	def atomic_add(self, name, val):
		return self.lms.atomic_add(self.cred, name, val)

	@contextmanager
	def lock(self, name):
		self.acquire(name)
		
		yield True
		
		self.release(name)

if __name__ == "__main__":
	print "Here"
	lm = LockManager("db_lms.url")
	lm.acquire("mylock")
	lm.acquire("mylock", blocking=False)
	lm.release("mylock")
	print "var=", lm.atomic_get("bla")
	print "var=", lm.atomic_add("bla", 2)
	print "var=", lm.atomic_set("bla", -3)
	print "var=", lm.atomic_get("bla")
	import time
	#time.sleep(1)
	del lm
	print "Here"
	#time.sleep(6)
	print "Exit"
