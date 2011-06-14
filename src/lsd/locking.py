#!/usr/bin/env python

import os
import time
import errno
import contextlib
import socket

wait_interval = 0.1
max_timeout = int(100*3600*24*365)

class LockTimeout(Exception):
	pass

def acquire(lockfile, timeout=None):
	""" 
		Acquire a lock named 'lockfile', within 'timeout' seconds

		On success, this function will return a lock handle.
		On failure, it will raise an exception describing
		the failure.
		
		LockTimeout will be raised if the attempt to acquire the
		lock times out.

	"""
	if timeout is None:
		timeout = max_timeout # 100yrs should suffice
	retries = int(float(timeout)/wait_interval)

	_lock_acquire(lockfile, retries)
	
	return lockfile

def release(lockfile):
	"""
		Release an acquired lock, 'lockfile'

		The lock MUST have been previously acquired with acquire.
		Bad things will happen otherwise.

		NOTE: If you can, use the context manager 'lock' instead.
	"""
	# Must be called _only_ if the lockfile was successfully obtained
	os.unlink(lockfile)

@contextlib.contextmanager
def lock(lockfile, timeout=None):

	if timeout is None:
		timeout = max_timeout # 100yrs should suffice
	retries = int(timeout/wait_interval)

	# This method will return only if the lock is acquired
	_lock_acquire(lockfile, retries)

	try:
		yield
	finally:
		release(lockfile)

##  Internal ############

def _lock_acquire(lockfile, retries):
	# NFS-safe lockfile creation. Follows the prescription given in open(2)
	# man page.

	# Returns normally only if the lockfile is successfuly created
	tmpname = "%s.%s.%s" % (lockfile, socket.gethostname(), os.getpid())
	while True:
		fd = None
		try:
			fd = os.open(tmpname, os.O_EXCL | os.O_CREAT)		# This must succeed, otherwise it's an error
			try:
				os.link(tmpname, lockfile)			# If this succeeds, we've acquired the lock and should return
				return
			except OSError as e:
				if e.errno not in [errno.EEXIST, errno.EINTR]:	# If failed for EEXIST or EINTR, retry later. Otherwise ...
					if os.fstat(fd).st_nlink != 2:		# If appears to have failed but hardlink count == 2, we've succeeded. Otherwise...
						raise				# An error has happened. pass it along.
		except:
			# Try to clean-up an already acquired lock
			if fd is not None and os.fstat(fd).st_nlink == 2:
				unlink(lockfile)
			raise
		finally:
			if fd is not None:		# Whatever happened, ensure the file descriptor is closed and the temp file unlocked
				os.close(fd)
				os.unlink(tmpname)
			elif os.path.exists(tmpname):	# fd == None and path existing can happen if a signal interrupted fd = os.open() call
				os.unlink(tmpname)	#    after open() executed, but before its result was assigned to fd

		if retries == 0:
			break

		retries -= 1
		time.sleep(wait_interval)

	raise LockTimeout("Timed out waiting to lock %s" % (lockfile))

