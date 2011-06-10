#!/usr/bin/env python

import os
import time
import errno
import contextlib

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
	# Returns normally only if the lockfile is successfuly created
	while True:
		try:
			os.close(os.open(lockfile, os.O_CREAT | os.O_EXCL))
			return
		except OSError as e:
			if e.errno not in [errno.EEXIST, errno.EINTR]:
				raise

		if retries == 0:
			break

		retries -= 1
		time.sleep(wait_interval)

	raise LockTimeout("Timed out waiting to lock %s" % (lockfile))

