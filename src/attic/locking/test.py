#!/usr/bin/env python

import lsd.locking as locking
import lsd.pool2 as pool2
from multiprocessing import Process

sumfile = '/n/panlfs/mjuric/sumfile.txt'
lockfn = '/n/panlfs/mjuric/locky.lock'

def worker(x):
	""" Example with context manager """
	with locking.lock(lockfn, timeout=None):
		with open(sumfile, "r+") as fp:
			n = int(fp.readline().strip()) + 1
			fp.seek(0)
			fp.truncate()
			fp.write(str(n) + "\n")
	yield n

def worker2(x):
	""" Example without the context manager """

	lock = locking.acquire(lockfn, timeout=None)

	try:
		with open(sumfile, "r+") as fp:
			n = int(fp.readline().strip()) + 1
			fp.seek(0)
			fp.truncate()
			fp.write(str(n) + "\n")
	finally:
		locking.release(lock)

	yield n


if __name__ == "__main__":
	pool = pool2.Pool()
	arr = range(100)
	for result in pool.map_reduce_chain(arr, [worker2], progress_callback=pool2.progress_pass):
		print result
