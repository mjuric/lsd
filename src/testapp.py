#!/usr/bin/env python

import numpy as np
import sys
import mrp2p.peer

def mapper(values):
	for v in values:
		yield v % 16, 1

def reducer1(kv):
	k, v = kv
	key = 'even' if k % 2 == 0 else 'odd'
	yield key, sum(v)

def reducer(kv):
	k, v = kv
	yield k, sum(v)

if __name__ == "__main__":
	v = np.arange(2**10)

	# Parallel
	pool = mrp2p.peer.Pool('peers')
	res1 = []
	for res in pool.map_reduce_chain(v, [mapper, reducer1, reducer]):
		print >>sys.stderr, "Result: ", res
		res1.append(res)
	res1 = sorted(res1)
	exit()

	# Classic
	res2 = sorted(list(mapper(v)))

	if res1 == res2:
		print "Result OK."
	else:
		print res1
		print res2
		print "Check FAILED"
