#!/usr/bin/env python

import numpy as np
import sys
import mrp2p.peer
import lsd.pool2

class MyClass(object):
	nada = "aaa"
	pass

def mapper(v):
	my = MyClass()
	my.nada = 'gaga'

	ntot = 2
	for wt in xrange(0, ntot):
		my.ct = 1./ntot
		yield v % 1024, (my, np.arange(ntot))

def reducer1(kv):
	key, v = kv
	yield key, sum(my.ct for my, _ in v)

def reducer2(kv):
	k, v = kv
	key = 'even' if k % 2 == 0 else 'odd'
	yield key, sum(v)

def reducer(kv):
	k, v = kv
	yield k, sum(v)

if __name__ == "__main__":
	#v = np.arange(128*2**10)
	v = np.arange(16*1024)

	# Parallel
	if True:
		pool = mrp2p.peer.Pool('peers')
	else:
		pool = lsd.pool2.Pool()
	res1 = []
	for res in pool.map_reduce_chain(v, [mapper, reducer1, reducer2, reducer]):
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
