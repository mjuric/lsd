#!/usr/bin/env python

import numpy as np
import sys
import mrp2p.peer

def mapper(values):
	for v in values:
		yield (v, v+2)

if __name__ == "__main__":
	v = np.arange(2000)

	# Parallel
	pool = mrp2p.peer.Pool('peers')
	res1 = []
	for res in pool.map_reduce_chain(v, [mapper]):
		print >>sys.stderr, "Result: ", res
		res1.append(res)
	res1 = sorted(res1)
	
	# Classic
	res2 = sorted(list(mapper(v)))

	if res1 == res2:
		print "Result OK."
	else:
		print res1
		print res2
		print "Check FAILED"
