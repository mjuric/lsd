#!/usr/bin/env python

import numpy as np
import mrp2p.peer

def mapper(values):
	for v in values:
		yield v

if __name__ == "__main__":
	v = [1, 2, 3]
	pool = mrp2p.peer.Pool('peers')
	for res in pool.map_reduce_chain(v, [mapper]):
		print "Result: ", res
	print "Here!"
