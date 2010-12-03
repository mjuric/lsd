#!/usr/bin/env python

from lsd import DB
import numpy as np

def mapper(qresult, bins):
	for rows in qresult:
		counts, _ = np.histogram(rows['dec'], bins)
		for (bin, count) in zip(bins, counts):
			if count != 0:
				yield (bin, count)

def reducer(kv):
	bin, counts = kv
	yield (bin, sum(counts))

db = DB('db')
query = db.query("SELECT dec FROM sdss")

ddec = 10.
bins = np.arange(-90, 90.0001, ddec)

hist = {}
for (bin, count) in query.execute([(mapper, bins), reducer]):
	hist[bin + ddec/2] = count

for binctr in sorted(hist.keys()):
	print "%+05.1f %10d" % (binctr, hist[binctr])

print "Total number of objects:", sum(hist.values())
