#!/usr/bin/env python

from lsd import DB

def row_counter_kernel(qresult):
	for rows in qresult:
		yield len(rows)

db = DB('db')
query = db.query("SELECT obj_id FROM ps1_obj, sdss")

total = 0
for subtotal in query.execute([row_counter_kernel]):
	total += subtotal

print "The total number of rows returned by the query '%s' is %d" % (query, total)
