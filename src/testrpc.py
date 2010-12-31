#!/usr/bin/env python

import xmlrpclib
import numpy as np

s = xmlrpclib.ServerProxy('http://localhost:1024')
#print s.pow(2,3)  # Returns 2**3 = 8
#print s.add([2],[3])  # Returns 5
#print s.div(5,2)  # Returns 5//2 = 2
#print s.mad(1, 2, 3)

# Print list of available methods
print s.system.listMethods()

#s.shutdown()
