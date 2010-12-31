#!/usr/bin/env python

"""
p2p MapReduce Engine for Python

Terminology:
    - Peer:
        A running p2p.Peer() process
    - Directory:
        A directory on the shared filesystems where Peers register
        themselves
    - Cluster:
        The set of Peers registered in the same Directory
    - Client:
        A client submitting tasks to the Cluster, and receiving the results.
    - Coordinator:
        The Peer to which a Client connected and submitted a task, that
        coordinates the execution of the Task
    - Worker:
        A process spawned by the Peer to execute a given MR Task
    - Master:
        The Peer process, when discussed in context of communication with
        its worker

General Design:
    - Peers start up independently, and are given the Directory to register
      in. They register by creating a file with their IP and port on which
      they're listening. They prepend some 'busyness statistic' to the
      filename.
    - A Client looks up the least busy Peer in the Directory, and initiates
      a connection. If the connection is refused, the client retries with a
      different Peer until one is found.
    - Once connected, the Client submits a task to the Peer. The submission
      consists of the module filename where the task code resides, the list of
      kernels from that module that are to be executed, extra local
      variables, and the list of items to map.

    - Upon receiving the submission, the Peer to which the Client is
      connected becomes the Coordinator for the Task. It prepends a "startup
      kernel" to the list of Task kernels, which takes as a single item the
      entire list that was passed by the Client, and emits each item. This
      becomes the 0th step of the task. It appends the "return kernel",
      whose purpose is to collect the final return values of the Task. This
      becomes the n+1st step of the task.
      It assigns a 64-bit task ID to the task, the first 32bits being its
      own IP, and the other 32bits being an ever-increasing integer. It
      stores the list of kernels+all other task data into a 'task struct'.
    - The Coordinator then messages itself to initiate the Task. It next
      messages the Gatherer to start step 0 on the task, and step n+1.
      The n+1st step kernel will launch a HTTP server on which the Client
      can listen to and retrieve the results. The address of this server is
      returned to the Client.
      The Coordinator next connects to the returned Gatherer and sends the
      item to map, followed by the END_OF_STEP sentinel.
    - When a Peer is contacted (by the Coordinator) to initiate a Task, it
      launches a Worker+Gatherer+Scatterer for the Task and returns the
      Gatherer address back to the requesting Peer.
    - When a Gatherer is contacted (by the Coordinator) to initiate a step,
      it just sets a flag that it's active for that particular step.

    - When a Gatherer starts up it a) fork()s a Scatterer, b) performs
      user-defined per-Task initialization (calls a function) and c) fork()s
      a Worker
    - When a Gatherer receives a (step, key, value) pair, it stores them into
      its memory-mapped buffer in the form of a linked list. For each key, it
      keeps an in-memory list of the starting offset in the buffer, and the
      number of items in the buffer.
    - When a Gatherer receives an END_OF_STEP sentinel, it marks all keys
      belonging to that step as 'finalized' -- expecting no more data. Once
      all keys belonging to the step have been processed by the Worker, the
      Gatherer responds to the Worker's query for more with an END_OF_STEP
      sentinel for the given step, and removes the step from the list of
      active steps.
    - The Worker queries the Gatherer for the next steps/keys to reduce,
      letting the Gatherer know which steps/keys it is reducing at the
      moment. The Gatherer responds with a list of new items to reduce for
      those steps/keys, and/or new steps/keys to operate on. The worker
      reduces all of these in an interlieved fashon (i.e., round-robins
      calling next() for each of the active reducers). If a returned step
      has never-before been seen, the worker performs any (user-defined) 
      per-step initialization. When the worker receives an END_OF_STEP 
      sentinel for a given step, it queues it to the Scatterer and performs
      any user-defined per-step cleanup.
    - When the Gatherer is asked by the worker for the next items to operate
      on, it can use the opportunity to compactify the memory-mapped buffer
      if this is deemed to be helpful.
    - When a Scatterer encounters the END_OF_STEP sentinel, it messages the
      Coordinator with an EndOfStep message. The Coordinator records that
      this Worker is processing one less step, as well as that there are one
      less workers processing this particular step.
    - Once the number of active Workers for step k drops to zero, the 
      Coordinator emits an END_OF_STEP sentinel to all Gatherers
      active for step k+1. If this was the last step of a Task, the
      Coordinator messages all active Gatherers to shut down, followed by
      deletion the Task-related structures.

    - The Scatterer listens on the Output FIFO. Data items in the Output
      FIFO are key/value pairs. For each new key, the Scatterer contacts the
      Coordinator to obtain the address of the Gatherer to whom to send the
      data. Otherwise it just uses a previously established connection to
      forward the data.
    - The Coordinator, when queried by a Scatterer where to send a datum for
      (task tid, step sid, key k), either returns the already known remote
      Gatherer address for a key (or a hash thereof) it has already seen, or
      selects the next least busy Peer from the Directory. The Coordinator
      attempts to initiate the Task tid on the remote Peer; if unsuccessful,
      retries with the next Peer from the Directory. If successful, the
      remote Peer will respond with the address of the Gatherer on which the
      task tid, step sid, is receiving data.
    - If the Coordinator is queried for the destination to send the last
      step, it returns the location of its own Gatherer. That Gatherer has
      opened an HTTP port on which the Client can listen for the results.
    - The Gatherers periodically report on their progress by messaging the
      Coordinator.

Output FIFO design:
	- Two memory maps. While the Worker writes in one, the Scatterer
	  reads from the other

        - When the Scatterer finishes forwarding its map, it sets the
          scattererWorking Event to False. It then waits on the Event

        - When the Worker is about to write a value, it checks if
	  scattererWorking == False. If False, it replaces the current map
	  with the (now empty) new map. It sets scattererWorking == True to
	  signal the Scatterer to continue

Requirements:
    - Peer event loop has to be reentrant (i.e., a Peer must be able to
      message itself.)

Schematic of a Peer node and the Worker:

         v   ^
         |   |
     -------------
    | Coordinator |
     ------------- 
          | ^
          v |
 ->-\ -----------     --     -------------     --     ----------- /->-
  ->-| Gatherer  |->-|IF|->-|   Worker    |->-|OF|->-| Scatterer |->-
 ->-/ -----------     --     -------------     --     ----------- \->-


"""

import SocketServer
from SimpleXMLRPCServer import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

class ThreadedXMLRPCServer(SocketServer.ThreadingMixIn, SimpleXMLRPCServer):
	pass

class TestObject:
	def pow(self, x, y):
		return pow(x, y)
 
	def add(self, x, y) :
		return x + y
 
	def div(self, x, y):
		return float(x) / float(y)

	def mul(self, x, y):
		return x * y

	def mad(self, x, y, z):
		import xmlrpclib
		s = xmlrpclib.ServerProxy('http://localhost:5000')
		a = s.mul(x,y)
		return s.add(a, z)

if __name__ == '__main__':
	server = ThreadedXMLRPCServer(('', 5000), SimpleXMLRPCRequestHandler)
	server.register_instance(TestObject())
	server.register_introspection_functions()

	server.serve_forever()
