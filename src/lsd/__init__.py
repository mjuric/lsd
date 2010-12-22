"""
The Large Survey Database

Important classes (and methods) to know about
---------------------------------------------
DB : class
    Main database manager object (currently in the join_ops submodule). The
    most important function to know about is DB.query()

Query : class
    The class representing a query (currently in the join_ops submodule).
    The important functions to know about:

        Query.fetch() : execute the query, returning the entire result a
            single ColGroup

        Query.iterate() : execute the query, yielding the results row by
            row, or in blocks of rows (see its return_blocks argument). This
            function is a generator.

        Query.execute() : execute the query, passing its results to a chain
           of MapReduce kernels, yielding back the result of the final kernel.
           This function is a generator.

ColGroup : class
    A functional (and API-wise) equivalent of a numpy structured array, with
    data internally stored in columns. All LSD functions returning query
    results return them as instances of this class.

Table : class
    The class representing an LSD Table. Obtain an instance by calling
    DB.table(). Usually not needed to be used directly
"""
from tasks import *
from join_ops import DB

__version__ = "0.3.1"
