#include <Python.h>
#include <numpy/arrayobject.h>
#include "table_join.h"
#include <iostream>

/***************************** Module *******************************/
#define DOCSTR_LSD_NATIVE_MODULE \
"skysurvey.native -- native code accelerators for LSD"

////////////////////////////////////////////////////////////

statichere PyObject *NativeError;

// C++ exception class that sets a Python exception and throws
struct E
{
	E(PyObject *err = NULL, const std::string &msg = "")
	{
		if(err)
		{
			PyErr_SetString(err, msg.c_str());
		}
	}
};

struct PyOutput
{
	/* Aux class for table_join that stores the output directly into NumPy arrays */
	npy_int64 *idx1, *idx2;
	npy_bool *isnull;
	int64_t size, reserved;

	PyArrayObject *o_idx1, *o_idx2, *o_isnull;

	PyOutput() : size(0), reserved(0), o_idx1(NULL), o_idx2(NULL), o_isnull(NULL)
	{
		npy_intp dims = reserved;

		o_idx1   = (PyArrayObject *)PyArray_SimpleNew(1, &dims, PyArray_INT64);
		o_idx2   = (PyArrayObject *)PyArray_SimpleNew(1, &dims, PyArray_INT64);
		o_isnull = (PyArrayObject *)PyArray_SimpleNew(1, &dims, PyArray_BOOL);
	}

	void resize(npy_intp dims)
	{
		PyArray_Dims shape;
		shape.ptr = &dims;
		shape.len = 1;

		if(PyArray_Resize(o_idx1,   &shape, false, NPY_CORDER) == NULL) throw E();
		if(PyArray_Resize(o_idx2,   &shape, false, NPY_CORDER) == NULL) throw E();
		if(PyArray_Resize(o_isnull, &shape, false, NPY_CORDER) == NULL) throw E();

		idx1   = (npy_int64 *)o_idx1->data;
		idx2   = (npy_int64 *)o_idx2->data;
		isnull = (npy_bool  *)o_isnull->data;
		
		reserved = dims;
	}

	void push_back(int64_t i1, int64_t i2, bool in)
	{
		if(size >= reserved)
		{
			resize(2*std::max(size, int64_t(1)));
		}
		idx1[size] = i1;
		idx2[size] = i2;
		isnull[size] = in;
		size++;
	}

	~PyOutput()
	{
		Py_XDECREF(o_idx1);
		Py_XDECREF(o_idx2);
		Py_XDECREF(o_isnull);
	}
};

// Python interface: (idx1, idx2, isnull) = table_join(idx1, idx2, m1, m2, join_type)
#define DOCSTR_TABLE_JOIN \
"idx1, idx2, isnull = table_join(id1, id2, m1, m2)\n\
\n\
Join columns id1 and id2, using linkage information\n\
in (m1, m2).\n\
\n\
:Arguments:\n\
	- id1 : First table key\n\
	- id2 : Second table key\n\
	- m1  : First table link key\n\
	- m2  : Second table link key\n\
\n\
The output will be arrays of indices\n\
idx1, idx2, and isnull such that:\n\
\n\
	id1[idx1], id2[idx2]\n\
\n\
(where indexing is performed in NumPy-like vector sense)\n\
will form the resulting JOIN-ed table.\n\
\n\
If join_type=='inner', the result is roughly equivalent\n\
to the result of the following SQL fragment:\n\
\n\
	SELECT id1, id2 ... WHERE id1 == m1 and m2 == id2\n\
\n\
If join_type=='ouuter', the result will include those\n\
rows where id1 has no id2 counterparts. For such rows\n\
idx2 will be set to 0, but isnull will be true.\n\
\n\
Both id1 and id2 are allowed to have repeated elements.\n\
"
static PyObject *Py_table_join(PyObject *self, PyObject *args)
{
	PyObject *ret = NULL;

	PyObject *id1 = NULL, *id2 = NULL, *m1 = NULL, *m2 = NULL;
	const char *join_type = NULL;

	try
	{
		PyObject *id1_, *id2_, *m1_, *m2_;
		if (! PyArg_ParseTuple(args, "OOOOs", &id1_, &id2_, &m1_, &m2_, &join_type))	throw E(PyExc_Exception, "Wrong number or type of args");

		if ((id1 = PyArray_ContiguousFromAny(id1_, PyArray_UINT64, 1, 1)) == NULL)	throw E(PyExc_Exception, "id1 is not a 1D uint64 NumPy array");
		if ((id2 = PyArray_ContiguousFromAny(id2_, PyArray_UINT64, 1, 1)) == NULL)	throw E(PyExc_Exception, "Could not cast the value of id2 to 1D NumPy array");
		if ((m1  = PyArray_ContiguousFromAny(m1_,  PyArray_UINT64, 1, 1)) == NULL)	throw E(PyExc_Exception, "Could not cast the value of m1 to 1D NumPy array");
		if ((m2  = PyArray_ContiguousFromAny(m2_,  PyArray_UINT64, 1, 1)) == NULL)	throw E(PyExc_Exception, "Could not cast the value of m2 to 1D NumPy array");

		if (PyArray_DIM(m1, 0) != PyArray_DIM(m2, 0))  throw E(PyExc_Exception, "The sizes of len(m1) and len(m2) must be the same");

		#define DATAPTR(type, obj) ((type*)PyArray_DATA(obj))
		PyOutput o;
		table_join(
			o,
			DATAPTR(uint64_t, id1), PyArray_Size(id1),
			DATAPTR(uint64_t, id2), PyArray_Size(id2),
			DATAPTR(uint64_t, m1), DATAPTR(uint64_t, m2), PyArray_Size(m2),
			join_type
		);
		#undef DATAPTR
		o.resize(o.size);

		ret = PyTuple_New(3);
		// because PyTuple will take ownership (and PyOutput will do a DECREF on destruction).
		Py_INCREF(o.o_idx1);
		Py_INCREF(o.o_idx2);
		Py_INCREF(o.o_isnull);
		PyTuple_SetItem(ret, 0, (PyObject *)o.o_idx1);
		PyTuple_SetItem(ret, 1, (PyObject *)o.o_idx2);
		PyTuple_SetItem(ret, 2, (PyObject *)o.o_isnull);
	}
	catch(const E& e)
	{
		ret = NULL;
	}

	Py_XDECREF(id1);
	Py_XDECREF(id2);
	Py_XDECREF(m1);
	Py_XDECREF(m2);

	return ret;
}


static PyMethodDef nativeMethods[] =
{
	{"table_join", (PyCFunction)Py_table_join,   METH_VARARGS, DOCSTR_TABLE_JOIN},
	{NULL}        /* Sentinel */
};

extern "C" PyMODINIT_FUNC initnative(void)
{
	// initialize our module
	PyObject *m = Py_InitModule3("native", nativeMethods, DOCSTR_LSD_NATIVE_MODULE);

	// initialize numpy
	import_array();

	// add our exception type
	NativeError = PyErr_NewException("native.error", NULL, NULL);
	Py_INCREF(NativeError);
	PyModule_AddObject(m, "error", NativeError);
}
