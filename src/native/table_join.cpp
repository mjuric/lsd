#include "table_join.h"
#include <cstdlib>
#include <iostream>

/*
	C++ test code for table_join
*/

struct Output
{
	int64_t *idx1, *idx2;
	bool *isnull;
	int64_t size, reserved;
	
	Output() : size(0), reserved(0), idx1(NULL), idx2(NULL), isnull(NULL) {}
	
	void push_back(int64_t i1, int64_t i2, int64_t in)
	{
		if(size+1 >= reserved)
		{
			reserved = 2*std::max(size, int64_t(1));
			idx1   = (int64_t *)realloc(idx1,   sizeof(*idx1)*reserved);
			idx2   = (int64_t *)realloc(idx2,   sizeof(*idx2)*reserved);
			isnull =    (bool *)realloc(isnull, sizeof(*isnull)*reserved);
		}
		idx1[size] = i1;
		idx2[size] = i2;
		isnull[size] = in;
		size++;
	}

	~Output()
	{
		free(idx1); free(idx2); free(isnull);
	}
};

int main(int argc, char **argv)
{
	Output o;
	uint64_t t1[] = {3, 11, 4, 2, 7, 4};
	uint64_t m1[] = {2, 4, 11, 4, 8, 422, 0, 4};
	uint64_t m2[] = {7, 1, 2, 3, 2, 321, 6, 42};
	uint64_t t2[] = {2, 7, 1, 3, 1, 578, 422};

	table_join_hashjoin(o,
		t1, sizeof(t1)/sizeof(t1[0]),
		t2, sizeof(t2)/sizeof(t2[0]),
		m1, m2, sizeof(m2)/sizeof(m2[0]),
		"outer"
	);

	for(int i = 0; i != o.size; i++)
	{
		std::cout << "(" << o.idx1[i] << ") " << t1[o.idx1[i]] << " -- " << t2[o.idx2[i]] << " (" << o.idx2[i] << ")    isnull=" << o.isnull[i] << "\n";
	}
}
