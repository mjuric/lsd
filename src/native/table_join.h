#include <algorithm>
#include <vector>
#include <map>
#include <string>

template<typename T>
	struct argcomp_obj
	{
		const T *a;

		argcomp_obj(const T *a_) : a(a_) {}

		bool operator()(const size_t &i, const size_t &j) const
		{
			return a[i] < a[j];
		}
	};

template<typename T>
	argcomp_obj<T> argcomp(const T*a)
	{
		return argcomp_obj<T>(a);
	}

template<typename T>
	inline void arange(std::vector<T> &x, size_t n)
	{
		x.resize(n);
		for(size_t i = 0; i != n; i++) { x[i] = i; }
	}

template<typename Output>
	int table_join(
		Output &o,
		uint64_t *id1, size_t nid1,
		uint64_t *id2, size_t nid2,
		uint64_t *m1, uint64_t *m2, size_t nm,
		const std::string &join_type)
	{
		/*
			Join columns id1 and id2, using linkage information
			in (m1, m2). The output will be arrays of indices
			idx1, idx2, and isnull such that:
			
				id1[idx1], id2[idx2]
				
			(where indexing is performed in NumPy-like vector sense)
			will form the resulting JOIN-ed table.

			If join_type=="inner", the result is roughly equivalent
			to the result of the following SQL fragment:
			
				SELECT id1, id2 ... WHERE id1 == m1 and m2 == id2

			If join_type=="ouuter", the result will include those
			rows where id1 has no id2 counterparts. For such rows
			idx2 will be set to 0, but isnull will be true.

			Both id1 and id2 are allowed to have repeated elements.
		*/
		const int INNER = 0;
		const int OUTER = 1;
		int join = 0;
		     if(join_type == "inner") { join = INNER; }
		else if(join_type == "outer") { join = OUTER; }
		else return -1;

		// sort m1 and m2
		std::vector<std::pair<uint64_t, uint64_t> > m(nm);
		for(size_t i = 0; i != nm; i++)
		{
			m[i] = std::make_pair(m1[i], m2[i]);
		}
		std::sort(m.begin(), m.end());

		// argsort id1, and make a multimap for id2
		std::vector<size_t> i1;
		arange(i1, nid1);
		std::sort(i1.begin(), i1.end(), argcomp(id1));
		std::multimap<uint64_t, size_t> i2;	// map from id2->idx2
		for(size_t i = 0; i != nid2; i++)
		{
			i2.insert(std::make_pair(id2[i], i));
		}

		// stream through i1, resolving links as needed
		size_t at = 0, at0, at1;
		for(size_t i = 0; i != nid1; /* incremented at the end */)
		{
			uint64_t id = id1[i1[i]];

			// find the corresponding m1 block
			std::pair<uint64_t, uint64_t> mm;
			while(at < m.size() && (mm = m[at]).first < id) at++;

			// resolve all id->* links
			at0 = o.size;
			if(mm.first != id)
			{
				if(join == OUTER)
				{
					// register a NULL if this is an outer JOIN
					o.push_back(i1[i], 0, true);
				}
			}
			else
			{
				// establish links
				do
				{
					// find the block into which we map
					// Note: this conveniently ignores any dangling links (where m2 is not found in id2)
					std::pair<typeof(i2.begin()), typeof(i2.begin())> r = i2.equal_range(mm.second);
					for(typeof(i2.begin()) ii = r.first; ii != r.second; ii++)
					{
						// store the result
						o.push_back(i1[i], ii->second, false);
					}

					mm = m[++at];
				} while(mm.first == id);
			}
			at1 = o.size;

			// if there are repeat copies of id in id1, just duplicate the output
			while(++i != nid1 && id == id1[i1[i]])
			{
				// duplicate the block
				for(size_t j = at0; j != at1; j++)
				{
					o.push_back(i1[i], o.idx2[j], o.isnull[j]);
				}
			}
		}

		return 0;
	}
