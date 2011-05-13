#include <algorithm>
#include <vector>
#include <map>
#include <iostream>
#include <string>
#include <tr1/unordered_map>
#include <tr1/tuple>

#define MULTIMAP std::tr1::unordered_multimap
//#define MULTIMAP std::map

using std::tr1::get;

struct tcomp
{
	typedef std::pair<uint64_t, size_t> T;

	bool operator()(const T &i, const uint64_t &j) const
	{
		return i.first < j;
	}

	bool operator()(const uint64_t &i, const T &j) const
	{
		return i < j.first;
	}
};

template<typename T>
	struct wrapper
	{
		T v;
		explicit wrapper(const T& v_) : v(v_) {}
	};

template<typename T, typename V>
struct argcomp_val_obj
{
	const T *a;

	argcomp_val_obj(const T *a_) : a(a_) {}

	bool operator()(const size_t &i, const wrapper<V> &j) const
	{
		return a[i] < j.v;
	}

	bool operator()(const wrapper<V> &i, const size_t &j) const
	{
		return i.v < a[j];
	}
};

template<typename T, typename V>
	argcomp_val_obj<T, V> argcomp_val(const T*a, const V &v)
	{
		return argcomp_val_obj<T, V>(a);
	}

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
	int table_join_mixed(
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
		MULTIMAP<uint64_t, size_t> i2;	// map from id2->idx2
		for(size_t i = 0; i != nid2; i++)
		{
			i2.insert(std::make_pair(id2[i], i));
		}

		// stream through sorted i1, resolving links as needed
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
					std::pair<typeof(i2.begin()), typeof(i2.begin())> r = i2.equal_range(mm.second);
					if(r.first == r.second)
					{
						if(join == OUTER)
						{
							// Aangling link (where m2 is not found in id2)
							// register a NULL if this is an outer JOIN
							o.push_back(i1[i], 0, true);
						}
					}
					else
					{
						for(typeof(i2.begin()) ii = r.first; ii != r.second; ii++)
						{
							// store the result
							o.push_back(i1[i], ii->second, false);
						}
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

// This appears to be the fastest variant
template<typename Output>
	int table_join_argsort(
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

		// argsort id1
		std::vector<size_t> i1;
		arange(i1, nid1);
		std::sort(i1.begin(), i1.end(), argcomp(id1));

		// argsort id2 (for quick lookups)
		std::vector<size_t> i2;
		arange(i2, nid2);
		std::sort(i2.begin(), i2.end(), argcomp(id2));

		// stream through sorted i1, resolving links as needed
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
					std::pair<typeof(i2.begin()), typeof(i2.begin())> r = std::equal_range(i2.begin(), i2.end(), wrapper<size_t>(mm.second), argcomp_val(id2, mm.second));
					if(r.first == r.second)
					{
						if(join == OUTER)
						{
							// Aangling link (where m2 is not found in id2)
							// register a NULL if this is an outer JOIN
							o.push_back(i1[i], 0, true);
						}
					}
					else
					{
						for(typeof(i2.begin()) ii = r.first; ii != r.second; ii++)
						{
							// store the result
							o.push_back(i1[i], *ii, false);
						}
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

template<typename Output>
	int table_join_sort(
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

		bool sorted;

		// sort (m1, m2)
		std::vector<std::tr1::tuple<uint64_t, uint64_t, uint64_t> > m(nm);
		sorted = true;
		for(size_t i = 0; i != nm; i++)
		{
			m[i] = std::tr1::make_tuple(m1[i], m2[i], i);
			if(i) { sorted &= m1[i] >= m1[i-1]; }
		}
		if(!sorted)
			std::sort(m.begin(), m.end());

		// sort id1
		std::vector<std::pair<uint64_t, size_t> > i1(nid1);
		sorted = true;
		for(size_t i = 0; i != nid1; i++)
		{
			i1[i] = std::make_pair(id1[i], i);
			if(i) { sorted &= id1[i] >= id1[i-1]; }
		}
		if(!sorted)
			std::sort(i1.begin(), i1.end());

		// sort id2
		std::vector<std::pair<uint64_t, size_t> > i2(nid2);
		sorted = true;
		for(size_t i = 0; i != nid2; i++)
		{
			i2[i] = std::make_pair(id2[i], i);
			if(i) { sorted &= id2[i] >= id2[i-1]; }
		}
		if(!sorted)
			std::sort(i2.begin(), i2.end());

		// stream through sorted i1, resolving links as needed
		size_t at = 0, at0, at1;
		for(size_t i = 0; i != nid1; /* incremented at the end */)
		{
			uint64_t id = i1[i].first;
			size_t idx = i1[i].second;

			// find the corresponding m1 block
			std::tr1::tuple<uint64_t, uint64_t, uint64_t> mm;
			while(at < m.size() && get<0>(mm = m[at]) < id) at++;

			// resolve all id->* links
			at0 = o.size;
			if(get<0>(mm) != id)
			{
				if(join == OUTER)
				{
					// register a NULL if this is an outer JOIN
					o.push_back(idx, 0, true, 0);
				}
			}
			else
			{
				// establish links
				do
				{
					// find the block into which we map
					std::pair<typeof(i2.begin()), typeof(i2.begin())> r = std::equal_range(i2.begin(), i2.end(), get<1>(mm), tcomp());
					if(r.first == r.second)
					{
						if(join == OUTER)
						{
							// Dangling link (where m2 is not found in id2)
							// register a NULL if this is an outer JOIN
							o.push_back(idx, 0, true, 0);
						}
					}
					else
					{
						for(typeof(i2.begin()) ii = r.first; ii != r.second; ii++)
						{
							// store the result
							o.push_back(idx, ii->second, false, get<2>(mm));
						}
					}

					mm = m[++at];
				} while(get<0>(mm) == id);
			}
			at1 = o.size;

			// if there are repeat copies of id in id1, just duplicate the output
			while(++i != nid1 && id == i1[i].first)
			{
				idx = i1[i].second;
				// duplicate the block
				for(size_t j = at0; j != at1; j++)
				{
					o.push_back(idx, o.idx2[j], o.isnull[j], o.idxLink[j]);
				}
			}
		}

		return 0;
	}

template<typename Output>
	int table_join_hashjoin(
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

///		--- COMMENTED OUT: Actually made it slower in real life (??)
//		// guess the output size
//		o.resize(std::max(nid1, nid2));

		// build hash table for (m1, m2)
		typedef MULTIMAP<uint64_t, uint64_t> h2_t;
		h2_t h2;
		h2_t::iterator at = h2.begin();
		for(size_t i = 0; i != nid2; i++)
		{
			at = h2.insert(at, std::make_pair(id2[i], i));
		}
		
		// build hash table for m1->[idx2]
		typedef MULTIMAP<uint64_t, uint64_t> hm_t;
		hm_t hm;
		for(size_t i = 0; i != nm; i++)
		{
			uint64_t m = m1[i];
			std::pair<h2_t::iterator, h2_t::iterator> r = h2.equal_range(m2[i]);

			hm_t::iterator at = hm.begin();
			while(r.first != r.second)
			{
				at = hm.insert(at, std::make_pair(m, r.first->second));
				r.first++;
			}
		}

		// stream through id1 and create links
		for(size_t i = 0; i != nid1; i++)
		{
			std::pair<hm_t::iterator, hm_t::iterator> r = hm.equal_range(id1[i]);

			if(r.first == r.second)
			{
				if(join == OUTER)
				{
					o.push_back(i, 0, true);
				}
			}
			else
			{
				do
				{
					o.push_back(i, r.first->second, false);
					r.first++;
				}
				while(r.first != r.second);
			}
		}

		return 0;
	}

#define table_join table_join_sort
