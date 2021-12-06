

#ifndef _TREE_
#define _TREE_


#include <vector>
#include <set>
#include <tuple>
#include <map>
#include <unordered_map>
#include <iostream>
#include <stdexcept>
#include <iomanip>       // std::setw(...)
#include <cmath>         // log(...)
#include <algorithm>     // std::stable_sort(...)
#include <numeric>       // std::iota(...);
#include <limits>        // std::numeric_limits<double>::infinity();
#include <iterator>      // std::advance(...)
#include "pbPlots.h"
#include "supportLib.h"




template<typename T>
void print(T v)
{
	for (const auto& i : v)
		std::cout << i << ' ';
	std::cout << '\n';
}







template<typename U>
class TreeNode
{
	std::pair<std::size_t, double> split;
	bool leaf;
	U val;
	TreeNode *L = NULL, *R = NULL;
		
public:
	// Default constructor
	TreeNode() {}
	
	// Constructor for internal node
	TreeNode(std::size_t d, double sp)
	: split(std::make_pair(d, sp)), leaf(false) {}
	
	// Constructor for leaf node
	TreeNode(U vl) : leaf(true), val(vl) {}
	
	// Access methods
	void setL(TreeNode<U>* n) {L = n;}
	void setR(TreeNode<U>* n) {R = n;}
	bool getLeaf() {return leaf;}
	U getVal() {return val;}
	TreeNode* getL() {return L;}
	TreeNode* getR() {return R;}
	std::pair<std::size_t, double> getSplit() {return split;}
	
	// Display information about node
	void display()
	{
		std::cout << std::setw(20) << "Leaf? ";
		if (leaf)
		{
			std::cout << "Yes\n";
			std::cout << std::setw(20) << "Output: " << val << '\n';
		}
		else
		{
			std::cout << "No\n";
			std::cout << std::setw(20) << "Split dimension: " << split.first << '\n';
			std::cout << std::setw(20) << "Split value: " << split.second << '\n';
		}
		std::cout << "   -----------------------\n";
		if (R != NULL)
			R->display();
		if (L != NULL)
			L->display();
	}
	
	

};







template<typename T, typename U>
class ClassTree
{
	
	// MEMBER OBJECTS
	
	// Vector of training example inputs, where each example is a vector
	std::vector< std::vector<T> > data;
	
	// Vector of output data corresponding to example inputs
	std::vector<U> outputs;
	
	// Indices of examples, sorted along each dimension. Consecutive examples are grouped
	// in the same vector if they have the same value in the given dimension.
	std::vector< std::map< std::size_t, std::set< std::size_t> > > indices;
	
	// Vector indicating sorted position of each data point with respect to each dimension
	std::vector< std::vector<std::size_t> > pntLocator;
	
	// Vector of all unique classes to which an input can be classified
	std::vector<U> classes;  
	
	// Maps classes to number of occurrences in training set
	std::unordered_map<U, std::size_t> tally;
	
	// D = # dimensions, N = #examples, K = #classes
	std::size_t D, N, K;      

	// Pointer to TreeNode object representing the root node of the tree
	TreeNode<U>* root;
	
	
	




	// METHODS

	// Sort a vector and returns the new order of the original indices
	std::vector<std::size_t> sortIndices(const std::vector<T>& v)
	{
		std::vector<std::size_t> idx(v.size());
		std::iota(idx.begin(), idx.end(), 0);
		std::stable_sort(idx.begin(), idx.end(),
			[&v](std::size_t i1, std::size_t i2) {return v[i1] < v[i2];});
		return idx;
	}
	
	
	
	// Transpose a vector of vectors (similar to matrix transpose)
	std::vector< std::vector<T> > transposeData()
	{
		std::vector< std::vector<T> > v(D);
		for (std::size_t d = 0; d < D; ++d)
		{
			v[d].resize(N);
			for (std::size_t n = 0; n < N; ++n)
				v[d][n] = data[n][d];
		}
		return v;
	}
	
	
	
	// Find all unique classes, and counts the number of instances of each
	void findClasses()
	{
		for (const U& o : outputs)
			++tally[o];
		for (auto const& m : tally)
			classes.push_back(m.first);
		K = classes.size();
	}
	
	
	
	
	// Create member object 'indices'
	void indicesTable()
	{
		std::vector< std::vector<std::size_t> > idx(D);
		std::vector< std::vector<T> > transpose = transposeData();
		indices.resize(D);
		
		for (std::size_t d = 0; d < D; ++d)
		{
			idx[d] = sortIndices(transpose[d]);
			indices[d][0] = std::set<std::size_t>{idx[d][0]};
			
			std::size_t s = 0;
			for (std::size_t n = 1; n < N; ++n)
				if (data[idx[d][n]][d] == data[ idx[d][n-1] ][d])
					indices[d][s].insert(idx[d][n]);
				else
					indices[d][++s] = std::set<std::size_t>{idx[d][n]};
		}
	}	
	
	
	
	
	
	// Create vector where each element is a vector for a given dimension. Each inner element contains the key for the 
	// corresponding index in the indices map for that dimension, allowing us to locate a data point in the map based 
	// on one iteration through the map at the start, instead of having to iterate the map for every new node.
	void locatePnts()
	{
		pntLocator.resize(D);
		for (std::size_t d = 0; d < D; ++d)
		{
			pntLocator[d].resize(N);
			for (const auto& set : indices[d])
				for (const std::size_t& pnt : set.second)
					pntLocator[d][pnt] = set.first;
			print< std::vector<std::size_t> >(pntLocator[d]);
		}
	}
	




	// Create all possible splits between N data points in D dimensions
	std::vector< std::vector<double> > createSplits(std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds)	
	{	
		std::size_t s;
		std::vector< std::vector<double> > splits(D);
		for (std::size_t d = 0; d < D; ++d)
		{
			splits[d].resize(inds[d].size()-1);
			auto it = inds[d].begin();
			s = 0;

			while (it->first != inds[d].rbegin()->first)
				splits[d][s++] = ((double)data[ *(it->second.begin()) ][d]
								+ (double)data[ *((++it)->second.begin()) ][d]) / 2;
			print< std::vector<double> >(splits[d]);
		}
		return splits;
	}
	
	
	
	
	
	// Create map between classes and number of new output occurrences when moving along one split
	std::unordered_map<U, int> countMap(const std::map<std::size_t, 
											std::set< std::size_t> >::iterator& it)
	{
		std::unordered_map<U, int> count;
		for (const std::size_t& i : it->second)
			++count[ outputs[i] ];
		return count;
	}
	
	
	
	
	// Initialise probabilities in first split for a given dimension
	void initialFracs(std::unordered_map< U, std::pair<int, int> >& lf, 
				std::unordered_map< U, std::pair<int, int> >& rf, const std::size_t& d,
				std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds,
				std::unordered_map<U, std::size_t>& tal, std::size_t& nPts)
	{		
		std::unordered_map<U, int> count = countMap(inds[d].begin());
		for (const U& c : classes)
		{
			const auto& search = count.find(c);
			if (search != count.end())
			{
				lf[c] = std::make_pair( count[c], ((inds[d].begin())->second).size() );
				rf[c] = std::make_pair( tal[c] - count[c], nPts - ((inds[d].begin())->second).size() );
			}
			else
			{
				lf[c] = std::make_pair( 0, ((inds[d].begin())->second).size() );
				rf[c] = std::make_pair( tal[c], nPts - ((inds[d].begin())->second).size() );
			}
		}
	}
	
	
	// Update fraction probabilites between successive splits along a given dimension
	void updateFracs(std::unordered_map< U, std::pair<int, int> >& lf, 
				std::unordered_map< U, std::pair<int, int> >& rf, std::size_t d,
				const std::map<std::size_t, std::set< std::size_t> >::iterator& it)	
	{
		std::unordered_map<U, int> count = countMap(it);
		for (const U& c : classes)
		{
			lf[c].second += (it->second).size();
			rf[c].second -= (it->second).size();
			
			auto search = count.find(c);
			if (search != count.end())
			{
				lf[c].first += count[c];
				rf[c].first -= count[c];
			}
		}
	}
	
	
	// Convert numerator-denominator pair into decimal
	double fracToDec(const std::pair<int, int>& fr)
	{
		return (double) fr.first / fr.second;
	}
	
	
	// Calculate self-information for a given probability
	double selfInfo(const double& p)
	{
		if (p == 0)
			return 0;
		return -log(p); 
	}
	
	
	// Calculate weighted average entropy for a given split
	double weightedEntropy(std::unordered_map< U, std::pair<int, int> > lf, 
				std::unordered_map< U, std::pair<int, int> > rf, std::size_t& nPts)
	{
		double lH = 0, rH = 0;
		double lWght = (double)lf[ classes[0] ].second / nPts;
		double rWght = (double)rf[ classes[0] ].second / nPts;
		
		for (const U& c : classes)
		{
			lH += fracToDec(lf[c]) * selfInfo(fracToDec(lf[c]));
			rH += fracToDec(rf[c]) * selfInfo(fracToDec(rf[c]));
		}
		
		return lWght * lH + rWght * rH;
	}



	// Compare current value to minimum and reassign minimum value and location if necessary
	void compare(const double& current, double& minVal, 
				std::pair<std::size_t, std::size_t>& min, const std::size_t& d, std::size_t s)
	{
		std::cout << "AE(" << d << ", " << s << ") = " << current << '\n';
		if (current < minVal)
		{
			minVal = current;
			min.first = d; 
			min.second = s;
		}
	}




	// Along each dimension, calculate weighted average entropy of each split
	std::pair<std::size_t, std::size_t> chooseSplit(std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds,
							std::unordered_map<U, std::size_t>& tal, std::size_t& nPts)
	{
		std::unordered_map< U, std::pair<int, int> > lfracs, rfracs;         
		double current, minVal = std::numeric_limits<double>::infinity();
		std::pair<std::size_t, std::size_t> min = std::make_pair(-1, -1);
		std::size_t s;
		
		for (std::size_t d = 0; d < D; ++d)
		{
			if (inds[d].size() > 1)
			{
				initialFracs(lfracs, rfracs, d, inds, tal, nPts);
				s = 0;
				for (const U& c : classes)
				{
					std::cout << "(" << d << ", " << s << "): L(" << c << ") = " << lfracs[c].first << "/" << lfracs[c].second
							<< ", R(" << c << ") = " << rfracs[c].first << "/" << rfracs[c].second << "      ";
				}
				std::cout << '\n';
								
				current = weightedEntropy(lfracs, rfracs, nPts);
				compare(current, minVal, min, d, s);

				for (auto it = ++inds[d].begin();
							it != --inds[d].end(); ++it)
				{
					++s;
					updateFracs(lfracs, rfracs, d, it);
					for (const U& c : classes)
					{
						std::cout << "(" << d << ", " << s << "): L(" << c << ") = " << lfracs[c].first << "/" << lfracs[c].second
								<< ", R(" << c << ") = " << rfracs[c].first << "/" << rfracs[c].second << "      ";
					}
					std::cout << '\n';
					current = weightedEntropy(lfracs, rfracs, nPts); 
					compare(current, minVal, min, d, s);
				}
			}				
		}
		return min;
	}
	
	
	
	
	
	// Determine location and value of split
	std::tuple< double, std::size_t, std::size_t >
		split(std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds,
			std::unordered_map<U, std::size_t>& tal, std::size_t& nPts)
	{
		std::vector< std::vector<double> > splits = createSplits(inds);
		std::pair<std::size_t, std::size_t> min = chooseSplit(inds, tal, nPts);
		std::size_t d = min.first, s = min.second;
		std::cout << "d = " << d << ", s = " << s << '\n';
		return std::make_tuple(splits[d][s], d, s);
	}
	
	
	
	
	
	void printInds(std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds,
					std::vector< std::map< std::size_t, std::set< std::size_t> > >& rInds)
	{
		std::cout << "\n----------------------------\n-----------------------------\n";
		for (std::size_t d = 0; d < D; ++d)
		{
			for (auto& i : inds[d])
			{
				std::cout << "[" << i.first << "]->{ ";
				for (auto j : i.second)
					std::cout << j << " ";
				std::cout << "}  ";
			}
			std::cout << '\n';
		}
		std::cout << "--------------------------------\n";
		for (std::size_t d = 0; d < D; ++d)
		{
			for (auto& i : rInds[d])
			{
				std::cout << "[" << i.first << "]->{ ";
				for (auto j : i.second)
					std::cout << j << " ";
				std::cout << "}  ";
			}
			std::cout << '\n';
		}
		std::cout << "----------------------------\n-----------------------------\n";
	}
	
	
	
	
	
	
	
	// Split indices maps into corresponding branches from current node
	std::tuple< std::size_t,
				double,
				std::vector< std::map< std::size_t, std::set< std::size_t> > >,
				std::unordered_map<U, std::size_t>, 
				std::size_t >
					splitIndices(std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds,
									std::unordered_map<U, std::size_t>& tal, std::size_t& nPts)
	{
		// Extract information about split
		const auto tp = split(inds, tal, nPts);
		double splitVal = std::get<0>(tp);
		std::size_t dSplit = std::get<1>(tp), splitPnt = std::get<2>(tp);
		
		// Initialise variables
		std::size_t pos, *loc, rNPts = 0;
		std::set<std::size_t> set;
		std::unordered_map<U, std::size_t> rTal;
		std::vector< std::map< std::size_t, std::set< std::size_t> > > rInds(D);
		
		// Initialise iterators
		auto it = inds[dSplit].begin();
		std::advance(it, inds[dSplit].size()-1);
		auto stop = inds[dSplit].begin();
		std::advance(stop, splitPnt);
		std::cout << "it->first = " << it->first << ", stop->first = " << stop->first << '\n';
		
		// Iterate through region being separated by split
		while (it->first != stop->first)
		{
			// Move separated indices to new map
			pos = it->first; set = it->second;
			--it;  
			rInds[dSplit][pos] = set;			
			inds[dSplit].erase(pos);
			  		
			
			// Fill in amended index maps for the other dimensions
			bool done = false;
			for (std::size_t d = 0; d < D; ++d)
				if (d != dSplit)
				{
					std::cout << "d = " << d << "\n--------------------\n";
					for (auto& el : set)
					{
						// Update counters
						if (!done)
						{
							++rTal[outputs[el]]; --tal[outputs[el]];
							++rNPts; --nPts;
							
						}

						// Access point location
						loc = &pntLocator[d][el];
						
						// Add index to existing set or create new set
						auto key = rInds[d].find(*loc);
						if (key != rInds[d].end())
							key->second.insert(el);
						else
							rInds[d] [*loc]  = std::set<std::size_t>{el};
						
						std::cout << "nPts = " << nPts << ", rNPts = " << rNPts << '\n';
						for (U c : classes)
							std::cout << "tal[" << c << "] = " << tal[c] << "    ";
						std::cout << '\n';
						for (U c : classes)
							std::cout << "rTal[" << c << "] = " << rTal[c] << "    ";
						std::cout << '\n';
				
						// Remove index from set - if it leaves an empty set, then remove set from map
						inds[d][*loc].erase(el);
						if (inds[d][*loc].empty())
							inds[d].erase(*loc);
					}
					done = true;
				}
			std::cout << "it->first = " << it->first << '\n';
		}
		printInds(inds, rInds);
		return std::make_tuple(dSplit, splitVal, rInds, rTal, rNPts);
	}
		
	
	
	
	
	
	
	// Check if a node is a leaf. If it is, also return output value at the leaf.
	std::pair<bool, U> isLeaf(std::unordered_map<U, std::size_t> tal, std::size_t nPts)
	{
		for (const auto& c : tal)
			if (c.second == nPts)
				return std::make_pair(true, c.first);
		return std::make_pair(false, U());
	}
	
	
	
	// Construct the right and left branches from a node and add current node to linked tree data structure
	TreeNode<U>* makeBranches(std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds,
						std::unordered_map<U, std::size_t>& tal, std::size_t& nPts)
	{
		std::pair<bool, U> leaf = isLeaf(tal, nPts);     // check if current node is a leaf
		if (leaf.first)
		{
			TreeNode<U>* n = new TreeNode<U>(leaf.second);  // create pointer to leaf TreeNode object
			return n;
		}	
		
		// Extract information from splitting
		std::cout << "splitIndices(...)\n";
		auto tp = splitIndices(inds, tal, nPts);
		std::size_t dSplit = std::get<0>(tp);    				     // dimension on which to split
		double splitVal = std::get<1>(tp);         					 // value on which to split
		std::vector< std::map< std::size_t, 
			std::set< std::size_t> > > rInds = std::get<2>(tp);      // right indices maps
		std::unordered_map<U, std::size_t> rTal = std::get<3>(tp);   // right tally map
		std::size_t rNPts = std::get<4>(tp);					     // number of points on right of split
			
		// Recursive function call to create each new branch
		TreeNode<U>* nextL = makeBranches(inds, tal, nPts);     // left branch
		TreeNode<U>* nextR = makeBranches(rInds, rTal, rNPts);  // right branch
		
		// Add current node to linked tree data structure
		TreeNode<U>* n = new TreeNode<U>(dSplit, splitVal);  // create pointer to internal TreeNode object 
		n->setL(nextL); n->setR(nextR);                      // set next pointers to child nodes
		return n;

	}

	
	// Build tree with initial call to the recursive function makeBranches(...)
	void buildTree()
	{
		root = makeBranches(indices, tally, N);
		std::cout << "\n\n\n";
		root->display();
	}
	

	

	
	
public:

	ClassTree(std::vector< std::vector<T> > dt, std::vector<U> out)
	: data(dt), outputs(out)
	{
		N = dt.size(); D = dt[0].size();
		findClasses();
		indicesTable();
		locatePnts();

		
		for (std::size_t d = 0; d < D; ++d)
			for (std::size_t s = 0; s < indices[d].size(); ++s)
			{
				std::cout << "indices[" << d << "][" << s << "]: ";
				print< std::set<std::size_t> >(indices[d][s]);
			}
		
		std::cout << "\nInputs:\n";
		for (std::size_t n = 0; n < N; ++n)
			print< std::vector<double> >(data[n]);
		std::cout << "\nOutputs:\n";
		print< std::vector<U> >(outputs);
		std::cout << '\n';
		
		buildTree();
	}
	
	
	U classify(std::vector<T> in)
	{
		TreeNode<U>* node = root;
		
		while (true)
		{
			if (node->getLeaf())
				break;
				
			auto pr = node->getSplit();
			if (in[pr.first] < pr.second)
				node = node->getL();
			else
				node = node->getR();
		}
		return node->getVal();
	}
	
	

	
};

#endif   // _TREE_