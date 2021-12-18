

#ifndef _TREES_
#define _TREES_


#include <vector>
#include <set>
#include <tuple>
#include <map>
#include <unordered_map>
#include <iostream>
#include <stdexcept>
#include <iomanip>       // std::setw
#include <cmath>         // log, pow
#include <algorithm>     // std::stable_sort
#include <numeric>       // std::iota
#include <limits>        // std::numeric_limits<double>::infinity()
#include <iterator>      // std::advance
#include <random>        // std::random_device












// Abstract base class which prepares training data for decision tree construction, as well as defining
// certain generic properties of the constructed tree
template<typename T, typename U>
class TreeData
{
	// METHODS

	// Sort a vector and returns the new order of the original indices
	std::vector<std::size_t> sortIndices(const std::vector<T>& v) const 
	{
		std::vector<std::size_t> idx(v.size());
		std::iota(idx.begin(), idx.end(), 0);
		std::stable_sort(idx.begin(), idx.end(),
			[&v](std::size_t i1, std::size_t i2) {return v[i1] < v[i2];});
		return idx;
	}
	
	
	// Transpose data vector of vectors (similar to matrix transpose)
	std::vector< std::vector<T> > transposeInputs() const 
	{
		std::vector< std::vector<T> > v(D);
		for (auto ftr : features)
		{
			v[ftr].resize(N);
			for (std::size_t n = 0; n < N; ++n)
				v[ftr][n] = inputs[n][ftr];
		}
		return v;
	}
	
	
	// Create member object 'indices'
	void indicesTable() 
	{
		std::vector< std::vector<std::size_t> > idx(D);
		std::vector< std::vector<T> > transpose = transposeInputs();
		indices.resize(D);
		
		for (auto ftr : features)
		{
			idx[ftr] = sortIndices(transpose[ftr]);
			indices[ftr][0] = std::set<std::size_t>{idx[ftr][0]};
			
			std::size_t s = 0;
			for (std::size_t n = 1; n < N; ++n)
				if (inputs[idx[ftr][n]][ftr] == inputs[ idx[ftr][n-1] ][ftr])
					indices[ftr][s].insert(idx[ftr][n]);
				else
					indices[ftr][++s] = std::set<std::size_t>{idx[ftr][n]};	
		}
	}	
	
	
	// Create vector where each element is a vector for a given dimension. Each inner element contains the key for the 
	// corresponding index in the indices map for that dimension, allowing us to locate a data point in the map based 
	// on one iteration through the map at the start, instead of having to iterate the map for every new node.
	void locatePnts()
	{
		pntLocator.resize(D);
		for (auto ftr : features)
		{
			pntLocator[ftr].resize(N);
			for (const auto& set : indices[ftr])
				for (const std::size_t& pnt : set.second)
					pntLocator[ftr][pnt] = set.first;
		}
	}
	
	
	

protected:
	
	// NESTED CLASS
	
	// An object of this class stores the information necessary to describe a particular node 
	// of a decision tree. This includes pointers to any nodes directly branching from the 
	// current node, analogous to a linked list structure with at most two successors per node.
	class TreeNode
	{
		// Member objects
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
		void setL(TreeNode* n) {L = n;}
		void setR(TreeNode* n) {R = n;}
		bool getLeaf() const {return leaf;}
		U getVal() const {return val;}
		TreeNode* getL() const {return L;}
		TreeNode* getR() const {return R;}
		std::pair<std::size_t, double> getSplit() const {return split;}
		
		// Display information about node
		void display() const 
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




	// METHODS

	// Create all possible splits between N data points in D dimensions
	std::map< std::size_t, std::vector<double> > createSplits(const std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds) const	
	{	
		std::size_t s;
		std::map< std::size_t, std::vector<double> > splits;
		
		for (auto ftr : selectedFeatures)
		{
			splits[ftr] = std::vector<double>(inds[ftr].size()-1);
			auto it = inds[ftr].begin();
			s = 0;
			while (it->first != inds[ftr].rbegin()->first)
				splits[ftr][s++] = ((double)inputs[ *(it->second.begin()) ][ftr]
								+ (double)inputs[ *((++it)->second.begin()) ][ftr]) / 2;
		}
		return splits;
	}
	
	
	// Compare current value to minimum and reassign minimum value and location if necessary
	void compare(const double& current, double& minVal, 
				std::pair<std::size_t, std::size_t>& min, const std::size_t& d, std::size_t& s) const
	{
		if (current < minVal)
		{
			minVal = current;
			min.first = d; 
			min.second = s;
		}
	}
	
	
	// Check if all input points are identical (implying repeated or conflicting data)
	bool identicalInputs(const std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds) const
	{
		std::size_t equal = 0;
		for (auto ftr : inds)
		{
			if (ftr.size() > 1)
				break;
			++equal;
		}
		if (equal == D)
			return true;
		return false;
	}
	

	// Implement random feature selection
	void randomFeatures(const std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds)
	{
		// Identify set of useful features, meaning features on which at least one split is possible
		std::vector<std::size_t> usefulFtrs;
		for (std::size_t d = 0; d < D; ++d)
			if (inds[d].size() > 1)
				usefulFtrs.push_back(d);
		
		std::size_t nrUselessFtrs = D - usefulFtrs.size();
		if (subD <= nrUselessFtrs)
			selectedFeatures = usefulFtrs;
		else
		{
			std::vector<std::size_t> copy = usefulFtrs;
			std::random_device rd;
			std::mt19937 gen(rd());
			std::shuffle(copy.begin(), copy.end(), gen);
			copy.erase(copy.begin()+subD, copy.end());
			selectedFeatures = copy;
		}
	}	

	
	// Pure virtual declaration of function for building tree
	virtual void buildTree() = 0;
	

	
	// MEMBER OBJECTS
	
	// Vector of training example inputs, where each example is a vector
	std::vector< std::vector<T> > inputs;
	
	// Vector of output data corresponding to example inputs
	std::vector<U> outputs;
	
	// Vector of feature numbers
	std::vector<std::size_t> features, selectedFeatures;
	
	// Indices of examples, sorted along each dimension. Consecutive examples are grouped
	// in the same vector if they have the same value in the given dimension.
	std::vector< std::map< std::size_t, std::set< std::size_t> > > indices;
	
	// Vector indicating sorted position of each data point with respect to each dimension
	std::vector< std::vector<std::size_t> > pntLocator;
	
	// D = # dimensions, N = # examples, subD = # features checked at each node
	std::size_t D, N, subD;    
	
	// Tree properties to track for stopping criteria
	std::size_t maxDepth, minLeafSize{0}, depth{0};

	// Pointer to TreeNode object representing the root node of the tree
	TreeNode* root;

	
	
public:
	
	// Constructor
	TreeData(std::vector< std::vector<T> > in, std::vector<U> out)
	: inputs(in), outputs(out)
	{
		N = in.size(); D = in[0].size(); maxDepth = N; subD = D;
		features.resize(D);
		std::iota(features.begin(), features.end(), 0);
		selectedFeatures = features;
		indicesTable();
		locatePnts();	
	}
	
	
	// Default constructor
	TreeData() = default;
	
	
	// Pruning properties of trees
	void setMinLeafSize(const std::size_t& s) {minLeafSize = s;}
	void setMaxDepth(const std::size_t& d) {maxDepth = d;}
	void setNrSelectedFeatures(const std::size_t n) {subD = n; selectedFeatures.resize(subD);}
	std::size_t getMinLeafSize() const {return minLeafSize;}
	std::size_t getMaxDepth() const {return maxDepth;}
	std::size_t getNrSelectedFeatures() {return subD;}
	
	
	// Predict output associated with new input data using tree
	U predict(const std::vector<T>& in) const
	{
		TreeNode* node = root;
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
	
	
	// Display information describing all nodes in tree
	void display() {root->display();}
	
};	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

// A class which constructs a classification tree according to the training data provided
template<typename T, typename U>
class ClassificationTree : public TreeData<T, U>
{

	// MEMBER OBJECTS
	
	// Vector of all unique classes to which an input can be classified
	std::vector<U> classes;  
	
	// Maps classes to number of occurrences in training set
	std::unordered_map<U, std::size_t> tally;

	// Number of unique classes 
	std::size_t K;
	
	// Indicates whether impurity should be measured with entropy ('e') or Gini ('g')
	char impurity{'e'};


	// METHODS
	
	// Find all unique classes, and counts the number of instances of each
	void findClasses()
	{
		for (const U& o : this->outputs)
			++tally[o];
		for (auto const& m : tally)
			classes.push_back(m.first);
		K = classes.size();
	}
	
	
	// Create map between classes and number of new output occurrences when moving along one split
	std::unordered_map<U, int> countMap(const std::map<std::size_t, 
										std::set< std::size_t> >::iterator& it) const
	{
		std::unordered_map<U, int> count;
		for (const std::size_t& i : it->second)
			++count[ this->outputs[i] ];
		return count;
	}
	

	// Initialise probabilities in first split for a given dimension
	void initialFracs(std::unordered_map< U, std::pair<int, int> >& lf, 
				std::unordered_map< U, std::pair<int, int> >& rf, const std::size_t& d,
				std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds,
				std::unordered_map<U, std::size_t>& tal, const std::size_t& nPts) const
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
				const std::map<std::size_t, std::set< std::size_t> >::iterator& it)	const
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
	double fracToDec(const std::pair<int, int>& fr) const
	{
		return (double) fr.first / fr.second;
	}
	
	
	// Calculate self-information for a given probability
	double selfInfo(const double& p) const
	{
		if (p == 0)
			return 0;
		return -log(p); 
	}
	

	// Calculate weighted average entropy for a given split
	double weightedImpurity(std::unordered_map< U, std::pair<int, int> >& lf, 
				std::unordered_map< U, std::pair<int, int> >& rf, const std::size_t& nPts) const
	{
		double lH = 0, rH = 0;
		double lWght = (double)lf[ classes[0] ].second / nPts;
		double rWght = (double)rf[ classes[0] ].second / nPts;
		
		if (impurity == 'e')
			for (const U& c : classes)
			{
				lH += fracToDec(lf[c]) * selfInfo(fracToDec(lf[c]));
				rH += fracToDec(rf[c]) * selfInfo(fracToDec(rf[c]));
			}
		else
			for (const U& c : classes)
			{
				lH += fracToDec(lf[c]) * (1 - fracToDec(lf[c]));
				rH += fracToDec(rf[c]) * (1 - fracToDec(rf[c]));
			}
		return lWght * lH + rWght * rH;
	}

	
	// Along each dimension, calculate weighted average entropy of each split and hence return best split
	std::pair<std::size_t, std::size_t> chooseSplit(std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds,
							std::unordered_map<U, std::size_t>& tal, const std::size_t& nPts) const
	{
		std::unordered_map< U, std::pair<int, int> > lfracs, rfracs;         
		double current, minVal = std::numeric_limits<double>::infinity();
		std::pair<std::size_t, std::size_t> min = std::make_pair(-1, -1);
		std::size_t s;
		
		for (const auto& ftr : this->selectedFeatures)
			if (inds[ftr].size() > 1)
			{
				initialFracs(lfracs, rfracs, ftr, inds, tal, nPts);
				s = 0;
				current = weightedImpurity(lfracs, rfracs, nPts);
				this->compare(current, minVal, min, ftr, s);

				for (auto& it = ++inds[ftr].begin(); it != --inds[ftr].end(); ++it)
				{
					++s;
					updateFracs(lfracs, rfracs, ftr, it);
					current = weightedImpurity(lfracs, rfracs, nPts); 
					this->compare(current, minVal, min, ftr, s);
				}
			}				
		return min;
	}
	

	// Determine location and value of split
	std::tuple< double, std::size_t, std::size_t >
		split(std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds,
			std::unordered_map<U, std::size_t>& tal, const std::size_t& nPts) const
	{
		std::map< std::size_t, std::vector<double> > splits = this->createSplits(inds);
		std::pair<std::size_t, std::size_t> min = chooseSplit(inds, tal, nPts);
		std::size_t d = min.first, s = min.second;
		return std::make_tuple(splits[d][s], d, s);
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
		std::vector< std::map< std::size_t, std::set< std::size_t> > > rInds(this->D);
		
		// Initialise iterators
		auto it = inds[dSplit].begin();
		std::advance(it, inds[dSplit].size()-1);
		auto stop = inds[dSplit].begin();
		std::advance(stop, splitPnt);
		
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

			// for (std::size_t d = 0; d < this->D; ++d)
			for (const auto& ftr : this->features)
				if (ftr != dSplit)
				{
					for (const auto& el : set)
					{
						// Update counters
						if (!done)
						{
							++rTal[this->outputs[el]]; --tal[this->outputs[el]];
							++rNPts; --nPts;
						}

						// Access point location
						loc = &(this->pntLocator[ftr][el]);
						
						// Add index to existing set or create new set
						auto key = rInds[ftr].find(*loc);
						if (key != rInds[ftr].end())
							key->second.insert(el);
						else
							rInds[ftr][*loc]  = std::set<std::size_t>{el};
				
						// Remove index from set - if it leaves an empty set, then remove set from map
						inds[ftr][*loc].erase(el);
						if (inds[ftr][*loc].empty())
							inds[ftr].erase(*loc);
					}
					done = true;
				}
		}
		return std::make_tuple(dSplit, splitVal, rInds, rTal, rNPts);
	}
	
	
	// Check if a node is a leaf. If it is, also return output value at the leaf.
	std::pair<bool, U> isLeaf(const std::unordered_map<U, std::size_t>& tal, const std::size_t& nPts, 
	        const std::size_t& depth, const std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds) const
	{
		// Check if any of the early stopping criteria are satisfied
		if (nPts <= this->minLeafSize || depth >= this->maxDepth || this->identicalInputs(inds))
		{
			std::pair<U, std::size_t> max = std::make_pair(U(), 0);
			for (const auto& c : tal)
				if (c.second > max.second)
					max = c;
			return std::make_pair(true, max.first);
		}	
	
		// Check for unanimous outputs
		for (const auto& c : tal)
			if (c.second == nPts)
				return std::make_pair(true, c.first);
		return std::make_pair(false, U());
		
		
	}	


	// Construct the right and left branches from a node and add current node to linked tree data structure
	typename TreeData<T, U>::TreeNode* makeBranches(std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds,
						std::unordered_map<U, std::size_t>& tal, std::size_t& nPts, std::size_t& depth) 
	{
		// Simplify syntax for inheritance of nested class type
		typedef typename TreeData<T, U>::TreeNode TreeNode;
		
		// Check if node is leaf
		std::pair<bool, U> leaf = isLeaf(tal, nPts, depth, inds);   
		if (leaf.first)
		{
			TreeNode* n = new TreeNode(leaf.second);
			return n;
		}	
		
		// If necessary, randomly select set of features from which to choose split
		if (this->subD < this->D)
			this->randomFeatures(inds);

		// Extract information from splitting
		auto tp = splitIndices(inds, tal, nPts);
		std::size_t dSplit = std::get<0>(tp);    				    
		double splitVal = std::get<1>(tp);         		
		std::vector< std::map< std::size_t, std::set< std::size_t> > > rInds = std::get<2>(tp);     
		std::unordered_map<U, std::size_t> rTal = std::get<3>(tp);   
		std::size_t rNPts = std::get<4>(tp);					  
			
		// Recursive function call to create each new branch
		++depth;
		TreeNode* nextL = makeBranches(inds, tal, nPts, depth);    
		TreeNode* nextR = makeBranches(rInds, rTal, rNPts, depth); 
		
		// Add current node to linked tree data structure
		TreeNode* n = new TreeNode(dSplit, splitVal); 
		n->setL(nextL); n->setR(nextR);              
		return n;

	}

	

	
	
	
	
public:

	// Constructor
	ClassificationTree(std::vector< std::vector<T> > in, std::vector<U> out)
	: TreeData<T, U>(in, out) {findClasses();}
	
	
	// Default constructor
	ClassificationTree() = default;
	
	
	// Access impurity measure
	void setImpurity(char c) 
	{
		if (c != 'e' && c != 'g')
			throw std::invalid_argument("Impurity must be either 'e' (entropy) or 'g' (Gini)\n");
		impurity = c;
	}
	char getImpurity() {return impurity;}
	

	// Build tree with initial call to the recursive function makeBranches(...)
	void buildTree()
	{
		std::size_t depth = 0;
		this->root = makeBranches(this->indices, tally, this->N, depth);
	}
	
};






















// A class which constructs a regression tree according to the training data provided
template<typename T, typename U>
class RegressionTree : public TreeData<T, U>
{
	
	// MEMBER OBJECTS
	
	// totSum = sum of all output values, totSqSum = square sum of all output values
	U totSum, totSqSum;
	
	
	// METHODS
	
	// Calculate values of totSum and totSqSum
	void totalSums()
	{
		totSum = 0; totSqSum = 0;
		U out;
		for (auto& set : this->indices[0])
			for (auto& el : set.second)
			{
				out = this->outputs[el];
				totSum += out;
				totSqSum += pow(out, 2);
			}
	}
	
	
	// Sum and square sum for smallest example input(s) on a given dimension
	void initialSums(U& sum, U& sqsum, U& lsum, U& rsum, U& lsqsum, U& rsqsum, 
			const std::size_t& d,
			const std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds) const
	{
		lsum = 0; lsqsum = 0;
		for (auto el : inds[d].begin()->second)
		{
			lsum += this->outputs[el];
			lsqsum += pow(this->outputs[el], 2);
		}
		rsum = sum - lsum;
		rsqsum = sqsum - lsqsum;
	}
	
	
	// Update sum and square sum along a given dimension 
	void updateSums(U& lsum, U& rsum, U& lsqsum, U& rsqsum, const std::size_t& d,
			const std::map<std::size_t, std::set< std::size_t> >::iterator& it) const
	{
		U out;
		for (auto el : it->second)
		{
			out = this->outputs[el];
			lsum += out; rsum -= out;
			out *= out;
			lsqsum += out; rsqsum -= out;
		}
	}
	
	
	// Calculate weighted variance for a given split
	double weightedVariance(const U& lsum, const U& rsum, const U& lsqsum, const U& rsqsum, 
							const std::size_t& nPts, const std::size_t& lNPts) const
	{
		double lmean, rmean, lVar, rVar, lWght, rWght;
		lmean = (double)lsum / lNPts;
		rmean = (double)rsum / (nPts - lNPts);
		lVar = lsqsum - 2 * lsum * lmean + pow(lmean, 2);
		rVar = rsqsum - 2 * rsum * rmean + pow(rmean, 2);
		lWght = (double)lNPts / nPts;
		rWght = 1 - lWght;
		return lWght * lVar + rWght * rVar;
	}	
	
	
	// Along each dimension, calculate weighted variance of each split and hence return best split
	std::pair<std::size_t, std::size_t> 
			chooseSplit(std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds,
							U& sum, U& sqsum, const std::size_t& nPts) const
	{
		U lsum, rsum, lsqsum, rsqsum;         
		double current, minVal = std::numeric_limits<double>::infinity();
		std::pair<std::size_t, std::size_t> min = std::make_pair(-1, -1);
		std::size_t s, lNPts;
		
		for (const auto& ftr : this->selectedFeatures)
			if (inds[ftr].size() > 1)
			{
				initialSums(sum, sqsum, lsum, rsum, lsqsum, rsqsum, ftr, inds);
				lNPts = 1;
				s = 0;
				current = weightedVariance(lsum, rsum, lsqsum, rsqsum, nPts, lNPts);
				this->compare(current, minVal, min, ftr, s);

				for (auto& it = ++inds[ftr].begin(); it != --inds[ftr].end(); ++it)
				{
					++lNPts;
					++s;
					updateSums(lsum, rsum, lsqsum, rsqsum, ftr, it);
					current = weightedVariance(lsum, rsum, lsqsum, rsqsum, nPts, lNPts);
					this->compare(current, minVal, min, ftr, s);
				}
			}				
		return min;
	}	
	

	// Determine location and value of split
	std::tuple< double, std::size_t, std::size_t >
		split(std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds,
			U& sum, U& sqsum, const std::size_t& nPts) const
	{
		std::map< std::size_t, std::vector<double> > splits = this->createSplits(inds);
		std::pair<std::size_t, std::size_t> min = chooseSplit(inds, sum, sqsum, nPts);
		std::size_t d = min.first, s = min.second;
		return std::make_tuple(splits[d][s], d, s);
	}	


	// Split indices maps into corresponding branches from current node
	std::tuple< std::size_t, double, std::vector< std::map< std::size_t, std::set< std::size_t> > >,
		U, U, std::size_t > 
		splitIndices(std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds,
            U& sum, U& sqsum, std::size_t& nPts)
	{
		
		// Extract information about split
		const auto tp = split(inds, sum, sqsum, nPts);
		double splitVal = std::get<0>(tp);
		std::size_t dSplit = std::get<1>(tp), splitPnt = std::get<2>(tp);
		
		// Initialise variables
		std::size_t pos, *loc, rNPts = 0;
		std::set<std::size_t> set;
		U rSum = 0, rSqSum = 0;
		std::vector< std::map< std::size_t, std::set< std::size_t> > > rInds(this->D);
		
		// Initialise iterators
		auto it = inds[dSplit].begin();
		std::advance(it, inds[dSplit].size()-1);
		auto stop = inds[dSplit].begin();
		std::advance(stop, splitPnt);
		
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
			for (const auto& ftr : this->features)
				if (ftr != dSplit)
				{
					for (const auto& el : set)
					{
						// Update counters
						if (!done)
						{
							rSum += this->outputs[el]; sum -= this->outputs[el];
							rSqSum += pow(this->outputs[el], 2);
							sqsum -= pow(this->outputs[el], 2);
							++rNPts; --nPts;						}
						
						// Access point location
						loc = &(this->pntLocator[ftr][el]);
						
						// Add index to existing set or create new set
						auto key = rInds[ftr].find(*loc);
						if (key != rInds[ftr].end())
							key->second.insert(el);
						else
							rInds[ftr] [*loc]  = std::set<std::size_t>{el};
				
						// Remove index from set - if it leaves an empty set, then remove set from map
						inds[ftr][*loc].erase(el);
						if (inds[ftr][*loc].empty())
							inds[ftr].erase(*loc);
					}
					done = true;
				}
		}
		return std::make_tuple(dSplit, splitVal, rInds, rSum, rSqSum, rNPts);
	}	
	
	
	// Check if a node is a leaf. If it is, also return output value at the leaf.
	std::pair<bool, double> isLeaf(const U& sum, const std::size_t& nPts, const std::size_t& depth,
	            const std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds) const
	{
		if (nPts <= this->minLeafSize || depth >= this->maxDepth || this->identicalInputs(inds))
			return std::make_pair(true, (double)sum / nPts);
		return std::make_pair(false, 0);
	}
	
	
	// Construct the right and left branches from a node and add current node to linked tree data structure
	typename TreeData<T, U>::TreeNode* makeBranches(std::vector< std::map< std::size_t, std::set< std::size_t> > >& inds,
						U& sum, U& sqsum, std::size_t& nPts, std::size_t& depth)
	{
		// Simplify syntax for inheritance of nested class type
		typedef typename TreeData<T, U>::TreeNode TreeNode;
		
		// Check if node is leaf
		std::pair<bool, double> leaf = isLeaf(sum, nPts, depth, inds);
		if (leaf.first)
		{
			TreeNode* n = new TreeNode(leaf.second);
			return n;
		}	
		
		// If necessary, randomly select set of features from which to choose split
		if (this->subD < this->D)
			this->randomFeatures(inds);
	
		// Extract information from splitting
		auto tp = splitIndices(inds, sum, sqsum, nPts);
		std::size_t dSplit = std::get<0>(tp);    			
		double splitVal = std::get<1>(tp);         		
		std::vector< std::map< std::size_t, std::set< std::size_t> > > rInds = std::get<2>(tp);    
		U rSum = std::get<3>(tp);   
		U rSqSum = std::get<4>(tp);
		std::size_t rNPts = std::get<5>(tp);					     
			
		// Recursive function call to create each new branch
		++depth;
		TreeNode* nextL = makeBranches(inds, sum, sqsum, nPts, depth);     
		TreeNode* nextR = makeBranches(rInds, rSum, rSqSum, rNPts, depth);  
		
		// Add current node to linked tree data structure
		TreeNode* n = new TreeNode(dSplit, splitVal); 
		n->setL(nextL); n->setR(nextR);                  
		return n;
	}

	

	


public:

	// Constructor
	RegressionTree(std::vector< std::vector<T> > in, std::vector<U> out)
	: TreeData<T, U>(in, out) 
	{
		totalSums();
		this->setMinLeafSize(10);
	}
	
	
	// Default constructor
	RegressionTree() = default;


	// Build tree with initial call to the recursive function makeBranches(...)
	void buildTree()
	{
		std::size_t depth = 0;
		this->root = makeBranches(this->indices, totSum, totSqSum, this->N, depth);
	}

	
};




#endif     // _TREES_








	