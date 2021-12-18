
#ifndef _BAGGING_
#define _BAGGING_

#include "trees.h"


// Implement bootstrap sampling to determine training data to be used for each tree
template<typename T, typename U>
static inline std::tuple< std::vector< std::vector< std::vector<T> > >, 
    std::vector< std::vector<U> >, std::vector< std::set<std::size_t> > >
    bootstrap(const std::vector< std::vector<T> >& in, const std::vector<U>& out, const std::size_t& nr)
{
	std::size_t N = in.size();
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distrib(0, N-1);
	
	std::vector< std::vector< std::vector<T> > > bagInputs(nr);
	std::vector< std::vector<U> > bagOutputs(nr);

	std::vector< std::set<std::size_t> > unusedSamples(N);
	for (std::size_t i = 0; i < N; ++i)
		for (std::size_t j = 0; j < nr; ++j)
			unusedSamples[i].insert(unusedSamples[i].end(), j);
			
	for (std::size_t i = 0; i < nr; ++i)
	{
		bagInputs[i].resize(N);
		bagOutputs[i].resize(N);
		for (std::size_t n = 0; n < N; ++n)
		{
			int pnt = distrib(gen);
			bagInputs[i][n] = in[pnt];
			bagOutputs[i][n] = out[pnt];
			unusedSamples[pnt].erase(i);
		}
	}
	
	return std::make_tuple(bagInputs, bagOutputs, unusedSamples);
}



// Construct all trees for bagging procedure, according to user-defined parameters
template<typename T, typename U, typename Tr>
static inline std::pair< std::vector<Tr>, std::vector< std::set<std::size_t> > > 
    baggingTrees(const std::vector< std::vector<T> >& in, const std::vector<U>& out, 
            std::vector<Tr>& trees, const std::tuple<std::size_t, std::size_t, std::size_t, char>& props)
{
	std::size_t nr = trees.size();
	auto tp = bootstrap(in, out, nr);
	std::vector< std::vector< std::vector<T> > > inputs = std::get<0>(tp);
	std::vector< std::vector<U> > outputs = std::get<1>(tp);
	std::vector< std::set<std::size_t> > unusedSamples = std::get<2>(tp);
	
	std::size_t lf = std::get<0>(props), dp = std::get<1>(props), nrFtrs = std::get<2>(props);
	
	for (std::size_t n = 0; n < nr; ++n)
	{
		trees[n] = Tr(inputs[n], outputs[n]);
		trees[n].setMinLeafSize(lf);
		trees[n].setMaxDepth(dp);
		trees[n].setNrSelectedFeatures(nrFtrs);
		if constexpr (std::is_same< Tr, ClassificationTree<T,U> >::value)
			trees[n].setImpurity(std::get<3>(props));
		trees[n].buildTree();
		std::cout << "made " << n+1 << '\n';
	}
	return std::make_pair(trees, unusedSamples);
	
}	
	
	


// Class defining a set of bagged classification trees
template<typename T, typename U>
class BaggedClassificationTrees
{
	// MEMBER OBJECTS
	
	// Vector of training data inputs
	std::vector< std::vector<T> > inputs;
	
	// Vector of training data outputs
	std::vector<U> outputs;
	
	// Number of examples in training set
	std::size_t N;
	
	// nrOfSamples = # bootstrap samples taken, subD = # features checked at each node
	std::size_t nrOfSamples, minLeafSize{0}, maxDepth, subD;
	
	// Vector of sampled trees
	std::vector< ClassificationTree<T, U> > trees;
	
	std::vector< std::set<std::size_t> > unusedSamples;
	
	char impurity{'e'};
	
public:

	// METHODS 
	
	// Constructor
	BaggedClassificationTrees(const std::vector< std::vector<T> >& in, const std::vector<U>& out, const std::size_t& nr)
	: inputs(in), outputs(out), N(in.size()), nrOfSamples(nr), maxDepth(N) {trees.resize(nr); subD = inputs[0].size();}
	
	
	// Pruning properties of trees
	void setMinLeafSize(const std::size_t& s) {minLeafSize = s;}
	void setMaxDepth(const std::size_t& d) {maxDepth = d;}
	void setImpurity(char c) 
	{
		if (c != 'e' && c != 'g')
			throw std::invalid_argument("Impurity must be either 'e' (entropy) or 'g' (Gini)\n");
		impurity = c;
	}
	std::size_t getMinLeafSize() const {return minLeafSize;}
	std::size_t getMaxDepth() const {return maxDepth;}
	char getImpurity() {return impurity;}
	
	
	// Random feature selection
	void setNrSelectedFeatures(const std::size_t n) {subD = n;}
	std::size_t getNrSelectedFeatures() {return subD;}
	

	// Construct all trees
	void buildTrees()
	{
		std::tuple<std::size_t, std::size_t, std::size_t, char> properties = std::make_tuple(minLeafSize, maxDepth, subD, impurity);
		auto bagData = baggingTrees< T, U, ClassificationTree<T, U> >(inputs, outputs, trees, properties);
		trees = bagData.first; unusedSamples = bagData.second;
	}
	
	
	// Predict new output value for a given input point, based on aggregate of trees
	U predict(const std::vector<T>& in) const
	{
		std::map<U, std::size_t> count;
		for (const auto& tree : trees)
			++count[tree.predict(in)];
			
		std::pair<U, std::size_t> max = std::make_pair(U(), 0);
		for (const auto& el : count)
			if (el.second > max.second)
				max = el;
				
		return max.first;
	}
	
	
	// Calculate out of bag error by testing each training point on the trees which did not sample that point
	double outOfBagError()
	{
		std::size_t pntError, size = 0;
		double error = 0;
		for (std::size_t n = 0; n < N; ++n)
		{
			if (!unusedSamples[n].empty())
			{
				++size;
				pntError = 0;
				for (std::size_t sample : unusedSamples[n])
					if (trees[sample].predict(inputs[n]) != outputs[n])
						++pntError;
				error += pntError / unusedSamples[n].size();	
			}
		}
		return error / size;
	}
};


// Class defining a set of bagged regression trees
template<typename T, typename U>
class BaggedRegressionTrees
{
	
	// MEMBER OBJECTS
	
	// Vector of training data inputs
	std::vector< std::vector<T> > inputs;
	
	// Vector of training data outputs
	std::vector<U> outputs;
	
	// Number of examples in training set
	std::size_t N;
	
	// nrOfSamples = # bootstrap samples taken, subD = # features checked at each node
	std::size_t nrOfSamples, minLeafSize{10}, maxDepth, subD;
	
	// Vector of sampled trees
	std::vector< RegressionTree<T, U> > trees;
	
	std::vector< std::set<std::size_t> > unusedSamples;

	
public:
 
	// METHODS 
	
	// Constructor
	BaggedRegressionTrees(const std::vector< std::vector<T> >& in, const std::vector<U>& out, const std::size_t& nr) 
	: inputs(in), outputs(out), N(in.size()), nrOfSamples(nr), maxDepth(N) {trees.resize(nr); subD = inputs[0].size();}


	// Pruning properties of trees
	void setMinLeafSize(const std::size_t& s) {minLeafSize = s;}
	void setMaxDepth(const std::size_t& d) {maxDepth = d;}
	std::size_t getMinLeafSize() const {return minLeafSize;}
	std::size_t getMaxDepth() const {return maxDepth;}
	
	
	// Random feature selection
	void setNrSelectedFeatures(const std::size_t n) {subD = n;}
	std::size_t getNrSelectedFeatures() {return subD;}
	
	
	// Construct all trees
	void buildTrees()
	{
		std::tuple<std::size_t, std::size_t, std::size_t, char> properties = std::make_tuple(minLeafSize, maxDepth, subD, ' ');
		auto bagData = baggingTrees< T, U, RegressionTree<T, U> >(inputs, outputs, trees, properties);
		trees = bagData.first; unusedSamples = bagData.second;
	}	
	
	
	// Predict new output value for a given input point, based on aggregate of trees
	U predict(const std::vector<T>& in) const
	{
		U sum = 0;
		std::map<U, std::size_t> count;
		for (const auto& tree : trees)
			sum += tree.predict(in);
		
		return sum / nrOfSamples;
	}
	

	// Calculate out of bag error by testing each training point on the trees which did not sample that point
	double outOfBagError()
	{
		std::size_t size = 0;
		double pntError, error = 0;
		for (std::size_t n = 0; n < N; ++n)
			if (!unusedSamples[n].empty())
			{
				++size;
				pntError = 0;
				for (std::size_t sample : unusedSamples[n])
					pntError += pow(trees[sample].predict(inputs[n]) - outputs[n], 2);
				error += pntError / unusedSamples[n].size();	
			}
		return error / size;
	}	
};






#endif   // _BAGGING_
