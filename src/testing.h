
#ifndef _TESTING_
#define _TESTING_

#include <vector>
#include <algorithm>
#include <fstream>	
#include "trees.h"
		

// Randomly sample a percentage of nr indices and return the vector of samples as well as 
// the vector of remaining indices 		
static inline std::pair< std::vector<std::size_t>, std::vector<std::size_t> > 
        randomSampling(const std::size_t& nr, const double& pc)
{
	std::vector<std::size_t> v1(nr), v2;
	std::iota(v1.begin(), v1.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(v1.begin(), v1.end(), gen);
	std::size_t segment = ceil(nr*pc/100);
	
	v2.insert(v2.end(), std::make_move_iterator(v1.begin()), 
                    std::make_move_iterator(v1.begin()+segment));
	v1.erase(v1.begin(), v1.begin()+segment);
	
	return std::make_pair(v1, v2);
}		
		
	

// Split a dataset into a training set and a test set by defining a percentage of all data points
// to hold out for testing
template<typename T, typename U>
static inline std::pair< std::pair< std::vector< std::vector<T> >, std::vector<U> >,
				  std::pair< std::vector< std::vector<T> >, std::vector<U> > >
		splitDataset(const std::vector< std::vector<T> >& in, const std::vector<U>& out, const double& pc)
{
	auto pr = randomSampling(in.size(), pc);
	std::vector<std::size_t> trPnts = pr.first, tstPnts = pr.second;
	std::size_t nTr = trPnts.size(), nTst = tstPnts.size();
	std::vector< std::vector<T> > trainIn(nTr), testIn(nTst);
	std::vector<U> trainOut(nTr), testOut(nTst);
	
	for (std::size_t i = 0; i < nTr; ++i)
	{
		trainIn[i] = in[ trPnts[i] ];
		trainOut[i] = out[ trPnts[i] ];
	}
	
	for (std::size_t i = 0; i < nTst; ++i)
	{
		testIn[i] = in[ tstPnts[i] ];
		testOut[i] = out[ tstPnts[i] ];
	}
	
	return std::make_pair( std::make_pair(trainIn, trainOut), std::make_pair(testIn, testOut) );
}





template<typename Tr, typename T, typename U>
static inline double classificationError(Tr tree, std::vector< std::vector<T> > testingInputs, std::vector<U> testingOutputs)
{
	U predictedOutput;
	std::size_t nrTest = testingInputs.size();
	std::size_t incorrect = 0;
	for (std::size_t i = 0; i < nrTest; ++i)
	{
		predictedOutput = tree.predict(testingInputs[i]);
		if (predictedOutput != testingOutputs[i])
			++incorrect;
	}
	return (double) incorrect / nrTest;
}

template<typename T, typename U>
static inline double classificationError(ClassificationTree<T, U> tree, std::vector< std::vector<T> > testingInputs, std::vector<U> testingOutputs)
{
	return classificationError< ClassificationTree<T, U>, T, U >(tree, testingInputs, testingOutputs);
}

template<typename T, typename U>
static inline double classificationError(BaggedClassificationTrees<T, U> trees, std::vector< std::vector<T> > testingInputs, std::vector<U> testingOutputs)
{
	return classificationError< BaggedClassificationTrees<T, U>, T, U >(trees, testingInputs, testingOutputs);
}

template<typename Tr, typename T, typename U>
static inline double meanSquareError(Tr tree, std::vector< std::vector<T> > testingInputs, std::vector<U> testingOutputs)
{
	U predictedOutput;
	std::size_t nrTest = testingInputs.size();
	double sqError = 0;
	for (std::size_t i = 0; i < nrTest; ++i)
	{
		predictedOutput = tree.predict(testingInputs[i]);
		sqError += pow(predictedOutput-testingOutputs[i], 2);
	}
	return (double) sqError / nrTest;
}


template<typename T, typename U>
static inline double meanSquareError(RegressionTree<T, U> tree, std::vector< std::vector<T> > testingInputs, std::vector<U> testingOutputs)
{
	return meanSquareError< RegressionTree<T, U>, T, U >(tree, testingInputs, testingOutputs);
}

template<typename T, typename U>
static inline double meanSquareError(BaggedRegressionTrees<T, U> trees, std::vector< std::vector<T> > testingInputs, std::vector<U> testingOutputs)
{
	return meanSquareError< BaggedRegressionTrees<T, U>, T, U >(trees, testingInputs, testingOutputs);
}








#endif   // _TESTING_