

#ifndef _NAIVE_
#define _NAIVE_

#include "boolData.h"

// Type definition
typedef std::vector< std::pair<double, double> > R_VALS;

template<typename T>
class NaiveBayes : public BoolTable<T>
{
private:
	// Member variables
	BoolTable<T> dataset;
	R_VALS rValues;
	
	// Methods for classification
	R_VALS findRVals()	                           // Calculate set of R values
	{
		R_VALS rValues(BoolTable<T>::nrFtrs());
		BOOL_VEC outputs = this->outputCheck().first;
		int i, posOn, ngOn;
		std::vector<int> posExmpls, ngExmpls;
		
		for (i = 0; i < this->nrExmpls(); ++i)
			if (outputs[i])
				posExmpls.push_back(i);
			else
				ngExmpls.push_back(i);
		
		for (i = 0; i < (int)rValues.size(); ++i)
		{
			posOn = 0; ngOn = 0;
			for (const int& p : posExmpls)
				if (dataset[p][i])
					++posOn;
			for (const int& n : ngExmpls)
				if (dataset[n][i])
					++ngOn;
		
			rValues[i] = std::make_pair(((double)posOn + 1) / (posExmpls.size() + 2),
										((double)ngOn + 1) / (ngExmpls.size() + 2));		
		}
		return rValues;
	}
	
	
	
	double R(int ftr, bool ftrVal, bool outVal)   // Find a particular R value
	{
		if (outVal)
		{
			if (ftrVal)
				return rValues[ftr].first;
			return 1 - rValues[ftr].first;
		}
			
		if (ftrVal)
			return rValues[ftr].second;
		return 1 - rValues[ftr].second;
	}
	
	


	
public:



	// Constructor
	NaiveBayes(BoolTable<T> dt) 
	: BoolTable<T>(dt), dataset(dt) {rValues = findRVals();}
	
	// Access methods
	R_VALS getRVals() {return rValues;}

	// Classify new input
	bool classify(BOOL_VEC input)
	{
		double posScore = 0, ngScore = 0;
		for (int i = 0; i < (int)input.size(); ++i)
		{
			posScore += log(R(i, input[i], true));
			ngScore += log(R(i, input[i], false));
		}
		if (posScore > ngScore)
			return true;
		return false;
	}
	
};

#endif   // _NAIVE_