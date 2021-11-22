

#ifndef _NAIVE_
#define _NAIVE_

#include "boolData.h"

// Type definition
typedef std::vector< std::pair<double, double> > R_VALS;

class NaiveBayes : public BoolTable
{
private:
	// Member variables
	BoolTable dataset;
	R_VALS rValues;
	double R(int ftr, bool ftrVal, bool outVal)
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
	
	// Method to find R values
	R_VALS findRVals();	

public:
	// Constructor
	NaiveBayes(BoolTable dt) 
	: BoolTable(dt), dataset(dt) {rValues = findRVals();}
	
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
		std::cout << "S(1) = " << posScore
				<< "\nS(2) = " << ngScore
				<< '\n';
		
		if (posScore > ngScore)
			return true;
		return false;
	}
	
};

#endif   // _NAIVE_