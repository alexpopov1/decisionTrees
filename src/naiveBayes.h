

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
	
	// Methods for classification
	R_VALS findRVals();	                           // Calculate set of R values
	double R(int ftr, bool ftrVal, bool outVal);   // Find a particular R value

public:
	// Constructor
	NaiveBayes(BoolTable dt) 
	: BoolTable(dt), dataset(dt) {rValues = findRVals();}
	
	// Access methods
	R_VALS getRVals() {return rValues;}

	// Classify new input
	bool classify(BOOL_VEC input);
	
};

#endif   // _NAIVE_