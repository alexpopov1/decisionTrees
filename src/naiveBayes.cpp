	
#include "naiveBayes.h"
	
R_VALS NaiveBayes::findRVals()
{
	R_VALS rValues(BoolTable::nrFtrs());
	BOOL_VEC outputs = outputCheck().first;
	int i, posOn, ngOn;
	std::vector<int> posExmpls, ngExmpls;
	
	for (i = 0; i < nrExmpls(); ++i)
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