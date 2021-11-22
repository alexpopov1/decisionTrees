	
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



double NaiveBayes::R(int ftr, bool ftrVal, bool outVal)
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



bool NaiveBayes::classify(BOOL_VEC input)
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