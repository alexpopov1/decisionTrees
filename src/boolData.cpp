
#include "boolData.h"

void BoolExample::display()
{
	BOOL_VEC::iterator it;
	for (it = ftrs.begin(); it < ftrs.end(); ++it)
		std::cout << *it << " ";
	std::cout << "| " << output << std::endl;
}



int BoolTable::nrFtrs()    					
{	
	if (data.size() == 0)
		return 0;
	return data[0].nrFtrs();
}


void BoolTable::addExmpl(BoolExample exmpl)       
{
	if (BoolTable::nrFtrs() > 0 && exmpl.nrFtrs() != BoolTable::nrFtrs())
		throw std::length_error("Error: New example has incorrect number of features!\n");
	data.push_back(exmpl);
}




void BoolTable::display()
{
	BOOL_DATA::iterator it;
	for (it = data.begin(); it < data.end(); ++it)
		(*it).BoolExample::display();
	std::cout << std::endl;
}



double BoolTable::entropy(double p)
{
	if (p < 0 || p > 1)
		throw std::invalid_argument("Error: Invalid probability!\n");
	if (p == 0 || p == 1)
		return 0;
	return p*selfInfo(p) + (1-p)*selfInfo(1-p);
}



double BoolTable::avEntropy(int ftrNr)
{
	if (ftrNr < 0 || ftrNr >= nrFtrs())
		throw std::out_of_range("Error: Attempt to access non-existent feature!\n");
	PROB ps = prob(ftrNr);
	return ps["all"] * entropy(ps["true"]) + (1-ps["all"]) * entropy(ps["false"]);
}



PROB BoolTable::prob(int ftrNr)
{
	if (ftrNr < 0 || ftrNr >= nrFtrs())
		throw std::out_of_range("Error: Attempt to access non-existent feature!\n");
		
	int ftrCount = 0, trueCount = 0, falseCount = 0, nr = nrExmpls();
	for (int ex = 0; ex < nr; ++ex)
		if ((*this)[ex][ftrNr])
		{
			++ftrCount;
			if ((*this)[ex].getOut())
				++trueCount;
		}
		else
			if ((*this)[ex].getOut())
				++falseCount;
			
	double trueFtrProb = ftrCount==0 ? 0 : (double)trueCount / ftrCount;
	double falseFtrProb = nr==ftrCount ? 0 : (double)falseCount / (nr-ftrCount);
	double ftrProb = (double)ftrCount / nr;
	
	PROB ps { {"all", ftrProb}, {"false", falseFtrProb},
				{"true", trueFtrProb} };
	return ps;
}



int BoolTable::splitPnt()
{
	int pnt = 0;
	double av, min = avEntropy(pnt);
	for (int i = 1; i < nrFtrs(); ++i)
	{
		av = avEntropy(i);
		if (av < min)
		{
			pnt = i; min = av;
		}
	}
	return pnt;
}



OUT_PAIR BoolTable::outputCheck()
{
	bool uniform = true;
	BOOL_VEC outputs(nrExmpls());
	outputs[0] = (*this)[0].getOut();
	for (int ex = 1; ex < nrExmpls(); ++ex)
	{
		outputs[ex] = (*this)[ex].getOut();
		if (outputs[ex] != outputs[ex-1])
			uniform = false; 
	}
	OUT_PAIR pr = std::make_pair(outputs, uniform);
	return pr;
}



std::pair<BoolTable, BoolTable> BoolTable::split(int pnt)
{
	BoolTable trueTree, falseTree;
	for (int ex = 0; ex < nrExmpls(); ++ex)
		if ((*this)[ex][pnt])
			trueTree.addExmpl((*this)[ex]); 
		else
			falseTree.addExmpl((*this)[ex]); 
	return std::make_pair(trueTree, falseTree);
}



bool BoolTable::uniformInputs()
{
	if (nrExmpls() > 1)
		for (int i = 1; i < nrExmpls(); ++i)
			if ((*this)[i].getFtrs() != (*this)[0].getFtrs())
				return false;
	return true;
}



bool BoolTable::stopVal()
{	
	BOOL_VEC outputs = outputCheck().first;
	int tCnt = 0, fCnt = 0;
	for (int i = 0; i < (int)outputs.size(); ++i)
		if (outputs[i])
			++tCnt;
		else
			++fCnt;
	if (tCnt >= fCnt)
		return true;
	return false;
}