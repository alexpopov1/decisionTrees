


#ifndef _BOOL_
#define _BOOL_



#include <vector>
#include <iostream>
#include <iomanip>
#include <utility>
#include <cmath>
#include <map>
#include <string>
#include <stdexcept>
#include <algorithm>



// Print vector
template<typename T>
void print(std::vector<T> v)
{
	for (int i = 0; i < (int)v.size(); ++i)
		std::cout << std::setw(5) << v[i];
	std::cout << std::endl;
}


// Type definition for base and derived classes
typedef std::vector<bool> BOOL_VEC;
/*-------------------------------------------------------------------
BoolExample class:
A BoolExample object represents a training example of boolean values
indicating whether each of a set of features is true of false. For
these inputs, the example then has a correponding boolean output.
This acts as a base class with which decision tree learning can be
implemented.
--------------------------------------------------------------------*/
class BoolExample
{
	
private:

	// Member objects
	BOOL_VEC ftrs;    // vector of features
	bool output;      // output for this example, current accessed value


	
public:

	// Constructors
	BoolExample(BOOL_VEC f, bool out)   // initialisation
	: ftrs(f), output(out) {}           
	BoolExample() {}   	  	            // default constructor
	
	
	// Access methods
	bool getOut() {return output;}   			
	BOOL_VEC getFtrs() {return ftrs;}
	virtual int nrFtrs() {return ftrs.size();}  
	void set(int i, bool const& val) {ftrs[i] = val;}
	
	
	// Display example
	virtual void display()
	{
		BOOL_VEC::iterator it;
		for (it = ftrs.begin(); it < ftrs.end(); ++it)
			std::cout << *it << " ";
		std::cout << "| " << output << std::endl;
	}
	
	
	// Read-only subscript operator
	bool operator[](int i) const   // read-only subscript
	{
		if (i < 0 || i >= (int)ftrs.size())
			throw std::out_of_range("Error: Attempt to access non-existent feature!\n");
		return ftrs[i];
	}
};

















// Type definitions for subsequent classes
typedef std::vector<BoolExample> BOOL_DATA;
typedef std::map<std::string, double> PROB;
typedef std::pair<BOOL_VEC, bool> OUT_PAIR;

/*-------------------------------------------------------------------
BoolTable class:
A BoolExample object represents a set of training data comprising 
BoolExample objects. The BoolTable class is templated, allowing for 
objects to be constructed directly from a set of BoolExample objects,
or by first accepting numerical data and then converting it into 
a set of BoolExample objects where each feature represents a possible
splitting point. 
--------------------------------------------------------------------*/
template<typename T>
class BoolTable : public BoolExample
{

private:

	// Member objects
	BOOL_DATA data;          // vector of examples
	std::vector<T> ftrPnts;  // vector of possible numerical splits

	
	// Self-information
	double selfInfo(double p){return -log2(p);}   
	
	
	// Shannon entropy
	double entropy(double p)  		          
	{
		if (p < 0 || p > 1)
			throw std::invalid_argument("Error: Invalid probability!\n");
		if (p == 0 || p == 1)
			return 0;
		return p*selfInfo(p) + (1-p)*selfInfo(1-p);
	}
	
	
	// Map of relevant probabilities
	PROB prob(int ftrNr)
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
	
	
	// Average entropy
	double avEntropy(int ftrNr)	    	  
	{
		if (ftrNr < 0 || ftrNr >= nrFtrs())
			throw std::out_of_range("Error: Attempt to access non-existent feature!\n");
		PROB ps = prob(ftrNr);
		return ps["all"] * entropy(ps["true"]) + (1-ps["all"]) * entropy(ps["false"]);
	}
	
	
	// Convert numerical data into a vector of BoolExample objects
	BOOL_DATA numToBool(std::vector<T> numDt, BOOL_VEC outs)
	{
		int nr = numDt.size();
		ftrPnts.resize(nr-1);
		BOOL_DATA dt(nr);                               
		BOOL_VEC ex(nr-1);                             
		int i, j;
		std::vector<T> temp = numDt; 
		std::sort(temp.begin(), temp.end());

		for (i = 0; i < nr-1; ++i)
			ftrPnts[i] = (temp[i] + temp[i+1]) / 2; 

		for (i = 0; i < nr; ++i)
		{
			for (j = 0; j < nr-1; ++j)
				ex[j] = numDt[i] < ftrPnts[j] ? true : false;
			dt[i] = BoolExample(ex, outs[i]);
		}
		return dt;
	}
	
	
	
public:

	// Default constructor
	BoolTable() {}    	
	
	
	// Constructor with initialisation from a vector of BoolExample objects 
	BoolTable(BOOL_DATA dt)        
	: BoolExample(dt[0]), data(dt) 
	{
		for (int i = 1; i < (int)dt.size(); ++i)
			if (dt[i].nrFtrs() != dt[0].nrFtrs())
				throw std::invalid_argument("Error: Examples have different numbers of features!\n");
	}
	
	
	// Constructor with initialisation from a numerical inputs and boolean outputs
	BoolTable(std::vector<T> dt, BOOL_VEC outs) 
		{data = numToBool(dt, outs);}
		

	// Number of examples in dataset
	int nrExmpls() {return data.size();}    
	
	
	// Number of features in dataset
	int nrFtrs()  						
	{	
		if (data.size() == 0)
			return 0;
		return data[0].nrFtrs();
	}
	
	
	// Add example to dataset
	void addExmpl(BoolExample exmpl)       
	{
		if (BoolTable::nrFtrs() > 0 && exmpl.nrFtrs() != BoolTable::nrFtrs())
			throw std::length_error("Error: New example has incorrect number of features!\n");
		data.push_back(exmpl);
	}
	
	
	// Display table of data
	void display()
	{
		BOOL_DATA::iterator it;
		for (it = data.begin(); it < data.end(); ++it)
			(*it).BoolExample::display();
		std::cout << std::endl;
	}
	
	
	// Read-only subscript operator
	BoolExample operator[](int i) const   
	{
		if (i < 0 || i > (int)data.size())
			throw std::out_of_range("Error: Attempt to access non-existent example!\n");
		return data[i];
	}	
	
	
	// Find best feature on which to split data
	int splitPnt()          	                 
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
	
	
	// Construct tables resulting from data split
	std::pair<BoolTable, BoolTable> split(int pnt) 
	{
		BoolTable trueTree, falseTree;
		for (int ex = 0; ex < nrExmpls(); ++ex)
			if ((*this)[ex][pnt])
				trueTree.addExmpl((*this)[ex]); 
			else
				falseTree.addExmpl((*this)[ex]); 
		return std::make_pair(trueTree, falseTree);
	}
	
	
	// Returns pair, where first object is vector of boolean outputs of all examples,
	// and second object is boolean indicating whether all output values are the same
	OUT_PAIR outputCheck()                      
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
	
	
	// Returns true if all examples have matching inputs
	bool uniformInputs()        			  		 
	{
		if (nrExmpls() > 1)
			for (int i = 1; i < nrExmpls(); ++i)
				if ((*this)[i].getFtrs() != (*this)[0].getFtrs())
					return false;
		return true;
	}
	
	
	// Output value held by majority of matching examples
	bool stopVal()            			  		  
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
	
	
	// Convert numerical data point into boolean vector with respect to possible data splits
	BOOL_VEC boolConverter(T val)
	{
		BOOL_VEC v;
		for (int i = 0; i < (int)ftrPnts.size(); ++i)
			if (val < ftrPnts[i])
				v.push_back(true);
			else
				v.push_back(false);
		return v;
	}

};


#endif   // _BOOL_

