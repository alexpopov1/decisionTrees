


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


// Training example with feature values and a corresponding output
class BoolExample
{
private:
	// Member variables
	BOOL_VEC ftrs;  	// vector of features
	bool output;   		// output value for this example
	
public:
	// Constructors
	BoolExample(BOOL_VEC f, bool out) 
	: ftrs(f), output(out) {} 		   // initialisation
	BoolExample() {}    	  		   // default constructor

	// Access methods
	bool getOut() {return output;}   			
	BOOL_VEC getFtrs() {return ftrs;}
	virtual int nrFtrs() {return ftrs.size();}  // number of features for example
	
	// Display example
	virtual void display();

	// Operator overloading
	bool operator[](int i)   // read-only subscript
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

// Table of examples with feature values and outputs
class BoolTable : public BoolExample
{
private:
	// Member variables
	BOOL_DATA data;      // vector of examples
	
	// Entropy methods
	double selfInfo(double p){return -log2(p);}   // self-information
	double entropy(double p);      		      // Shannon entropy
	PROB prob(int ftrNr);           	      // map of relevant probabilities
	double avEntropy(int ftrNr); 		      // average entropy
	
public:
	// Constructors
	BoolTable() {}   		// default constructor
	BoolTable(BOOL_DATA dt)         // initialisation
	: BoolExample(dt[0]), data(dt) 
	{
		for (int i = 1; i < (int)dt.size(); ++i)
			if (dt[i].nrFtrs() != dt[0].nrFtrs())
				throw std::invalid_argument("Error: Examples have different numbers of features!\n");
	}
	

	// Access methods
	int nrExmpls() {return data.size();}    // number of examples in dataset
	int nrFtrs();    			// number of features in dataset
	void addExmpl(BoolExample exmpl);       // add example to dataset

	// Display table of data
	void display();
	
	// Operator overloading
	BoolExample operator[](int i) const   // read-only subscript
	{
		if (i < 0 || i > (int)data.size())
			throw std::out_of_range("Error: Attempt to access non-existent example!\n");
		return data[i];
		
	}	
	
	// Tree building methods
	int splitPnt();             	                  // best point to split tree
	std::pair<BoolTable, BoolTable> split(int pnt);   // construct two children of node
	OUT_PAIR outputCheck();		                  // returns example outputs at current node, as well as
				         		    // boolean indicating whether all output values are the same
	bool uniformInputs();         			  // returns true if all examples have matching inputs
	bool stopVal();               			  // output value held by majority of matching examples
	
};




#endif   // _BOOL_
