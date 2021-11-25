

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <map>
#include "tree.h"
#include "naiveBayes.h"

#define f false
#define t true





#define L "-----------------------------------------------------\n"

int main()
{
	
	// Create examples
	BoolExample a1({f, t, t, f}, f),
				a2({t, f, t, t}, t),
				a3({t, t, t, f}, t),
				a4({f, f, t, t}, t),
				a5({t, f, f, t}, f),
				a6({f, t, t, t}, f);
				
	BoolExample b1({f, t, t, f}, f),
				b2({t, f, t, f}, f),
				b3({t, t, t, f}, t),
				b4({f, f, f, t}, t),
				b5({t, f, f, t}, f),
				b6({f, t, f, t}, f);

	BoolExample c1({f, t, t, f}, f),
				c2({t, f, t, f}, f),
				c3({t, t, t, f}, t),
				c4({f, f, f, t}, t),
				c5({f, t, f, t}, t),
				c6({f, t, f, t}, f),
				c7({f, t, f, t}, f);
				
	// Create vectors of examples
	BOOL_DATA dataset1 = {a1, a2, a3, a4, a5, a6};      // Easy case
	BOOL_DATA dataset2 = {b1, b2, b3, b4, b5, b6};      // Exclusive or
	BOOL_DATA dataset3 = {c1, c2, c3, c4, c5, c6, c7};  // Ambiguous leaf
	
	// Vector of case studies to be tested below
	std::vector<BOOL_DATA> data{dataset1, dataset2, dataset3};
	
	
	
	// Test case studies one by one within try...catch block
	try
	{
		
		for (int i = 0; i < (int)data.size(); ++i)
		{
			// Banner 
			std::cout << "************ TEST " 
						<< i+1 << " *************\n\n";
						
			// Create training set
			BoolTable<BOOL_VEC> trainingSet = BoolTable<BOOL_VEC>(data[i]);
			
			// Display training set
			trainingSet.display();
			std::cout << "\nNumber of features: " << trainingSet.nrFtrs();
			std::cout << "\nNumber of examples: "
				<< trainingSet.nrExmpls() <<"\n\n";
			
	
				
				
			// Create decision tree from training set
			DecisionTree<BOOL_VEC> tree(trainingSet);
			std::cout << "----------------------------\n";
			
			// Classify new data using decision tree logic
			BOOL_VEC input({f, t, f, f});
			std::cout << "Input: "; print(input); std::cout << std::endl;
			bool output = tree.classify(input);
			std::cout << "\nOutput = " << output
				<< "\n\n\n\n\n\n\n";
			
		}
	}
	catch(std::invalid_argument& err)
	{
		std::cerr << err.what();
	}
	catch(std::out_of_range& err)
	{
		std::cerr << err.what();
	}
	
	std::cout << L << L << L << "\n\n\n";
	//------------------------------------------------------------------------------------------------
	
	


	

	
	std::vector<double> dataList{0.1, 0.2, 0.36, 0.45, 0.51, 0.8, 1.13, 1.29, 1.41};
	std::vector<bool> outputs{t, t, f, f, t, t, t, f, f};
	
	print(dataList);
	print(outputs);
	std::cout << "\n\n";
	DecisionTree<double> numberTree(dataList, outputs);
	std::cout << "----------------------------\n";
	
	double in = 0.65;
	std::cout << "Input: " << in << "\n\n";
	bool result = numberTree.classify(in);
	std::cout << "\nOutput: " << result << "\n\n\n\n";


	

	
	
	
	
	
	
	
	
	
	
	// Banner
	std::cout << "****************** NAIVE BAYES *****************\n\n";
	
	// Create training set
	typedef BoolExample B;
	BOOL_DATA dataset(10);
	dataset[0] = B({f, t, t, f}, t);
	dataset[1] = B({f, f, t, t}, t);
	dataset[2] = B({t, f, t, f}, t);
	dataset[3] = B({f, f, t, t}, t);
	dataset[4] = B({f, f, f, f}, t);
	dataset[5] = B({t, f, f, t}, f);
	dataset[6] = B({t, t, f, t}, f);
	dataset[7] = B({t, f, f, f}, f);
	dataset[8] = B({t, t, f, t}, f);
	dataset[9] = B({t, f, t, t}, f);
	
	BoolTable<BOOL_VEC> trainingSet(dataset);
	trainingSet.display();
	
	
	// Implement Naive Bayes
	NaiveBayes<BOOL_VEC> naive(trainingSet);
	R_VALS rVals = naive.getRVals();
	
	BOOL_VEC input{t, t, t, t};
	std::cout << "Input: "; print(input);
	bool output = naive.classify(input);
	std::cout << "Output = " << output << std::endl;
	
}

