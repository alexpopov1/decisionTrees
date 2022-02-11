


#include "decisionTrees.h"
#include <string>
#include <fstream>

/*

int main()
{
	std::size_t N = 414, D = 6;	
	std::ifstream inFile;
	std::vector< std::vector<double> > inputs(N);
	std::vector<double> outputs(N);
	std::string file = "C:/Users/apbab/OneDrive/Documents/ML Datasets/real estate prices.csv";
	inFile.open(file);
	

	if (inFile.is_open())
	{
		std::string str;
		std::size_t pos, n = 0;
		
		// Skip first line
		std::getline(inFile, str);
		
		// Read file line by line
		while(std::getline(inFile, str))
		{
			// Skip first column
			pos = str.find(',');
			str.erase(0, pos+1);
			
			
			// Extract input data
			inputs[n].resize(D);
			for (std::size_t d = 0; d < D; ++d) 
			{
				pos = str.find(',');
				inputs[n][d] = std::stod(str.substr(0, pos));
				str.erase(0, pos+1);
			}

			// Extract output data
			outputs[n] = std::stod(str);

			// Move down one line
			++n;
		}
	}
	else
	{
		std::cerr << "No file has been opened\n";
		exit(1);
	}
	
	// Hold out 10% of data for testing, use the remainder for training
	auto pr = splitDataset<double, double>(inputs, outputs, 10);
	std::vector< std::vector<double> > trainingInputs = pr.first.first; 
	std::vector<double> trainingOutputs = pr.first.second;
	std::vector< std::vector<double> > testingInputs = pr.second.first; 
	std::vector<double> testingOutputs = pr.second.second;

    // Construct regression tree and evaluate performance using test set
	RegressionTree<double, double> realEstateTree(trainingInputs, trainingOutputs);
	realEstateTree.setMaxDepth(200);
	realEstateTree.buildTree();
	double MSE = meanSquareError< RegressionTree<double, double> >(realEstateTree, testingInputs, testingOutputs);
	
    // Construct set of 10 bagged regression trees and evaluate performance using test set
	std::cout << "*** BAGGING ***\n";
	BaggedRegressionTrees<double, double> realEstateBaggedTrees(trainingInputs, trainingOutputs, 10);
	realEstateBaggedTrees.setMaxDepth(200);
	realEstateBaggedTrees.buildTrees();
	double baggingMSE = meanSquareError< BaggedRegressionTrees<double, double> >(realEstateBaggedTrees, testingInputs, testingOutputs);
	double oobError = realEstateBaggedTrees.outOfBagError();
	
    // Construct set of 10 bagged regression trees with random feature selection and evaluate performance using test set
	std::cout << "\n\n*** BAGGING WITH RANDOM FEATURE SELECTION ***\n";
	BaggedRegressionTrees<double, double> realEstateRandomBaggedTrees(trainingInputs, trainingOutputs, 10);
	realEstateRandomBaggedTrees.setMaxDepth(200);
	realEstateRandomBaggedTrees.setNrSelectedFeatures(2);
	realEstateRandomBaggedTrees.buildTrees();
	double randomBaggingMSE = meanSquareError< BaggedRegressionTrees<double, double> >(realEstateRandomBaggedTrees, testingInputs, testingOutputs);
	double rOobError = realEstateRandomBaggedTrees.outOfBagError();
	
	// Display key for abbreviations
	std::cout << "\n\n\nKEY: In the following tables, 'Bagging (10)' means 10 bootstrap samples used,\n" 
	<< "and 'RF (10, 2)' means 10 samples used, and 2 features randomly selected at each node.";
	
	// Display performance comparison using mean square error and out of bag error
	std::cout << "\n\n\n";
	std::cout << "Comparing performance of training methods using mean square error (MSE) and out-of-bag error (OOB):\n";
	std::cout << std::setw(12) << "Training" << std::setw(10) << "MSE" << std::setw(10) << "OOB\n";
	std::cout << std::setw(12) << "Default" << std::setw(10) << MSE << std::setw(10) << " - \n";
	std::cout << std::setw(12) << "Bagging (10)" << std::setw(10) << baggingMSE << std::setw(10) << oobError << '\n';
	std::cout << std::setw(12) << "RF (10, 2)" << std::setw(10) << randomBaggingMSE << std::setw(10) << rOobError << '\n';
	
	// Make single predictions using predict(...) function
	std::vector<double> in = testingInputs[0];
	double predDefault = realEstateTree.predict(in);
	double predBagging = realEstateBaggedTrees.predict(in);
	double predRF = realEstateRandomBaggedTrees.predict(in);
	
	// Display predictions for first test point using each predictor
	std::cout << "\n\n\n";
	std::cout << "Predictions for first test point (actual output = " << testingOutputs[0] << "):\n";
	std::cout << std::setw(12) << "Training" << std::setw(15) << "Prediction\n";
	std::cout << std::setw(12) << "Default" << std::setw(14) << predDefault << '\n';
	std::cout << std::setw(12) << "Bagging (10)" << std::setw(14) << predBagging << '\n';
	std::cout << std::setw(12) << "RF (10, 2)" << std::setw(14) << predRF << '\n';
	
	
}

	*/


