


#include "trees.h"
#include "bagging.h"
#include "testing.h"
#include <string>
#include <fstream>



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
		std::cout << "No file has been opened\n";
	
	
	auto pr = splitDataset<double, double>(inputs, outputs, 10);
	std::vector< std::vector<double> > trainingInputs = pr.first.first; 
	std::vector<double> trainingOutputs = pr.first.second;
	std::vector< std::vector<double> > testingInputs = pr.second.first; 
	std::vector<double> testingOutputs = pr.second.second;
	





 
	RegressionTree<double, double> realEstateTree(trainingInputs, trainingOutputs);
	realEstateTree.setMaxDepth(200);
	realEstateTree.buildTree();

	double MSE = meanSquareError< RegressionTree<double, double> >(realEstateTree, testingInputs, testingOutputs);
	
	
	
	
	
    // Construct set of 10 bagged regression trees and evaluate performance using test set
	BaggedRegressionTrees<double, double> realEstateBaggedTrees(trainingInputs, trainingOutputs, 10);
	realEstateBaggedTrees.setMaxDepth(200);
	realEstateBaggedTrees.buildTrees();
	
	double baggingMSE = meanSquareError< BaggedRegressionTrees<double, double> >(realEstateBaggedTrees, testingInputs, testingOutputs);
	double oobError = realEstateBaggedTrees.outOfBagError();
	
	
    // Construct set of 10 bagged regression trees with random feature selection and evaluate performance using test set
	BaggedRegressionTrees<double, double> realEstateRandomBaggedTrees(trainingInputs, trainingOutputs, 10);
	realEstateRandomBaggedTrees.setMaxDepth(200);
	realEstateRandomBaggedTrees.setNrSelectedFeatures(2);
	realEstateRandomBaggedTrees.buildTrees();
	
	double randomBaggingMSE = meanSquareError< BaggedRegressionTrees<double, double> >(realEstateRandomBaggedTrees, testingInputs, testingOutputs);
	double rOobError = realEstateRandomBaggedTrees.outOfBagError();
	
	
	
	
	// Display performance comparison
	std::cout << std::setw(12) << "Training" << std::setw(10) << "Error\n";
	std::cout << std::setw(12) << "Default" << std::setw(10) << MSE << '\n';
	std::cout << std::setw(12) << "Bagging (10)" << std::setw(10) << baggingMSE << '\n';
	std::cout << std::setw(12) << "RF (10, 2)" << std::setw(10) << randomBaggingMSE << '\n';
	
	std::cout << "\n\n\n";
	std::cout << std::setw(12) << "Training" << std::setw(15) << "OOB Error\n";
	std::cout << std::setw(12) << "Bagging (10)" << std::setw(15) << oobError << '\n';
	std::cout << std::setw(12) << "RF (10, 2)" << std::setw(15) << rOobError << '\n';

	
}



