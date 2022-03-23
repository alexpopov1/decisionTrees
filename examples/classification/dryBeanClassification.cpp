

#include "decisionTrees.h"
#include <string>
#include <fstream>



int main()
{
	// Prepare dataset objects
	std::size_t N = 13611, D = 16;	
	std::vector< std::vector<double> > inputs(N);
	std::vector<std::string> outputs(N);
	
	// Open file - example taken from UCI machine learning repository
	std::ifstream inFile;
	std::string file = "dry beans.csv";
	inFile.open(file);
	
	// Read file
	if (inFile.is_open())
	{
		std::string str;
		std::size_t pos, n = 0;
		
		// Skip first line
		std::getline(inFile, str);
		
		// Read file line by line
		while(std::getline(inFile, str))
		{			
			// Extract input data
			inputs[n].resize(D);
			for (std::size_t d = 0; d < D; ++d) 
			{
				pos = str.find(',');
				inputs[n][d] = std::stod(str.substr(0, pos));
				str.erase(0, pos+1);
			}

			// Extract output data
			outputs[n] = str;

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
	auto pr = splitDataset<double, std::string>(inputs, outputs, 10);
	std::vector< std::vector<double> > trainingInputs = pr.first.first; 
	std::vector<std::string> trainingOutputs = pr.first.second;
	std::vector< std::vector<double> > testingInputs = pr.second.first; 
	std::vector<std::string> testingOutputs = pr.second.second;
	
	// Construct classification tree and evaluate performance using test set
	ClassificationTree<double, std::string> beanTree(trainingInputs, trainingOutputs);
	beanTree.setMaxDepth(500);
	beanTree.setImpurity('g');
	beanTree.buildTree();
	double misclassificationRate = classificationError< ClassificationTree<double, std::string> >(beanTree, testingInputs, testingOutputs);

	// Construct set of 10 bagged classification trees and evaluate performance using test set
	std::cout << "*** BAGGING ***\n";
	BaggedClassificationTrees<double, std::string> beanBaggedTrees(trainingInputs, trainingOutputs, 10);
	beanBaggedTrees.setMaxDepth(500);
	beanBaggedTrees.setImpurity('g');
	beanBaggedTrees.buildTrees();
	double baggingMisclassificationRate = classificationError< BaggedClassificationTrees<double, std::string> >(beanBaggedTrees, testingInputs, testingOutputs);
	double oobError = beanBaggedTrees.outOfBagError();
	
	// Construct set of 10 bagged classification trees with random feature selection and evaluate performance using test set
	std::cout << "\n\n*** BAGGING WITH RANDOM FEATURE SELECTION ***\n";
	BaggedClassificationTrees<double, std::string> beanRandomBaggedTrees(trainingInputs, trainingOutputs, 10);
	beanRandomBaggedTrees.setMaxDepth(500);
	beanRandomBaggedTrees.setImpurity('g');
	beanRandomBaggedTrees.setNrSelectedFeatures(4);
	beanRandomBaggedTrees.buildTrees();
	double baggingRandomMisclassificationRate = classificationError< BaggedClassificationTrees<double, std::string> >(beanRandomBaggedTrees, testingInputs, testingOutputs);
	double rOobError = beanRandomBaggedTrees.outOfBagError();

	// Display key for abbreviations
	std::cout << "\n\n\nKEY: In the following tables, 'Bagging (10)' means 10 bootstrap samples used,\n" 
	<< "and 'RF (10, 2)' means 10 samples used, and 2 features randomly selected at each node.";
	
	// Display performance comparison using misclassification rate and out of bag error
	std::cout << "\n\n\n";
	std::cout << "Comparing performance of training methods using misclassification rate (MR) and out-of-bag error (OOB):\n";
	std::cout << std::setw(13) << "Training" << std::setw(13) << "MR" << std::setw(10) << "OOB\n";
	std::cout << std::setw(13) << "Default" << std::setw(13) << misclassificationRate << std::setw(13) << " - \n";
	std::cout << std::setw(13) << "Bagging (10)" << std::setw(13) << baggingMisclassificationRate << std::setw(13) << oobError << '\n';
	std::cout << std::setw(13) << "RF (10, 4)" << std::setw(13) << baggingRandomMisclassificationRate << std::setw(13) << rOobError << '\n';
	
	// Make single predictions using predict(...) function
	std::vector<double> in = testingInputs[0];
	std::string predDefault = beanTree.predict(in);
	std::string predBagging = beanBaggedTrees.predict(in);
	std::string predRF = beanRandomBaggedTrees.predict(in);
	
	// Display predictions for first test point using each predictor
	std::cout << "\n\n\n";
	std::cout << "Predictions for first test point (actual output = " << testingOutputs[0] << "):\n";
	std::cout << std::setw(12) << "Training" << std::setw(15) << "Prediction\n";
	std::cout << std::setw(12) << "Default" << std::setw(14) << predDefault << '\n';
	std::cout << std::setw(12) << "Bagging (10)" << std::setw(14) << predBagging << '\n';
	std::cout << std::setw(12) << "RF (10, 4)" << std::setw(14) << predRF << '\n';	
}


