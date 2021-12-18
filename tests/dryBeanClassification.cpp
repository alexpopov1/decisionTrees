

#include "trees.h"
#include "bagging.h"
#include "testing.h"
#include <string>
#include <fstream>



int main()
{

	std::size_t N = 13611, D = 16;	
	std::ifstream inFile;
	std::vector< std::vector<double> > inputs(N);
	std::vector<std::string> outputs(N);
	
	// Open and read file - example taken from UCI machine learning repository
	std::string file = "C:/Users/apbab/OneDrive/Documents/ML Datasets/Dry_Bean_Dataset.csv";
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
	BaggedClassificationTrees<double, std::string> beanBaggedTrees(trainingInputs, trainingOutputs, 10);
	beanBaggedTrees.setMaxDepth(500);
	beanBaggedTrees.setImpurity('g');
	beanBaggedTrees.buildTrees();
	double baggingMisclassificationRate = classificationError< BaggedClassificationTrees<double, std::string> >(beanBaggedTrees, testingInputs, testingOutputs);
	double oobError = beanBaggedTrees.outOfBagError();
	
	// Construct set of 10 bagged classification trees with random feature selection and evaluate performance using test set
	BaggedClassificationTrees<double, std::string> beanRandomBaggedTrees(trainingInputs, trainingOutputs, 10);
	beanRandomBaggedTrees.setMaxDepth(500);
	beanRandomBaggedTrees.setImpurity('g');
	beanRandomBaggedTrees.setNrSelectedFeatures(4);
	beanRandomBaggedTrees.buildTrees();
	double baggingRandomMisclassificationRate = classificationError< BaggedClassificationTrees<double, std::string> >(beanRandomBaggedTrees, testingInputs, testingOutputs);
	double rOobError = beanRandomBaggedTrees.outOfBagError();
	
	
	
	// Display performance comparison
	std::cout << std::setw(30) << "Training" << std::setw(30) << "Misclassification Rate\n";
	std::cout << std::setw(30) << "Default" << std::setw(30) << misclassificationRate << '\n';
	std::cout << std::setw(30) << "Bagging (10 samples)" << std::setw(30) << baggingMisclassificationRate << '\n';
	std::cout << std::setw(30) << "RF (10 samples, 4 ftrs)" << std::setw(30) << baggingRandomMisclassificationRate << '\n';
	
	std::cout << "\n\n\n";
	std::cout << std::setw(30) << "Training" << std::setw(30) << "Out of bag error\n";
	std::cout << std::setw(30) << "Bagging (10 samples)" << std::setw(30) << oobError << '\n';
	std::cout << std::setw(30) << "RF (10 samples, 4 ftrs)" << std::setw(30) << rOobError << '\n';

}


