
#ifndef _TREE_
#define _TREE_

#include "boolData.h"



// Node of a decision tree, based on table of boolean examples
class TreeNode : public BoolTable
{
private:
	// Type definition
	typedef std::pair<BoolTable, BoolTable> BRANCHES;
	
	// Member variables
	BoolTable dataset, trueBranch, falseBranch; // current dataset and sets from split
	bool leaf, leafVal;		 // leaf indicates if node is leaf, leafVal gives its value					
	int splitFtr = -1;       // feature on which node data is split
	
	// Build node structure by determining dataset split
	void buildNode();  

public:
	// Constructor
	TreeNode(BoolTable dt) 
	: BoolTable(dt), dataset(dt) {buildNode();}   // initialise with dataset

	// Access methods
	BoolTable getData() {return dataset;}
	BoolTable getTBranch() {return trueBranch;}
	BoolTable getFBranch() {return falseBranch;}
	bool isLeaf() {return leaf;}
	bool value() {return leafVal;}
	int getSplit() {return splitFtr;}
};








/* Decision tree to classify new data based on boolean training data

    A path through the tree is described with a vector which is appended with 
	two new integers at each node. The first integer gives the parent split 
	(set -1 for the first node) and the second integer gives the boolean value of 
	the incoming branch (true -> 1, false -> 0, set 0 for the first node).
	
	The nature of a node is determined by a map, which maps a path to an integer 
	for the node reached by the path. If this node is a leaf then this value
	indicates the output for that leaf (true -> -1, false -> -2). Otherwise, the 
	value refers to the feature on which the remaining data is split.
*/

class DecisionTree : public TreeNode
{
private:
	// Type definition
	typedef std::map< std::vector<int>, std::pair< int, BoolTable > > NODES;
	
	// Member variables
	BoolTable trainingSet;   // all training data
	NODES nodeSet;           // set of integer codes and examples identifying each node of tree
	bool output;             // output value for input data, as found from decision tree
	
	// Build tree with recursive method
	void buildTree(BoolTable dt, std::vector<int> path);
	
	// Find identifying value for a given path
	int node(std::vector<int> path) {return nodeSet[path].first;}
	
	// Implement decision tree logic
	void decisions(BOOL_VEC input, std::vector<int> path);
	
public:
	// Constructor
	DecisionTree(BoolTable dt) : TreeNode(dt), trainingSet(dt) 
		{buildTree(dt, std::vector<int>{-1, 0});}
	
	// Classify new input based on decision tree logic
	bool classify(BOOL_VEC input)
	{
		if ((int)input.size() != nrFtrs())
			throw std::invalid_argument("Error: Input data must have same number of features as training examples!\n");
		decisions(input, std::vector<int>({-1, 0}));
		return output;
	}
	
};


#endif   // _TREE_