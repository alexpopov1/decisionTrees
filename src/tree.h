
#ifndef _TREE_
#define _TREE_

#include "boolData.h"


/*-------------------------------------------------------------------
TreeNode class:
A TreeNode object implements the logic of a node in a decision tree.
The class is derived from BoolTable, and so can initially be used to 
create the first split on a training set. The member methods can 
similarly be applied to any subsequent node in the tree, both by
splitting a set of data and by checking if a sufficient stopping
criterion has been reached, in which case a node can be labelled as 
a leaf.
--------------------------------------------------------------------*/

template<typename T>
class TreeNode : public BoolTable<T>
{
	
private:

	// Type definition
	typedef std::pair< BoolTable<T>, BoolTable<T> > BRANCHES;
	
	
	// Member objects     
	BoolTable<T> trueBranch, falseBranch;  // data sets resulting from split
	bool leaf, leafVal;		               // leaf indicates if node is leaf, leafVal gives its value					
	int splitFtr = -1;                     // feature on which node data is split
	
	
	// Build node structure by determining dataset split
	void buildNode() 
	{
		OUT_PAIR opr = this->outputCheck();
		leaf = opr.second;
		
		if (leaf)
			leafVal = opr.first[0];
		else if (this->uniformInputs() && !leaf)
		{
			leaf = true;
			leafVal = this->stopVal();
		}
		else
		{	
			splitFtr = this->splitPnt();
			BRANCHES pr = this->split(splitFtr);
			trueBranch = pr.first;
			falseBranch = pr.second;	
		}	
	} 



public:

	// Constructor with initialisation from BoolTable object
	TreeNode(BoolTable<T> dt) 
	: BoolTable<T>(dt) {buildNode();}   


	// Constructor with initialisation from numerical training data
	TreeNode(std::vector<T> dt, BOOL_VEC outs) 
	: BoolTable<T>(dt, outs) {buildNode();}


	// Access methods
	BoolTable<T> getTBranch() {return trueBranch;}
	BoolTable<T> getFBranch() {return falseBranch;}
	bool isLeaf() {return leaf;}
	bool value() {return leafVal;}
	int getSplit() {return splitFtr;}
};








/*-------------------------------------------------------------------
DecisionTree class
An object of this class is constructed with a set of training data. 
The member methods then implement the logic of a decision tree, using
inherited methods from the TreeNode class to split data and identify 
leaves of the tree.
The nature of a node is determined by a map, which maps a path to an 
integer value describing the current node. The path is described in 
pairs of integers representing the previous nodes leading up to the 
current node. The first integer in the pair is the feature on which
that node was split, and the second is the branch (true/false) followed
from this split. The value for the current node indicates the next split
point, unless the current node is a leaf, in which case the value 
indicates the output for that leaf (true -> -1, false -> -2). 
--------------------------------------------------------------------*/

template<typename T>
class DecisionTree : public TreeNode<T>
{
	
private:

	// Type definition
	typedef std::map< std::vector<int>, std::pair< int, BoolTable<T> > > NODES;
	
	
	// Member objects
	NODES nodeSet;           // set of integer codes and examples identifying each node of tree
	bool output;             // output value for input data, as found from decision tree
	
	
	// Build tree with recursive splitting and identification of leaves
	void buildTree(BoolTable<T> dt, std::vector<int> path)
	{
		print(path);
		TreeNode<T> node(dt);
		
		if (node.isLeaf())
			nodeSet[path] = std::make_pair(node.value()-2, dt);
		else
		{
			nodeSet[path] = std::make_pair(node.getSplit(), dt);
			path.push_back(node.getSplit());
			path.push_back(1); 
			buildTree(node.getTBranch(), path);
			path.back() = 0;
			buildTree(node.getFBranch(), path);
		}
	}
	
	
	// Find identifying value for a given path
	int node(std::vector<int> path) {return nodeSet[path].first;}
	
	
	// Implement decision tree logic
	void decisions(BOOL_VEC input, std::vector<int> path)
	{

		if ((int)input.size() != this->nrFtrs())
			throw std::invalid_argument("Error: Input data must have same number of features as training examples!\n");
			
		int val = node(path);
		print(path); 
		if (val < 0)
		{
			output = val == -1 ? true : false;
		}
		else
		{
			path.push_back(val);
			if (input[val])
			{
				path.push_back(1);
				decisions(input, path);
			}
			else
			{
				path.push_back(0);
				decisions(input, path);
			}
		}
	}
	
	
	


	
public:
		
	// Constructor with BoolTable argument
	DecisionTree(BoolTable<T> dt) : TreeNode<T>(dt) 
			{std::cout << "DecisionTree BoolTable constructor\n";
				buildTree(*this, std::vector<int>{-1, 0});}				
			
			
	// Constructor for general (numerical) training data		
	DecisionTree(std::vector<T> dt, BOOL_VEC outs)
	: TreeNode<T>(dt, outs)
		{buildTree(*this, std::vector<int>{-1, 0});}
	
	
	// Classify new input based on decision tree logic	
	classify(T in)
	{
		BOOL_VEC input;
		if constexpr (std::is_same<T, BOOL_VEC>::value)
			input = in;
		else
			input = this->boolConverter(in);
			
		decisions(input, std::vector<int>({-1, 0}));
		return output;
	}
	
};


#endif   // _TREE_