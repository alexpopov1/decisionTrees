
#include "tree.h"



void TreeNode::buildNode()
{
	OUT_PAIR opr = dataset.outputCheck();
	leaf = opr.second;
	
	if (leaf)
		leafVal = opr.first[0];
	else if (uniformInputs() && !leaf)
	{
		leaf = true;
		leafVal = stopVal();
	}
	else
	{	
		splitFtr = dataset.splitPnt();
		BRANCHES pr = dataset.split(splitFtr);
		trueBranch = pr.first;
		falseBranch = pr.second;	
	}	
}






void DecisionTree::buildTree(BoolTable dt, std::vector<int> path)
{
	print(path);
	TreeNode node(dt);
	
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




void DecisionTree::decisions(BOOL_VEC input, std::vector<int> path)
{

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