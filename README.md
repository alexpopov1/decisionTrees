# decision-trees
An easy-to-use header-only library in C++ to generate classification and regression trees (CART) for datasets. The software provides flexibility when designing the predictor, with options to set and tune various hyperparameters, as well as to apply bagging (to reduce variance) and random feature selection (to prevent correlation between different sampled trees in bagging).

## Installation
Simply download the four header files in the 'src' folder. Ensure that all four files are saved in the same location on your computer.

## Classification Problems
To initialise a classification tree called 'classTree', define the object `ClassificationTree<T, U> classTree(in, out)`, where `in` is a collection of inputs of type `std::vector< std::vector<T> >` and `out` is the collection of corresponding outputs of type `std::vector<U>`. 
Before the tree is built, you can set a couple of its properties:
* `classTree.setMaxDepth(d)` will limit the tree depth to *d*.
* `classTree.setImpurity('g')` will change the impurity measure from Shannon entropy (default) to Gini impurity. To change it back to entropy, simply use the character input `'e'` instead.
The tree can then be constructed using `classTree.buildTree()`.


### Creating Bagged Classification Trees
To initialise a set 'baggedClassTrees' of *n* classification trees based on *n* samples from the dataset, define `BaggedClassificationTree<T, U> baggedClassTrees(in, out, n)`, where all parameters are defined as before. 
The maximum tree depth and impurity measure can be set just as for a single classification tree, and additionally random feature selection can be applied at each node of each tree using `baggedClassTrees.setNrSelectedFeatures(f)`, where *f* is the number of features to be considered at each node (must be no greater than the total number of features of the dataset). The set of trees can then be constructed with `baggedClassTrees.buildTrees()`. 
The out-of-bag classification error can be calculated with `baggedClassTrees.outOfBagError()`.




