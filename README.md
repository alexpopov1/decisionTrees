# decision-trees
An easy-to-use header-only library in C++ to generate classification and regression trees (CART) for datasets. The software provides flexibility when designing the predictor, with options to set and tune various hyperparameters, as well as to apply bagging (to reduce variance) and random feature selection (to prevent correlation between different sampled trees in bagging).

## Installation
Simply download the four header files in the 'src' folder. Ensure that all four files are saved in the same location on your computer.

## Creating a Classification Tree
To initialise a classification tree called 'classTree', define the object `ClassificationTree<T, U> classTree(in, out)`, where `in` is a collection of inputs of type `std::vector< std::vector<T> >` and `out` is the collection of corresponding outputs of type `std::vector<U>`. 
Before the tree is built, you can set a couple of its properties:
* `classTree.setMaxDepth(100)` will limit the tree depth to 100.
* `classTree.setImpurity('g')` will change the impurity measure from Shannon entropy (default) to Gini impurity. To change it back to entropy, simply use the character input `'e'` instead.

