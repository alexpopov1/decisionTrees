# decision-trees
An easy-to-use header-only library in C++ to generate classification and regression trees (CART) for datasets. The software provides flexibility when designing the predictor, with options to set and tune various hyperparameters, as well as to apply bagging (to reduce variance) and random feature selection (to prevent correlation between different sampled trees in bagging).

# Table of contents
- [Installation](#install) 
- [Classification Problems](#class) 
    * [Constructing a Classification Tree](#classt)
    * [Bagged Classification Trees](#bagclass) 
    * [Predictions](#bpred) 
    * [Classification Error](#cerror) 
- [Regression Problems](#reg)
    * [Constructing a Regression Tree](#regt)
    * [Bagged Regression Trees](#breg)
    * [Predictions](#rpred)
    * [Mean Squared Error](#mse)
- [Creating a Test Set](#test)
- [Examples](#ex)
    
## Installation <a name="install"></a>
Simply download the four header files in the [src folder](https://github.com/alexpopov1/decision-trees/tree/main/src). Ensure that all four files are saved in the same location on your computer. To use the library, you just need to write `#include "decisionTrees.h"` at the start of your program (make sure it's in the same directory), and then you're ready to go!<br/><br/>


## Classification Problems <a name="class"></a>


### Constructing a Classification Tree <a name="classt"></a>
To initialise a classification tree called 'classTree', define the object `ClassificationTree<T, U> classTree(in, out)`, where `in` is a collection of inputs of type `std::vector< std::vector<T> >` and `out` is the collection of corresponding outputs of type `std::vector<U>`. 

Before the tree is built, you can set a couple of its properties:
* `classTree.setMaxDepth(d)` will limit the tree depth to *d*.
* `classTree.setImpurity('g')` will change the impurity measure from Shannon entropy (default) to Gini impurity. To change it back to entropy, simply use the character input `'e'` instead.

The tree can then be constructed using `classTree.buildTree()`.


### Bagged Classification Trees <a name="bagclass"></a>
To initialise a set 'baggedClassTrees' of *n* classification trees based on *n* samples from the dataset, define `BaggedClassificationTrees<T, U> baggedClassTrees(in, out, n)`, where all parameters are defined as before. 

The maximum tree depth and impurity measure can be set just as for a single classification tree, and additionally random feature selection can be applied at each node of each tree using `baggedClassTrees.setNrSelectedFeatures(f)`, where *f* is the number of features to be considered at each node (must be no greater than the total number of features of the dataset). 

The set of trees can then be constructed with `baggedClassTrees.buildTrees()`. 

The out-of-bag classification error can be calculated with `baggedClassTrees.outOfBagError()`.

### Predictions <a name="bpred"></a>
To predict the class for some input variable `input`, use `classTree.predict(input)` and `baggedClassTrees.predict(input)` respectively for the two cases.

### Classification Error <a name="cerror"></a>
When testing the classifier with some test set of inputs `testInputs` and associated classes `testOutputs`, the mean classification error (misclassification rate) for the entire test set can be found as follows:
* `classificationError< ClassificationTree<T, U> >(classTree, testInputs, testOutputs)` for the original classification tree.
* `classificationError< BaggedClassificationTrees<T, U> >(baggedClassTrees, testInputs, testOutputs)` for the bagged trees.<br/><br/>


## Regression Problems <a name="reg"></a>

### Constructing a Regression Tree <a name="regt"></a>
To initialise a regression tree called 'regTree', define the object `RegressionTree<T, U> regTree(in, out)`, where `in` is a collection of inputs of type `std::vector< std::vector<T> >` and `out` is the collection of corresponding outputs of type `std::vector<U>`. All subsequent methods work in exactly the same way as for classification trees (except for the impurity, which is fixed as mean squared error for all regression problems).


### Bagged Regression Problems <a name="breg"></a>
To initialise a set 'baggedRegTrees' of *n* regression trees based on *n* samples from the dataset, define `BaggedRegressionTrees<T, U> baggedRegTrees(in, out, n)`, where all parameters are defined as before. 

Random feature selection can be incorporated just as for bagged classification trees, using `baggedRegTrees.setNrSelectedFeatures(f)` to randomly select *f* features to consider at each node.

The out-of-bag mean squared error can be calculated with `baggedRegTrees.outOfBagError()`.

### Predictions <a name="rpred"></a>
To predict the output value for some input variable `input`, use `regTree.predict(input)` and `baggedRegTrees.predict(input)` respectively for the two cases.

### Mean Squared Error <a name="mse"></a>
When testing the model with some test set of inputs `testInputs` and associated output values `testOutputs`, the mean squared error averaged across the entire test set can be found as follows:
* `meanSquareError< RegressionTree<T, U> >(regTree, testInputs, testOutputs)` for the original classification tree.
* `meanSquareError< BaggedRegressionTrees<T, U> >(baggedRegTrees, testInputs, testOutputs)` for the bagged trees.<br/><br/>


## Creating a Test Set <a name="test"></a>
The library also lets you prepare a test set by holding out a certain percentage *p* of the original data. Starting with `in` and `out` defined as above, define:

`auto pr = splitDataset<T, U>(in, out, p);`

The first element of `pr` is the input-output pair of training examples, and the second element is the corresponding pair for the newly extracted test set. Hence:

```
std::vector< std::vector<T> > trainInputs = pr.first.first;
std::vector<U> trainOutputs = pr.first.second;
std::vector< std::vector<T> > testInputs = pr.second.first;
std::vector<U> testOutputs = pr.second.second;
```

These newly defined objects can now be used to test the performance of your decision tree model.<br/><br/>


## Examples <a name=ex></a>
The [examples folder](https://github.com/alexpopov1/decision-trees/tree/main/examples) contains a [classification example (classification of dry beans)](https://github.com/alexpopov1/decision-trees/tree/main/examples/classification) and a [regression example (real estate prices)](https://github.com/alexpopov1/decision-trees/tree/main/examples/regression). For each example folder, the csv file contains the dataset used, and the cpp file contains the code.
