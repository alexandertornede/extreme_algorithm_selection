# Code for the ECAI 2020 submission: Extreme Algorithm Selection with Dyadic Feature Representation

This repository holds the code for the ECAI 2020 submission #1679 with the title "Extreme Algorithm Selection with Dyadic Feature Representation"
by Alexander Tornede, Marcel Wever and Eyke Hüllermeier. 

## Abstract
Algorithm selection (AS) deals with selecting an algorithm from a set of candidate algorithms most suitable for a 
specific instance of an algorithmic problem, e.g., choosing solvers for SAT problems. Standard benchmark suites 
for AS, such as ASlib, usually comprise candidate sets consisting of at most tens of algorithms. Motivated by the 
rise of automated machine learning, i.e., the automatic selection of machine learning algorithms for specific datasets, 
we consider AS problems with thousands of candidate algorithms in a setting we call _extreme algorithm selection (XAS)_, 
and investigate the ability of state-of-the-art AS approaches to scale with the size of such a set. 
This includes collaborative filtering as well as methods based on regression and ranking. 
Moreover, we propose the use of dyadic approaches, in which both problem instances as well as algorithms are represented 
in terms of feature information. In our evaluation study, we find that the latter are capable of dealing with the XAS 
setting and improve over the state of the art in various metrics.

## Benchmark Dataset
We created a benchmark dataset with over 1200 machine learning classification algorithms represented by feature information and evaluated them on almost 68 classification datasets from the [OpenML CC-18 benchmark](https://docs.openml.org/benchmark/#openml-cc18). This dataset contains both feature information for algorithms and datasets. Furthermore, the accuracy of the classification algorithms on the datasets is reported. 

### Datasets
We used all datasets contained in the OpenML CC-18 benchmark, except for the ones with the IDs 40923,40927,40996,554. The first two were excluded for technical reasons as they yielded errors within our code due to formatting errors. The latter two were excluded since none of the algorithms were sucessfully evaluated within a timelimit of 5 minutes.

Datasets are represented in terms of features by all 45 OpenML landmarkers, for which different configurations of the following learning algorithms are evaluated based on the error rate, area under the (ROC) curve, and Kappa coefficient: Naive Bayes, One-Nearest Neighbour, Decision Stump, Random Tree, REPTree and J48.

### Algorithms
As algorithms we used 18 classifiers from the Java machine learning library [WEKA](https://www.cs.waikato.ac.nz/ml/weka/): BayesNet (BN), DecisionStump (DS), DecisionTable (DT), IBk, J48, JRip (JR), KStar (KS), LMT, Logistic (L), MultilayerPerceptron (MP), NaiveBayes (NB), OneR (1R), PART, REPTree (REPT), RandomForest (RF), RandomTree (RT), SMO, ZeroR (0R).

An overview of these classifiers and their types of hyperparameters is given in the table below. The last row of the table sums up the items of the respective column, providing insights into the dimensionality of the space of potential candidate algorithms. From this space, we randomly sampled up to 100 distinct instantiations of each classifier, ensuring the instances being not too similar, yielding a set of 1270 candidate algorithms.

![](img/classifier_table.png)

Candidate algorithms are represented in terms of features by using their hyperparameters, or more precisely the values of their hyperparameters. Assume that an algorithm family is defined by the different instantiations of an algorithm featuring several hyperparameters. For example, we consider support vector machines (SVM) as an algorithm family and different configurations of an SVM as members of this family. Then, given a set of algorithm families, e.g. `{SVM, random forests (RF), logistic regression (LOR)}`, we compute the union over the set of parameters of the algorithm families and create a vector with one entry per numerical parameter and as many entries per categorical parameter as needed to allow for a one-hot-encoding. Furthermore, the feature representation has a binary feature per algorithm family, indicating whether a given candidate algorithm comes from that family or not. Then, when given any candidate algorithm from a known algorithm family, we create the associated feature representation by setting each element in the vector to the respective parameter value while initializing the irrelevant ones with 0.

As an example, consider again the set of algorithm families `{SVM, RF, LOR}` and assume for simplicity that each of these families has only a single numerical parameter. Then, given an SVM instantiation a where the associated parameter is set to `0.4` a feature representation according to our technique could be `(1,0.4,0,0,0,0)`. The first two elements of the vector correspond to the SVM family of which the first element, i.e. the 1, indicates that a is an SVM instantiation and the second element corresponds to its parameter value 0.4.

### Performance Values
We evaluted each of the candidate algorithms described above on all of the datasets given above using a 5-fold cross validation. The measure we used is classification accuracy. Each evaluation was constrained by a timeout of 5 minutes. Evaluations which did not finish in time are represented by a negative value in the dataset.


### Download and Structure
The benchmark dataset can be downloaded from this Github repository: (benchmark_dataset)[benchmark_dataset.zip]. Be aware that it expands to over 300 MB during decompression. The structure of the files is as follows. 

The zip file contains three CSV files:
* dataset_metafeature_new.csv: Each line corresponds to a dataset from the OpenML benchmark and its associated metafeature representation. The first element in the line is the OpenML ID of the dataset whereas the second element is the feature representation described above.
* algorithm_metafeatures.csv: Each line corresponds to one of the over 1200 candidate algorithms. The first element gives an id of the algorithm which is described by the line, whereas the second one gives a textual representation of the candidate algorithm in the JSON format. The third and last element gives the feature representation of the associated algorithm as described earlier.
* algorithm_evaluations_with_timeouts.csv: Each line corresponds to an evaluation of an algorithm on a dataset. The first element gives the seed used for the evaluation, the second element gives the ID of the associated dataset, the third element gives the textual description of the algorithm used for the evaluation and the fourth element gives the ID of the algorithm which can be referred to to find it in the algorithm metafeature table. The fifth value gives the actual accuracy or a negative value if the evaluation timed out. The last value gives the stack-trace of any exception which occurred during the evaluation, i.e. timeout related exceptions.

## Experiment and Execution Details
For the sake of reproducibility, we will detail how to reproduce the results presented in the paper below. 

**Coming in the following days (status 20/11/19)**
