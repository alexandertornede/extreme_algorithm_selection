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

## Experiment and Execution Details
For the sake of reproducibility, we will detail how to reproduce the results presented in the paper below. 

**Coming in the following days (status 20/11/19)**
