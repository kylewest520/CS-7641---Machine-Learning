## CS 7641 Assignment 1: Supervised Learning Classification
This project seeks to understand the computatitonal and predictive qualities of five classification algorithms (Neural Network, SVM, kNN, Decision Tree, and Boosted Trees).
Each algorithm will be run for two binary classification datasets so that we can compare and contrast them for two different problems (one for a balanced target variable and the other for an unbalanced target variable).

Dataset 1: Phishing Websites - available at https://www.openml.org/d/4534

Dataset 2: Bank Marketing - available at https://www.openml.org/d/1461


## Getting Started & Prerequisites
For testing on your own machine, you need only to install python 3.6 and the following packages:
- pandas, numpy, scikit-learn, matplotlib, itertools, timeit


## Running the Classifiers
Optimal Way: Work with the iPython notebook (.ipnyb) using Jupyter or a similar environment. This allows you to "Run All" or you can run only the classifiers that you are interested in.

Second Best Option: Run the python script (.py) after first editing the location where you have the two datasets saved on your local machine.

Final Option (view only): Feel free to open up the (.html) file to see a sample output of all of the algorithms for both datasets.

The code is broken up into three main sections:
1. Data Load & Preprocessing -> Exactly as it sounds. This section loads the data, performs one-hot encoding, scales numeric features, and reorders some of the columns.
2. Helper Functions -> This section defines a few functions that are used across all of the classifiers. The functions include building learning curves and evaluating the final classifers.
3. The Fun Part: Machine Learning! -> This section has funcions and execution cells for each of the 5 classifiers.

