## CS 7641 Assignment 2: Randomized Optimization
This project seeks to understand the behavoiral and computatitonal and predictive qualities of four random search optimzatio methods:
- Randomized Hill CLimb (RHC)
- Simulated Annealing (SA)
- Genetic ALgorithms (GA)
- Mutual Information Maximizing Input Clustering (MIMIC)

##Prerequisites
These instructions apply for Windows 10 x64.
For testing on your own machine, you need only to install the following libraries.
- ABAGAIL: https://github.com/pushkar/ABAGAIL
- Apache Ant: https://ant.apache.org/bindownload.cgi
- Java Development Kit: https://www.oracle.com/technetwork/java/javase/downloads/jdk10-downloads-4416644.html
- Add Java and Ant to your windows environment and path variables. A helpful guide is found at: https://www.mkyong.com/ant/how-to-install-apache-ant-on-windows/

Once all of the prerequisites are installed, all of the methods are run from the Windows Command Prompt

##Getting Started
1. Download the PhishingWebsitesData_preprocessed.csv
	a. Original Phishing Websites Data - available at https://www.openml.org/d/4534
2. Edit the following .java files to point them towards your downloaded PhishingWebsitesData_preprocessed.csv file location
	a. You can also use this time to edit the .java files to change the neurnal network structure
- phishing_rhc.java
- phishing_sa_val.java
- phishing_ga_val.java
- phishingwebsite_finaltest.java
3. Convert all .java files to .class files with the following code from the command prompt
> javac phishing_rhc.java
> javac phishing_sa_val.java
> javac phishing_ga_val.java
> javac phishingwebsite_finaltest.java
4. Move all .class files to the location ~\ABAGAIL\opt\test
	a. Includes the 4 'phishing_' class files and the 3 '_Toy' class files


## Part 1: Training a Neural Network using Random Search (RHC, SA, GA)
This section will train a neural network on the phishing websites dataset using RHC, SA, and GA. These methods are compared to each other and to the same network structure trained using backpropagation.

Running the Models (via command prompt):
> cd ~\ABAGAIL
> ant
> java -cp ABAGAIL.jar opt.test.phishing_rhc
> java -cp ABAGAIL.jar opt.test.phishing_sa_val
> java -cp ABAGAIL.jar opt.test.phishing_ga_val
> java -cp ABAGAIL.jar opt.test.phishingwebsite_finaltest

The model results (training times and neural network accuracies) are stored in .csv files located at ~\ABAGAIL\Optimization_Results


## Part 2: Random Search Toy Problems
This section presents 3 toy optimization problems for which RHC, SA, GA, and MIMIC are all used to maximize the function fitness.

#1. Traveling Salesman Problem - Highlights GA
> java -cp ABAGAIL.jar opt.test.TravelingSalesman_Toy
#2. Continuous Peaks Problem - Highlights SA
> java -cp ABAGAIL.jar opt.test.ContinuousPeaks_Toy
#3. Four Peaks Problem - Highlights MIMIC
> java -cp ABAGAIL.jar opt.test.TravelingSalesman_Toy

The model results (training times and fitness function values) are stored in .csv files located at ~\ABAGAIL\Optimization_Results
