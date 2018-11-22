## CS 7641 Assignment 4: Markov Decision Processes and Reinforcement Learning
This project seeks to understand the three reinforcement learning algorithms by applying them each to two different Markov decision processes (MDP). The reinforcement learning methods are value iteration, policy iteration, and Q-learning. The two MDP toy problems are inspired by Pacman! There is a small 5x5 grid world, and a large 20x20 grid world.

For each grid, Pacman (our learning agent) starts in the top left corner and attempts to navigate his way to the goal by collecting a high score along his journey. Like the real game, Pacman has the opportunity to earn points by eating pellets and fruit, but he must avoid hitting the ghost at all costs. The reward structure for each grid world is represented by:
- Small pellets (S) = +1 point
- Medium fruit (M) = +2.5 points
- Large ghosts (L) = -50 points
- Reaching the goal = +100 points
- Every step = -5 points to encourage Pacman to reach his goal quickly.

## Getting Started & Prerequisites
For testing on your own machine, the easiest way to implement is to follow the steps below:
1. Download Eclipse IDE: https://www.eclipse.org/
2. Download or clone my github file repository: https://github.com/kylewest520/CS-7641---Machine-Learning/tree/master/Assignment%204%20Markov%20Decision%20Processes
3. Import the project into Eclipse: http://help.eclipse.org/kepler/index.jsp?topic=%2Forg.eclipse.platform.doc.user%2Ftasks%2Ftasks-importproject.htm
4. Update the current Eclipse project using Maven by right-clicking the top level project folder > Maven > Update Project
5. Download the latest Java SE Development kit: https://www.oracle.com/technetwork/java/javase/downloads/jdk11-downloads-5066655.html
6. Ready to go! You can now run the main.java class in the project folder using Eclipse

## Modifying the Code (in Main.java)
The code is set up to run all three reinforcement learning algorithms for both problems sets by modifying a few lines.
1. Change from running the small grid vs. large grid by modifying the PROBLEM parameter in line 50.
2. Change from running value iteration, policy iteration, Q-learning using the algorithm parameter in line 56.
3. Change stochastic behavior in line 88.
4. Change value iteration and policy iteration algorithm conditions under the line 134 switch statement
5. Change the maximum number of iteration steps in line 218.

## Creating Your Own Grid Worlds
You can create your own grid worlds in Main.java line 282 (small grid) and line 320 (large grid). Note that this implementation requires both grid worlds to be square grids. The code allows for inputting a starting location (X), goal location (G), walls (1's), and small (S), medium (M), and large (L) rewards or penalties. To make S/M/L block a penalty, simply set the HazardType to be a negative number.
An example for a small 5x5 grid is shown below:

X0001  
00001  
0LML0  
0000S  
110SG

## Code Outputs
After running the code for a given grid/algorithm there will be a few outputs.
1. Pop-up window of final policy showing actions and state values.
2. Data dump to eclipse terminal of the results of each iteration including number of steps, reward, and wall clock time to complete the iteration. This output is structure for easy copy and paste into a text editor so that the results can be quickly saved in a .csv format and used for further analysis in other programs (python, excel, etc.)
3. Algorithm summary statistics across all iterations: average reward, average steps, minimum steps, and average wall clock time.

## Acknowledgements
The source code for this assignment was modified from the original versions found at:
1. http://burlap.cs.brown.edu/
2. https://github.com/svpino/cs7641-assignment4. I am indebted to this github repository. Please visit it to see the original implementation with extensive notes in the README.
