To run:
python3 main.py

This program uses decision trees to predict data points' class labels given their feature values. Four different sets of synthetic data are included for testing, but the main purpose is to predict video games' Metacritic scores based on sales, developer, genre, etc.

This program will print the tree's accuracy (0 is no accuracy, 1 is completely accurate), and, in the case of the synthetic data, it will generate a graph to demonstrate the effectiveness of the model. In the graphs, the points are the given data, and the colored regions are what the model predicts future data would be.