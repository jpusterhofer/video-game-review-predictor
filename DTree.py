import math
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from Node import Node
from Bin import Bin

class DTree:
    def __init__(self, train_file, num_bins):

        #Read in video game and synthetic data differently
        if train_file == "./Video_Games_Sales.csv":
            self.data = pd.read_csv(train_file)
            self.data = self.data.dropna()
            self.cbins = self.discretize(self.data["Critic_Score"], 10)
            
        else:
            self.data = pd.read_csv(train_file, names=["A", "B", "Label"])
            self.cbins = self.discretize(self.data["Label"], 2)

        self.num_bins = num_bins
        self.root = self.ID3(self.data)

    #Discretize the data into bins
    def discretize(self, data, num_bins):
        bins = []
        
        #Make each unique entry in nominal data its own bin
        if type(data.iloc[0]) == str:
            names = set(data)
            for name in names:
                new_bin = Bin()
                new_bin.name = name
                bins.append(new_bin)
            
        #Make ranges of values into bins
        else:
            lower = min(data)
            upper = max(data)
            incr = (upper - lower) / num_bins
            for i in range(num_bins):
                new_bin = Bin()
                new_bin.lower = round(lower + i*incr, 4)
                new_bin.upper = round(lower + (i+1)*incr, 4)
                bins.append(new_bin)

        return bins

    #Train using the ID3 algorithm
    def ID3(self, data, depth=3):
        
        #If only 1 class label or every class label is the same, set leaf to that
        if len(data.iloc[:,-1]) == 1 or self.all_same(data.iloc[:, -1]):
            leaf = Node()
            leaf.label = self.find_bin(data.iloc[0, -1], self.cbins)
            return leaf
        
        #If at maximum depth or no features left, set leaf to most common label
        if depth == 0 or len(data.columns) == 1:
            leaf = Node()
            leaf.label = self.fullest_bin(data.iloc[:, -1])
            return leaf

        #Make new node to partition data from
        curr_node = Node()
        curr_node.feature = self.best_feature(data)
        partitions, curr_node.bins = self.partition(data, curr_node.feature)

        #Add children to the node
        for partition in partitions:
            if not partition.empty:
                curr_node.children.append(self.ID3(partition, depth-1))
            else:
                leaf = Node()
                leaf.label = self.fullest_bin(data.iloc[:, -1])
                curr_node.children.append(leaf)
        return curr_node

    #Check if every class label in the given data is the same
    def all_same(self, data):
        cbin = self.find_bin(data.iloc[0], self.cbins)
        for label in data:
            if label not in cbin:
                return False
        return True

    #Find which bin has the most common class label
    def fullest_bin(self, data):
        label_count = [0] * len(self.cbins)
        for label in data:
            label_count[self.find_bin_index(label, self.cbins)] += 1
        return self.cbins[label_count.index(max(label_count))]

    #Find which feature gives the most gain when partitioned
    def best_feature(self, data):

        cols = list(data)
        #Calculate each feature's info gain
        all_entropy = self.entropy(data[cols[-1]])
        gain = []
        for feature in cols:
            if feature is not cols[-1]:
                partitions, bins = self.partition(data, feature)
                feature_entropy = 0
                for partition in partitions:
                    feature_entropy += len(partition[cols[-1]]) / len(data[cols[-1]]) * self.entropy(partition[cols[-1]])
                gain.append(all_entropy - feature_entropy)
        
        #Return the name of the column that had the most gain
        return cols[gain.index(max(gain))]

    #Partition some data based on a feature
    def partition(self, data, feature):
        bins = self.discretize(data[feature], self.num_bins)
        partitions = []
        for test_bin in bins:
            
            if test_bin.name is not None:
                partitions.append(data[data[feature] == test_bin.name])
            elif test_bin.lower is not None:
                partitions.append(data[(data[feature] >= test_bin.lower) & (data[feature] <= test_bin.upper)])

        #Remove the column the data were partitioned on
        for partition in partitions:
            partition.pop(feature)
        
        return partitions, bins

    #Calculate the entropy of the class labels
    def entropy(self, data):
        label_count = [0] * len(self.cbins)
        for label in data:
            label_count[self.find_bin_index(label, self.cbins)] += 1

        entropy = 0
        length = len(data)
        for count in label_count:
            if count > 0:
                entropy -= count / length * math.log2(count / length)
        return entropy

    #Search for a bin based on a key
    def find_bin(self, key, bins):
        for check_bin in bins:
            if key in check_bin:
                return check_bin
        return False
    
    #Search for the index of a bin in a list
    def find_bin_index(self, key, bins):
        for i in range(len(bins)):
            if key in bins[i]:
                return i
            
        if bins[0].lower is not None:
            if key < bins[0].lower:
                return 0
            
            if key > bins[-1].upper:
                return -1

        return False
    
    #Calculate the accuracy of the decision tree
    def get_accuracy(self):
        correct = 0
        total = len(self.data)
        for i in range(total):
            if self.data.iloc[i, -1] in self.predict(self.root, self.data.iloc[i, :]):
                correct += 1
        return correct / total

    #Predict the class label for the given data
    def predict(self, curr_node, data):
        if curr_node.label is not None:
            return curr_node.label

        bin_index = self.find_bin_index(data[curr_node.feature], curr_node.bins)
        return self.predict(curr_node.children[bin_index], data)

   
    #Plot synthetic data on function approximation background
    def plot(self, title, filename):
        #Make scatterplot
        data0 = self.data[self.data["Label"] == 0]
        data1 = self.data[self.data["Label"] == 1]
        plt.scatter(data0["A"], data0["B"], c="r", label="Class 0")
        plt.scatter(data1["A"], data1["B"], c="b", label="Class 1")
        
        #Make function approximation
        bounds = [min(self.data["A"]) - 0.5, max(self.data["A"]) + 0.5, min(self.data["B"] - 0.5), max(self.data["B"]) + 0.5]
        vals = self.sample(bounds, 150)
        plt.imshow(vals, extent=bounds, origin="lower", cmap="bwr_r", alpha=0.3)

        #Label plot
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()

        #Save to file
        plt.savefig(filename)

    #Sample tree's predictions at many different points
    def sample(self, bounds, intervals):
        xlow = bounds[0]
        xhigh = bounds[1]
        ylow = bounds[2]
        yhigh = bounds[3]
        xincr = (xhigh - xlow) / intervals
        yincr = (yhigh - ylow) / intervals

        vals = []
        ycurr = ylow
        while(ycurr < yhigh):
            xcurr = xlow
            row = []
            while(xcurr < xhigh):
                d = {"A": [xcurr], "B": [ycurr], "Label": [1]}
                d = pd.DataFrame(d)
                if d.iloc[0, -1] in self.predict(self.root, d.iloc[0, :]):
                    row.append(1)
                else:
                    row.append(0)
                xcurr += xincr
            ycurr += yincr
            vals.append(row)
        return vals
