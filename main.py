from DTree import DTree

#tree = DTree("./synthetic-1.csv", 2)
#print(tree.get_accuracy())
#tree.plot("Plot 1", "plot1.png")

#tree = DTree("./synthetic-2.csv", 7)
#print(tree.get_accuracy())
#tree.plot("Plot 2", "plot2.png")

#tree = DTree("./synthetic-3.csv", 7)
#print(tree.get_accuracy())
#tree.plot("Plot 3", "plot3.png")

#tree = DTree("./synthetic-4.csv", 5)
#print(tree.get_accuracy())
#tree.plot("Plot 4", "plot4.png")

tree = DTree("./Video_Games_Sales.csv", 2)
print(tree.get_accuracy())