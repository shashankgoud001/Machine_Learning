# ### Module imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ### Reading Data
data = pd.read_csv("Train_B_Tree.csv")

# lists for depths of trees and their respective accuracies
depths = []
accuracies = []


# file to get the output
out = open("q1_results.txt", 'w')


class Regression_tree():  # predeclaration
    pass

# ### Node Class
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, mse=None, value=None):
        ''' 
            constructor for the node
            This forms the skleton of the tree
        '''
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.mse = mse
        
        # for leaf node
        self.value = value

    # function to prune
    def post_prune_tree(self, tree, accuracy, train_data, test_data):
        '''
            This function prune the tree and tests its accuracy
            if this pruned tree performs worse than the original, then it 
            recursively finds the best pruned tree
        '''

        if self.left is None and self.right is None:
            return accuracy

        temp_left = self.left
        temp_right = self.right

        self.left = None
        self.right = None

        reg_tree = Regression_tree()
        reg_tree.root = tree

        reg_tree.train(tree, train_data)  # training the new pruned tree

        X_test, Y_test = test_data[:,:-1], test_data[:, -1]
        

        sum_squares = 0
        
        predictions = reg_tree.predict(X_test)
        for j in range(len(Y_test)):
            sum_squares += (Y_test[j]-predictions[j])**2
        
        temp_accuracy = (sum_squares/len(test_data))**0.5
        
        new_accuracy = accuracy


        depths.append(self.find_depth(tree))
        accuracies.append(temp_accuracy)

        if temp_accuracy < new_accuracy:  # if predictions on pruned tree gives less or same variance as original, then prune
            new_accuracy = temp_accuracy
            return new_accuracy
            
        else:  # else restore the original tree
            self.left = temp_left
            self.right = temp_right
            self.value = None
            

        if self.left is not None:
            new_accuracy = self.left.post_prune_tree(tree, new_accuracy, train_data, test_data)
        if self.right is not None:
            new_accuracy = self.right.post_prune_tree(tree, new_accuracy, train_data, test_data)

        return new_accuracy

    # finding the depth of the tree
    def find_depth(self, node, curr_depth = 0):
        if node is None:
            return curr_depth-2
        
        return max(self.find_depth(node.left, curr_depth+1), self.find_depth(node.right, curr_depth+1))

# ### Regression Tree Class
class Regression_tree():
    def __init__(self, min_samples_split=10):
        ''' constructor for the tree '''
        self.root = None
        self.min_samples_split = min_samples_split

    # functions required for building tree
    def get_optimal_split(self, dataset, num_features):
        ''' function to find the best split '''
        optimal_split = {}
        max_mse = -float("inf")

        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)

                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_mse = self.min_sum_sqr_error(y, left_y, right_y)

                    if curr_mse>max_mse:
                        optimal_split["feature_index"] = feature_index
                        optimal_split["threshold"] = threshold
                        optimal_split["dataset_left"] = dataset_left
                        optimal_split["dataset_right"] = dataset_right
                        optimal_split["mse"] = curr_mse
                        max_mse = curr_mse
                        
        return optimal_split
    
    # splitting the data at a node according to its threshold value
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    # finding the minimum squared errors for a regression tree
    def min_sum_sqr_error(self, parent, l_child, r_child):
        ''' function to compute variance reduction '''
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction*len(parent)
    
    # building the tree
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''
        global tree_depth
        tree_depth = max(tree_depth, curr_depth)
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        optimal_split = {}
        
        if num_samples>=self.min_samples_split:
            optimal_split = self.get_optimal_split(dataset, num_features)
            
            if optimal_split["mse"]>0:
                left_subtree = self.build_tree(optimal_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(optimal_split["dataset_right"], curr_depth+1)

                return Node(optimal_split["feature_index"], optimal_split["threshold"], left_subtree, right_subtree, optimal_split["mse"])
        
        return Node(value=np.mean(Y))

    # functions for training on the data
    def fit(self, X, Y):
        ''' function to train the tree '''
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        
    # predict function for alling over dataset
    def predict(self, X):
        ''' function to predict a single data point '''
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    # recursively finding predictions
    def make_prediction(self, x, tree):
        ''' function to predict new dataset '''
        if tree is None:
            return 0
        if tree.value != None:
            return tree.value

        feature_val = x[tree.feature_index]
        
        if tree.threshold is None:
            return 0

        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    # function to print tree
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        if not tree:
            return

        if tree.value is not None:
            print(tree.value, file=out)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.mse, file=out)
            print("%sleft:" % (indent), end="", file=out)
            self.print_tree(tree.left, indent + " ")
            print("%sright:" % (indent), end="", file=out)
            self.print_tree(tree.right, indent + " ")
    
    # Training the tree
    def train(self, curr_node, data_set):
        if curr_node.left is None and curr_node.right is None and len(data_set) != 0:
            Y = data_set[:, -1].reshape(-1, 1)
            val = np.mean(Y)
            curr_node.val = val
            return

        if data_set is None:
            data_left = None
            data_right = None
            return
        if len(data_set) == 0:
            return

        data_left = np.array([row for row in data_set if row[curr_node.feature_index] <= curr_node.threshold])
        data_right = np.array([row for row in data_set if row[curr_node.feature_index] > curr_node.threshold])

        if curr_node.left is not None:
            self.train(curr_node.left, data_left)

        if curr_node.right is not None:
            self.train(curr_node.right, data_right)

# ### Split train-test, Fit model and Test Model
tree_depth = 0
over_fit_accuracy = float("inf")
over_fit_height = 0
min_sqr_sum = float("inf")

for i in range(10):
    train_data = data.sample(frac = 0.7)
    test_data = data.drop(train_data.index)

    # splitting training data
    features_train = train_data.iloc[:,:-1].values
    Y_train = train_data.iloc[:, -1].values.reshape(-1, 1)

    # splitting test data
    features_test = test_data.iloc[:,:-1].values
    Y_test = test_data.iloc[:, -1].values.reshape(-1, 1)
    
    tree_depth = 0

    # training
    regression_tree = Regression_tree(min_samples_split = 10)
    regression_tree.fit(features_train, Y_train)
    predicted_values = regression_tree.predict(features_train)

    sum_squares = 0
    for j in range(len(Y_train)):
        sum_squares += (Y_train[j][0]-predicted_values[j])**2
    
    sum_squares = (sum_squares/len(Y_train))**0.5
    
    if over_fit_accuracy > sum_squares:
        over_fit_accuracy = sum_squares
        over_fit_height = tree_depth

    # testing
    predicted_values = regression_tree.predict(features_test)
    sum_squares = 0
    for j in range(len(Y_test)):
        sum_squares += (Y_test[j][0]-predicted_values[j])**2
    
    sum_squares = (sum_squares/len(Y_train))**0.5

    if min_sqr_sum > sum_squares:
        optimal_tree = regression_tree
        min_sqr_sum = sum_squares
        optimal_depth = tree_depth

accuracy = min_sqr_sum
print("#################################################################################", file=out)
print("The accuracy and depth of the regression tree before pruning are:", file=out)
print("Best Accuracy: ", accuracy, file=out)
print("Depth of the optimal ", optimal_depth, file=out)
print(f"The tree overfits at a height of {over_fit_height} with accuracy {over_fit_accuracy}", file=out)

reg_tree = optimal_tree.root
train_data = data.sample(frac = 0.7)
test_data = data.drop(train_data.index)
train_data = train_data.iloc[:,:].values
test_data = test_data.iloc[:,:].values
reg_tree.post_prune_tree(reg_tree, accuracy, train_data, test_data)

zipped = zip(accuracies, depths)
zipped = sorted(zipped)
accuracies, depths = zip(*zipped)

# plotting and saving
plt.plot(accuracies, depths, "b")
plt.xlabel("Variance")
plt.ylabel("depths")
plt.savefig("regression_plot.jpg")


print("\n\n#################################################################################", file=out)
print("The accuracy and depth after pruning are:", file=out)
print("Accuracy: ", accuracies[0], file=out)
print("Depth: ", depths[0], file=out)


print("\n\n#################################################################################", file=out)
print("Index    Feature", file=out)
print("    0 -> Cement", file=out)
print("    1 -> Slag", file=out)
print("    2 -> Fly Ash", file=out)
print("    3 -> Water", file=out)
print("    4 -> Super Plasticizer", file=out)
print("    5 -> Coarse Aggregate", file=out)
print("    6 -> Fine Aggregate", file=out)
print("    7 -> Age", file=out)
print("\n\nThe pruned tree is: ", file=out)
optimal_tree.print_tree(tree=reg_tree)

# closing the file
out.close()