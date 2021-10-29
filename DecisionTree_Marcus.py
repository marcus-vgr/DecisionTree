import numpy as np


class Node:  # An Node is where there is a separation between two child-trees. 
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        '''This is just a constructor --> save info'''

        #decision mode
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        #leaf mode (node where data is already homogeneous)
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=2):

        #Initialize root of the tree
        self.root = None

        #Stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        '''recursive function to build the tree'''

        X, Y = dataset[:,:-1], dataset[:,-1]  # We are assuming that Y, the value we want to predict, 
        # is in the last column of the pandas dataframe.

        num_samples, num_features = np.shape(X) # Collecting info about the dataset

        #Split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            #find best split
            best_split = self.get_best_split(dataset, num_features)
            #Check if info gain is positive
            if best_split['info_gain'] > 0:
                left_subtree = self.build_tree(best_split['dataset_left'], curr_depth+1)
                right_subtree = self.build_tree(best_split['dataset_right'], curr_depth+1)
                # return Node
                return Node(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree, 
                best_split['info_gain'])

        # compute leaf node
        leaf_value = self.calculate_leaf_node(Y)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_features):
        '''Function to get the best split'''

        #store the best split info
        best_split = {}
        max_info_gain = -float('inf') 

        #Loop over all features
        for feature_index in range(num_features):
            feature_values = dataset[:,feature_index]
            possible_thresholds = np.unique(feature_values)
            #Loop over all feature values
            for threshold in possible_thresholds:
                #get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                #Check if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:,-1], dataset_left[:,-1], dataset_right[:,-1]
                    #Compute info gain
                    curr_info_gain = self.information_gain(y,left_y, right_y)
                    #Update best split
                    if curr_info_gain > max_info_gain:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['info_gain'] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split

    def split(self, dataset, feature_index, threshold):
        '''to Split data given a threshold'''

        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child): 
        '''Calculate Information Gain using GINI'''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)

        gain = self.gini_index(parent) - weight_l*self.gini_index(l_child) - weight_r*self.gini_index(r_child)

        return gain

    def gini_index(self, y):
        '''Calculate gini index'''

        class_labels = np.unique(y)
        gini = 0
        for c in class_labels:
            probability_c = len(y[y==c]) / len(y)
            gini += probability_c**2
        
        return 1 - gini

    def calculate_leaf_node(self, Y):
        '''function to compute leaf node'''
        Y = list(Y)
        return max(Y, key=Y.count)

    def fit(self, X, Y):
        '''function to train the tree'''

        dataset = np.concatenate((X,Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        '''make a prediction of a hole dataset'''
        
        return [self.make_prediction(x, self.root) for x in X]

    def make_prediction(self, x, tree):
        '''Make predicition of a single data point'''

        if tree.value != None: #Basically check if it is a leaf 
            return tree.value 

        # Run over the whole tree
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)


    def print_tree(self, tree=None):
        '''Function to print tree'''

        if not tree:
            tree = self.root
        
        if tree.value is not None:
            print(tree.value)
        
        else:
            print(f'X_{tree.feature_index} <= {tree.threshold} ? Info gain = {tree.info_gain}')
            print("     left:", end="")
            self.print_tree(tree.left)
            print("     right:", end="")
            self.print_tree(tree.right)