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



