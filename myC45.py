import pandas as pd
import math
import numpy as np

from myID3 import MyID3
from node import Node, Tree
from calculate import Calculate
from sklearn.datasets import load_iris

class MyC45:
    def __init__(self, gain_ratio=False):
        self.gain_ratio = gain_ratio

    def is_attr_continue(self, data_attr):
        ''' 
            Assume all values are the same type.
            Return True if the first value is continue
            Otherwise, return False
        '''    

        return isinstance(data_attr[0],(int, float)) 

    def find_threshold(self, data_attr, data_target):
    
        attr_name = data_attr.columns[0]
        target_name = data_target.columns[0]
        data = pd.concat([data_attr,data_target], axis = 1)
        data = data.sort_values(attr_name).reset_index(drop=True)
        
        # Retrieve all indexes with different value of target attribute
        diff_index = list()
        for i in range(len(data)-1):
            if data[target_name].iloc[i] != data[target_name].iloc[i+1]:
                diff_index.append(i)
        print(diff_index)

        

    def fit(self, data, features, target):
        continuous_features = list()
        discrete_features = list()
        for feature in features:
            is_continue = self.is_attr_continue(list(data[feature]))
            if is_continue:
               continuous_features.append(feature)

        if not continuous_features:
            return MyID3(self.gain_ratio).fit(data, features, target)
        
        # Continuous attribute
        # Sort
        # Tentuin pembatas-pembatas yang beda
        # Calculate information gain/gain ratio

        data_target = data[target]

        value_list = Calculate.get_unique_data(data, target_name)
        value_dict = dict()
        for key, value in value_list.items():
            value_dict[key] = len(value_list[key])
        
        
        # If only one value exist
        if len(value_dict == 1):
            return Tree(Node(None,
                             0.0, # Entropy must be 0 since only one value exist
                             value_dict, 
                             result = data_target[0],
                             is_leaf = True))


        if (len(features == 0)):
            return Tree(Node(None,
                             entropy_data_target,
                             value_dict, 
                             result = Calculate.most_label(data_target),
                             is_leaf = True))
        
        # Find best attribute and build tree recursively
        best_attr = 0
        best_gain = 0
        for feature in continuous_features:
            pass



# data = pd.read_csv("play_tennis.csv")
# dTree = MyC45(gain_ratio=True)
# dTree.find_threshold(data[['outlook']],data[['play']])
# dtree_view = dTree.fit(data, ['outlook', 'temp', 'humidity', 'wind'], 'play')
# dtree_view.print_tree()

iris = load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
dTree = MyC45(gain_ratio=True)
dTree.find_threshold(data[['sepal length (cm)']],data[['target']])
# print(data)