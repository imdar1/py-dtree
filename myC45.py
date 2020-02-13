import pandas as pd
import math
import numpy as np
import re

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

        best_point = 0
        best_idx = -1
        for i in diff_index:
            if (self.gain_ratio):
                point = Calculate.gain_ratio(data[attr_name], data[target_name],
                                             is_continue=True, split_index=i)
            else:
                point = Calculate.info_gain(data[attr_name], data[target_name],
                                            is_continue=True, split_index=i)
            if point > best_point:
                best_point = point
                best_idx = i
        
        best_splitter = (data[attr_name].iloc[best_idx]+data[attr_name].iloc[best_idx+1])/2

        return [best_splitter, best_point]


    def fit(self, data, features, target):
        continuous_features = list()
        discrete_features = list()
        for feature in features:
            if len(list(data[feature]))>0:
                is_continue = self.is_attr_continue(list(data[feature]))
                if is_continue:
                    continuous_features.append(feature)
                else:
                    discrete_features.append(feature)

        if not continuous_features:
            return MyID3(self.gain_ratio).fit(data, features, target)
        
        # Continuous attribute
        
        # If only one value exist
        entropy_data_target = Calculate.entropy(data[target])
        if entropy_data_target == 0:
            value_list = Calculate.get_unique_data(data, target)
            value_dict = dict()
            for key, value in value_list.items():
                value_dict[key] = len(value_list[key])

            return Tree(Node(None,
                             0.0, # Entropy must be 0 since only one value exist
                             value_dict, 
                             result = data[target][0],
                             is_leaf = True))

        if (len(features) == 0):
            value_list = Calculate.get_unique_data(data, target)
            value_dict = dict()
            for key, value in value_list.items():
                value_dict[key] = len(value_list[key])
            return Tree(Node(None,
                             entropy_data_target,
                             value_dict, 
                             result = Calculate.most_label(data_target),
                             is_leaf = True))
        
        # Find best attribute and build tree recursively
        best_attr = ''
        best_point = 0
        is_discrete = False
        best_splitter = 0
        chosen_edge = list(['',''])
        for feature in continuous_features:
            best_treshold = self.find_threshold(data[[feature]], data[[target]])
            if best_treshold[1] > best_point:
                best_attr = str(feature)
                chosen_edge[0] = best_attr + ' > ' + str(best_treshold[0])
                chosen_edge[1] = best_attr + ' <= ' + str(best_treshold[0])
                best_point = best_treshold[1]
                best_splitter = best_treshold[0]
        for feature in discrete_features:
            point = Calculate.info_gain(data[attr], data_target)
            if point > best_point:
                best_point = point
                best_attr = str(feature)
                is_discrete = True

        value_list = Calculate.get_unique_data(data, target)
        value_dict = dict()
        for key, value in value_list.items():
            value_dict[key] = len(value_list[key])
        dtree = Tree(Node(best_attr, best_point, value_dict))

        # Delete usage attribute in attributes
        # features.remove(best_attr)
        
        # Scan all posible value to be generated subtree
        if is_discrete:
            list_attribute = Calculate.get_unique_data(data, best_attr) 
        else:
            list_attribute = Calculate.split_by_threshold(data, best_attr, best_splitter)

        i = 0

        for attribute in list_attribute:
            data = pd.DataFrame(data=list_attribute[attribute]).reset_index(drop=True)
            # data.drop(best_attr, axis = 1, inplace=True)
            dtree.add_child(self.fit(data, features, target))
            if is_discrete:
                dtree.children[i].value.edge = attribute
            else:
                dtree.children[i].value.edge = chosen_edge[i]
            i += 1

        return dtree

    def predict(self, dtree, data_test):
        predicted_result = list()
    
        # Traverse through each row
        for i in range(len(data_test)):
            predicted_result.append(self._get_result(dtree, data_test.iloc[[i]]))
        
        return predicted_result

    def _get_result(self, dtree, test):
        if dtree.value.result is not None:
            return dtree.value.result
        
        curr_attr = dtree.value.current_node
        is_continue = self.is_attr_continue(list(data[curr_attr]))
        curr_data = test[curr_attr].iloc[0]
        
        if is_continue:
            splitter = re.findall("\d+\.\d+", dtree.children[0].value.edge)
            if float(curr_data) > float(splitter[0]):
                i = 0 # First index of children 
            else:
                i = 1

        else:
            # Find index of the edge in the tree children
            i = 0;
            for child in dtree.children:
                if curr_data == child.value.edge:
                    break
                else:
                    i += 1

        return self._get_result(dtree.children[i], test)



# data = pd.read_csv("play_tennis.csv")
# dTree = MyC45(gain_ratio=True)
# dTree.find_threshold(data[['outlook']],data[['play']])
# dtree_view = dTree.fit(data, ['outlook', 'temp', 'humidity', 'wind'], 'play')
# dtree_view.print_tree()

iris = load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                    columns= iris['feature_names'] + ['target'])
dTree = MyC45(True)
dtree_view = dTree.fit(data, iris['feature_names'], 'target')
dtree_view.print_tree()
print('ACTUAL: ', iris['target'])
print('FEATURE: ', iris['feature_names'])
hasil = dTree.predict(dtree_view, data[iris['feature_names']])
print('PREDICTED: ', hasil)

# splitter = data['sepal length (cm)'].iloc[15]
# print(type(splitter))
# dTree.find_threshold(data[['sepal length (cm)']],data[['target']])
# print(data))
