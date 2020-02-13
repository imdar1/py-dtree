import pandas as pd
import math

from node import Node, Tree
from calculate import Calculate

class MyID3():
    def __init__(self, gain_ratio=False):
        '''
            MyID3 constructor.
        '''

        self.gain_ratio = gain_ratio

    def fit(self, data, attributes, target_name):
        '''
            Built and return decision tree using ID3 algorithm
        '''

        data_target = data[target_name]
        
        # Data target contains one label
        entropy_data_target = Calculate.entropy(data_target)
        if entropy_data_target == 0:
            value_list = Calculate.get_unique_data(data, target_name)
            value_dict = dict()
            for key, value in value_list.items():
                value_dict[key] = len(value_list[key])

            # Set current_node, info_gain, values
            tree = Tree(Node(None,
                            entropy_data_target, 
                            value_dict, 
                            result = data_target[0],
                            is_leaf = True))
            return tree
        
        # Nothing attribute shall be chosen
        if len(attributes) == 0: # Masi salah
            # Set current_node, info_gain, values
            value_list = Calculate.get_unique_data(data, target_name)
            value_dict = dict()
            for key, value in value_list.items():
                value_dict[key] = len(value_list[key])

            tree = Tree(Node(None,
                            entropy_data_target,
                            value_dict, 
                            result = Calculate.most_label(data_target),
                            is_leaf = True))
            return tree
        else:
            # Find best attribute to be node using either info gain or gain ratio
            best_attr = ''
            best_point = 0 # Could be Info gain or Gain ratio
            for attr in attributes:
                if self.gain_ratio:
                    point = Calculate.gain_ratio(data[attr], data_target)
                    if point > best_point:
                        best_point = point
                        best_attr = attr
                else:
                    point = Calculate.info_gain(data[attr], data_target)
                    if point > best_point:
                        best_point = point
                        best_attr = attr

            value_list = Calculate.get_unique_data(data, target_name)
            value_dict = dict()
            for key, value in value_list.items():
                value_dict[key] = len(value_list[key])

            # Build decision tree recursively
            dtree = Tree(Node(best_attr, best_point, value_dict))

            # Delete usage attribute in attributes
            attributes.remove(best_attr)
            
            # Scan all posible value to be generated subtree
            list_attribute = Calculate.get_unique_data(data, best_attr)
            i = 0
            for attribute in list_attribute:
                data = pd.DataFrame(data=list_attribute[attribute]).reset_index(drop=True)
                data.drop(best_attr, axis = 1, inplace=True)
                dtree.add_child(self.fit(data, attributes, target_name))
                dtree.children[i].value.edge = attribute
                i += 1
            return dtree

    def _get_result(self, dtree, test):
        '''
            Get predicted result of test which only contains one tuple
        '''

        if dtree.value.result is not None:
            return dtree.value.result
        
        curr_attr = dtree.value.current_node
        curr_edge = test[curr_attr].iloc[0]
        
        # Find index of the edge in the tree children
        i = 0;
        for child in dtree.children:
            if curr_edge == child.value.edge:
                break
            else:
                i += 1

        return self._get_result(dtree.children[i], test)
            
    def predict(self, dtree, data_test):
        '''
            Predict data_test based on dtree
        '''
        
        predicted_result = list()
    
        # Traverse through each row
        for i in range(len(data_test)):
            predicted_result.append(self._get_result(dtree, data_test.iloc[[i]]))
        
        return predicted_result
        