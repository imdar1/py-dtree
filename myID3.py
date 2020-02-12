import pandas as pd
import math
from node import Node, Tree
from calculate import Calculate

def myID3(data, attributes, target_name, gain_ratio=False):
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
    if len(attributes) == 0:
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
            if gain_ratio:
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
            dtree.add_child(myID3(data, attributes, target_name, gain_ratio))
            dtree.children[i].value.edge = attribute
            i += 1
        return dtree
            


data = pd.read_csv("play_tennis.csv")
# print('INFO GAIN OUTLOOK', Calculate.info_gain(data['outlook'], data['play']))
# print('INFO GAIN TEMP', Calculate.info_gain(data['temp'], data['play']))
# print('INFO GAIN HUMIDITY', Calculate.info_gain(data['humidity'], data['play']))
# print('INFO GAIN WIND', Calculate.info_gain(data['wind'], data['play']))
print(data)
dtree = myID3(data, ['outlook', 'temp', 'humidity', 'wind'], 'play', True)
dtree.print_tree()