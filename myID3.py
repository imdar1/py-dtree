import pandas as pd
import math
from node import Node, Tree

def entropy(data_target):
    # Create dictionary to store number of target in data
    target_dict = dict()
    n_total = len(data)
    for target in data_target:
        # If target value is not in dictionary, add it to dictionary
        if target not in target_dict:
            target_dict[target] = 1
        # If target value is already in dictionary, increment it in dictionary
        else:
            target_dict[target] += 1

    # Count entropy of data
    if (n_total <= 0):
        return -9999
    else:
        result_entropy = 0
        if len(target_dict) == 1:
            return 0
        else:
            for target in target_dict:
                result_entropy -= (target_dict[target]/n_total) * math.log(target_dict[target]/n_total) # Entropy formula

        return result_entropy

def get_unique_data(data, attribute_name):
    attr_dict = dict()
    
    for attr_value in data[attribute_name]:
        if attr_value not in attr_dict:
            attr_dict[attr_value] = data.loc[data[attribute_name] == attr_value]

    return attr_dict

def most_label(data_target):
    target_list = dict()
    best_label_n = 0
    best_label = ''
    for target in data_target:
        if target not in target_list:
            target_list[target] = 1
        else:
            target_list[target] += 1

        if best_label_n < target_list[target]:
            best_label = target
            best_label_n = target_list[target]
    
    return best_label

def info_gain(data_attr, data_target):
    # save the entropy of the current data_target
    entropy_parent = entropy(data_target)

    # Clustering data_target based on value on attribute
    attr_dict = dict()
    i = 0
    for attr_value in data_attr:
        if attr_value not in attr_dict:
            attr_dict[attr_value] = [data_target[i]]
        else:
            attr_dict[attr_value].append(data_target[i])
        i += 1

    # Count the info gain
    result_info_gain = entropy_parent
    for attr in attr_dict:
        result_info_gain -= len(attr_dict[attr])/len(data_target)*entropy(attr_dict[attr])

    return result_info_gain

def myID3(data, attributes, target_name):
    data_target = data[target_name]
    
    # Data target contains one label
    if entropy(data_target) == 0:
        value_list = get_unique_data(data, target_name)
        value_dict = dict()
        for key, value in value_list.items():
            value_dict[key] = len(value_list[key])

        # Set current_node, info_gain, values
        tree = Tree(Node(None,
                         entropy(data_target), 
                         value_dict, 
                         result = data_target[0],
                         is_leaf = True))
        return tree
    
    # Nothing attribute shall be chosen
    if len(attributes) == 0:
        # Set current_node, info_gain, values
        value_list = get_unique_data(data, target_name)
        value_dict = dict()
        for key, value in value_list.items():
            value_dict[key] = len(value_list[key])

        tree = Tree(Node(None,
                         entropy(data_target),
                         value_dict, 
                         result = most_label(data_target),
                         is_leaf = True))
        return tree
    else:
        # Find best attribute to be node
        best_attr = ''
        best_info_gain = 0
        for attr in attributes:
            if info_gain(data[attr], data_target) > best_info_gain:
                best_info_gain = info_gain(data[attr], data_target)
                best_attr = attr

        value_list = get_unique_data(data, target_name)
        value_dict = dict()
        for key, value in value_list.items():
            value_dict[key] = len(value_list[key])

        # Build decision tree recursively
        dtree = Tree(Node(best_attr, best_info_gain, value_dict))

        # Delete usage attribute in attributes
        attributes.remove(best_attr)
        
        # Scan all posible value to be generated subtree
        list_attribute = get_unique_data(data, best_attr)
        i = 0
        for attribute in list_attribute:
            data = pd.DataFrame(data=list_attribute[attribute]).reset_index(drop=True)
            data.drop(best_attr, axis = 1, inplace=True)
            dtree.add_child(myID3(data, attributes, target_name))
            dtree.children[i].value.edge = attribute
            i += 1
        return dtree
            


data = pd.read_csv("play_tennis.csv")
print('INFO GAIN OUTLOOK', info_gain(data['outlook'], data['play']))
print('INFO GAIN TEMP', info_gain(data['temp'], data['play']))
print('INFO GAIN HUMIDITY', info_gain(data['humidity'], data['play']))
print('INFO GAIN WIND', info_gain(data['wind'], data['play']))
print(data)
dtree = myID3(data, ['outlook', 'temp', 'humidity', 'wind'], 'play')
dtree.print_tree()