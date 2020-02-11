import pandas as pd
import math

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
        print('DAUN ', data_target[0])
    
    # Nothing attribute shall be chosen
    if len(attributes) == 0:
        print('DAUN ', most_label(data_target))
    else:
        # Find best attribute to be node
        best_attr = ''
        best_info_gain = 0
        for attr in attributes:
            if info_gain(data[attr], data_target) > best_info_gain:
                best_info_gain = info_gain(data[attr], data_target)
                best_attr = attr

        print('ATTR ', best_attr)

        # Delete usage attribute in attributes
        attributes.remove(best_attr)
        
        # Scan all posible value to be generated subtree
        list_attribute = get_unique_data(data, best_attr)
        for attribute in list_attribute:
            myID3(list_attribute[attribute], attributes, target_name)



# data = pd.read_csv("play_tennis.csv")
# print('INFO GAIN OUTLOOK', info_gain(data['outlook'], data['play']))
# print('INFO GAIN TEMP', info_gain(data['temp'], data['play']))
# print('INFO GAIN HUMIDITY', info_gain(data['humidity'], data['play']))
# print('INFO GAIN WIND', info_gain(data['wind'], data['play']))

# myID3(data, ['outlook', 'temp', 'humidity', 'wind'], 'play')




