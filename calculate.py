
import math

class Calculate:
    @staticmethod
    def entropy(data_target):
        # Create dictionary to store number of target in data
        target_dict = dict()
        n_total = len(data_target)
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
                result_entropy -= (target_dict[target]/n_total) * math.log(target_dict[target]/n_total, 2) # Entropy formula

            return result_entropy

    @staticmethod
    def split_data(data_attr, data_target):
        attr_dict = dict()
        i = 0
        for attr_value in data_attr:
            if attr_value not in attr_dict:
                attr_dict[attr_value] = [data_target[i]]
            else:
                attr_dict[attr_value].append(data_target[i])
            i += 1
        
        return attr_dict

    @staticmethod
    def split_continue(data_attr, data_target, splitter):
        attr_dict = dict()
        # splitter = (data_attr.iloc[split_index]+data_attr.iloc[split_index+1])/2
        i = 0
        attr_dict['>'] = []
        attr_dict['<='] = []
        for i in range(len(data_attr)):
            if data_attr.iloc[i] > splitter:
                attr_dict['>'].append(data_target.iloc[i])
            else:
                attr_dict['<='].append(data_target.iloc[i])
        
        return attr_dict
        
    @staticmethod
    def info_gain(data_attr, data_target, is_continue=False, split_index=0):
        # save the entropy of the current data_target
        entropy_parent = Calculate.entropy(data_target)

        if is_continue:
            splitter = (data_attr[split_index]+data_attr[split_index+1])/2
            attr_dict = Calculate.split_continue(data_attr,data_target, splitter)
        else:
            # Clustering data_target based on value on attribute
            attr_dict = Calculate.split_data(data_attr, data_target)

        # Count the info gain
        result_info_gain = entropy_parent
        for attr in attr_dict:
            result_info_gain -= len(attr_dict[attr])/len(data_target)*Calculate.entropy(attr_dict[attr])

        return result_info_gain

    @staticmethod
    def gain_ratio(data_attr, data_target, is_continue=False, split_index=0):
        # save the entropy of the current data_target
        entropy_parent = Calculate.entropy(data_target)

        if is_continue:
            splitter = (data_attr[split_index]+data_attr[split_index+1])/2
            attr_dict = Calculate.split_continue(data_attr,data_target, splitter)

        else:
            # Clustering data_target based on value on attribute
            attr_dict = Calculate.split_data(data_attr, data_target)

        # Count the info gain
        info_gain = entropy_parent
        split_info = 0
        for attr in attr_dict:
            info_gain -= len(attr_dict[attr])/len(data_target)*Calculate.entropy(attr_dict[attr])
            if len(attr_dict[attr]) != 0:
                split_info -= len(attr_dict[attr])/len(data_target)*math.log(len(attr_dict[attr])/len(data_target), 2)
            else:
                return 0
        
        return info_gain/split_info

    @staticmethod
    def get_unique_data(data, attribute_name):
        attr_dict = dict()
        
        for attr_value in data[attribute_name]:
            if attr_value not in attr_dict:
                attr_dict[attr_value] = data.loc[data[attribute_name] == attr_value]

        return attr_dict

    @staticmethod
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

    @staticmethod
    def split_by_threshold(data, attribute_name, best_splitter):
        attr_dict = dict()
        attr_dict['>'] = data.loc[data[attribute_name] > best_splitter]
        attr_dict['<='] = data.loc[data[attribute_name] <= best_splitter]

        return attr_dict

    @staticmethod
    def most_value(value_dict):
        best_number = 0
        best_value = ''
        for value in value_dict:
            if value_dict[value] > best_number:
                best_value = value
                best_number = value_dict[value]

        return best_value
