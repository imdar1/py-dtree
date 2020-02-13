import pandas as pd
import math
from myID3 import MyID3
from node import Node, Tree
from calculate import Calculate

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

    def fit(self, data, features, target):
        continuous_feature = list()
        for feature in features:
            is_continue = self.is_attr_continue(list(data[feature]))
            if is_continue:
               continuous_feature.append(feature)

        if not continuous_feature:
            return MyID3(self.gain_ratio).fit(data, features, target)
        
        # Continuous attribute
        



data = pd.read_csv("play_tennis.csv")
dTree = MyC45(gain_ratio=True)
dtree_view = dTree.fit(data, ['outlook', 'temp', 'humidity', 'wind'], 'play')
dtree_view.print_tree()