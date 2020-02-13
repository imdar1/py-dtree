class Condition:
    def __init__(self, attribute, value, parameter):
        self.attribute = attribute
        self.value = value
        self.parameter = parameter # =, <=, >=, >, <

    def __str__(self):
        return self.attribute + ' ' + self.parameter + ' ' + self.value

    def check_condition(self, attr, val):
        if (self.attribute == attr):
            if (self.parameter == '=='): # val equal with value
                return val == self.value
            elif (self.parameter == '<='):
                return val <= self.value
            elif (self.parameter == '>='):
                return val >= self.value 
            elif (self.parameter == '>'):
                return val > self.value
            elif (self.parameter == '<'):
                return val < self.value
        else:
            return False

class Rules:
    def __init__(self, list_condition, result):
        self.list_condition = list_condition
        self.result = result

    def __str__(self):
        string = 'IF '
        i = 0
        for condition in self.list_condition:      
            string += '(' + str(condition) + ') '

        string += ' THEN '
        string += self.result
        
        return string

    def check_rules(self, data):
        for condition in self.list_condition:
            if data[condition.attribute] is not None:
                if not condition.check_condition(condition.attribute, data[condition.attribute]):
                    return False
            else:
                return False
        return True
