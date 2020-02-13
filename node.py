# Node representation of decision tree

class Node:
    def __init__(self, current_node, info_gain, values, edge = None, result = None, is_leaf=False):
        self.current_node = current_node
        self.info_gain = round(info_gain,3)
        self.values = values
        self.edge = edge
        self.result = result
        self.is_leaf = is_leaf
    
    def __str__(self):
        node_str = ''

        if self.edge is not None:
            node_str += '['+ str(self.edge) +']' + '; '

        if self.current_node is not None:
            node_str += str(self.current_node)

        node_str += '; VALUES: '+ str(self.values)

        if self.result is not None:
            node_str += '; RESULT: '+ str(self.result)

        return node_str


class Tree:

    def __init__(self, value=None):
        self.value = value
        self.children = list()

    def add_child(self, node):
        self.children.append(node)  

    def set_node(self, node):
        self.value = node

    def __text_builder(self, tree, depth):
        current_indent = ''
        for i in range(depth):
            current_indent +='----'
        current_indent += '| '

        if not tree.children:
            return current_indent + str(tree.value) +'\n'

        current_node = current_indent + str(tree.value) +'\n'
        for child in tree.children:
            current_node += str(self.__text_builder(child, depth+1))
        return current_node
       

    def print_tree(self):
        print(self.__text_builder(self,0))