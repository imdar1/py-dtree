# Node representation of decision tree

class Node:

    def __init__(self, value):
        
        self.value = value
        self.children = list()

    def add_child(self, node):
        self.children.append(node)          

    def __text_builder(self, tree, depth):
        current_indent = ''
        for i in range(depth):
            current_indent +='----'
        current_indent += '| '

        if not tree.children:
            return current_indent+str(tree.value)+'\n'

        current_node = current_indent + str(tree.value)+'\n'
        for child in tree.children:
            current_node += str(self.__text_builder(child, depth+1))
        return current_node
       

    def print_tree(self):
        print(self.__text_builder(self,0))