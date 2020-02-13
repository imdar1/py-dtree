from node import Node, Tree

# dtree = ID3DecisionTree()
# dtree = dtree.fit()
node1 = Node('sunny', 0.55, [5,10,20])
# print(node1)
# dtree.children[0].add_child(Node(7))
# dtree.children[0].add_child(Node(8))
# dtree.children[1].add_child(Node(9))
dtree = Tree(node1)
dtree.add_child(Tree(Node('hot', 0.1, [5,1,20])))
dtree.add_child(Tree(Node('cold', 0.2, [5,0,20])))
dtree.print_tree()

# outlook