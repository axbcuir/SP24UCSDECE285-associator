class Node(object):

    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = []

    def get_value(self):
        return self.value

    def set_parent(self, parent):
        if self.parent is not None:
            self.parent.children.remove(self)
        self.parent = parent
        self.parent.children.append(self)

    def get_root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.get_root()

    def get_children_r(self):
        """
        Recursively get children.
        :return: list of Node objects
        """
        children_r = self.children
        for child in self.children:
            children_r = children_r + child.get_children_r()
        return children_r


def trees_from_sorted_list(sorted_list, is_parent_func):
    """
    sorted_list is a list of masks sorted by decending area. is_parent_func is a function that judges whether a node
    is a parent. Returns a list of Node objects.
    :param sorted_list: list of values, to be contained in trees
    :param is_parent_func: function
    :return: list of Nodes
    """
    nodes = [Node(val) for val in sorted_list]
    for i in range(len(sorted_list) - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            if is_parent_func(sorted_list[i], sorted_list[j]):
                nodes[i].set_parent(nodes[j])

    return nodes


def nodes_to_list(nodes):
    """
    Unwraps a list of Node objects. Returns a list of values (not Node objects).
    :param nodes: list of Node objects
    :return: list
    """
    return [node.get_value() for node in nodes]
