#!/usr/bin/env python
# -*- coding: utf-8 -*-



class DepNode:
    def __init__(self, word_idx):
        self.idx = word_idx
        self.parent = None
        self.children = []
        self.children_num = 0

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)
        self.children_num += 1


class DepTree:
    def __init__(self):
        self.root = None
        self.size = None

    def assign_root(self, node):
        self.root = node

    def merge(self, a_tree):
        self.root.add_child(a_tree.root)

    def get_size(self):
        if self.size is None:
            self.size = 0
            node_queue = [self.root]
            while node_queue:
                node = node_queue.pop(0)
                self.size += 1
                for child_node in node.children:
                    node_queue.append(child_node)
        return self.size

    def bfs_tranverse(self):
        pass

    def post_order_tranverse(self, start_node=None, node_list=[]):
        if start_node is None:
            start_node = self.root
        for child in start_node.children:
            self.post_order_tranverse(start_node=child, node_list=node_list)
        node_list.append(start_node)
        return node_list


class ConstNode:
    def __init__(self, tag, word=None, word_idx=None, children=None):
        self.tag = tag
        self.word = word
        self.idx = word_idx
        self.parent = None
        self.children = children if children is not None else []

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def to_string(self):
        """convert the node and its children to string"""
        output = '('
        output += ' ' + self.tag + ' '
        if self.children:
            for child_node in self.children:
                output += child_node.to_string() + ' '
        else:
            output += str(self.word) + ' '
        output += ')'
        return output


class ConstTree:
    def __init__(self):
        self.root = None
        self.size = None
        self.leaf_num = None

    def load_from_string(self, s):
        tokens = s.strip().replace('\n', '').replace('(', ' ( ').replace(')', ' ) ').split()
        queue = tokens
        stack = []
        self.leaf_num = 0
        while queue:
            token = queue.pop(0)
            if token == ')':
                # If ')', start processing
                content = []  # Content in the stack
                while stack:
                    cont = stack.pop()
                    if cont == '(':
                        break
                    else:
                        content.append(cont)
                content.reverse()
                if len(content) == 0:
                    raise ValueError('empty content!')
                tag = content.pop(0)
                # the leaf node with word
                if not isinstance(content[0], ConstNode):
                    if len(content) > 1:
                        # print('Multiple words in one leaf node: ')
                        # print(content)
                        temp = ''.join(content)
                        content[0] = temp
                    self.leaf_num += 1
                    node = ConstNode(tag, word=content[0], word_idx=self.leaf_num-1)
                # the internal node
                else:
                    node = ConstNode(tag, children=content)
                    for child_node in content:
                        assert isinstance(child_node, ConstNode)
                        child_node.parent = node
                stack.append(node)
            else:
                # else, keep push into the stack
                stack.append(token)
        self.root = stack[0]

    def merge(self, a_tree):
        self.root.add_child(a_tree.root)
        self.leaf_num += a_tree.leaf_num

    def get_size(self):
        if self.size is None:
            self.size = 0
            node_queue = [self.root]
            while node_queue:
                node = node_queue.pop(0)
                self.size += 1
                for child_node in node.children:
                    node_queue.append(child_node)
        return self.size

    def bfs_tranverse(self):
        node_list = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            for child in node.children:
                queue.append(child)
            node_list.append(node)
        return node_list

    def post_order_tranverse(self, start_node=None, node_list=[]):
        if start_node is None:
            start_node = self.root
        if len(start_node.children) > 0:
            for child in start_node.children:
                self.post_order_tranverse(start_node=child, node_list=node_list)
        node_list.append(start_node)
        return node_list

    def compress(self):
        """remove redundant internal node"""
        node_queue = [self.root]
        while node_queue:
            cur_node = node_queue.pop(0)
            for child_idx in range(len(cur_node.children)):
                child_node = cur_node.children[child_idx]
                # replace the child node with a single child itself
                while len(child_node.children) == 1:
                    child_node = child_node.children[0]
                    cur_node.children[child_idx] = child_node
                    child_node.parent = cur_node
                node_queue.append(child_node)

    def binarize(self):
        """convert to binary tree"""
        node_queue = [self.root]
        while node_queue:
            node = node_queue.pop(0)
            # Construct binary tree
            if len(node.children) > 2:
                # Right-branching
                child_1 = node.children.pop(0)
                child_2 = ConstNode(tag=node.tag, children=[child for child in node.children])
                child_2.parent = node
                for child in child_2.children:
                    child.parent = child_2
                node.children = [child_1, child_2]
            node_queue += node.children

    def to_string(self):
        """convert the tree to string"""
        return self.root.to_string()



