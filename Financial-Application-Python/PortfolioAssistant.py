"""
Implemented algorithm
Kruskal minimal spanning tree for finding the least correlated path for a set of stocks
This project can be used to building a low-correlation stock portfolio
author: Yi Rong
updated on 1/25/21
reference: https://miscj.aut.ac.ir/article_490.html
"""

import numpy as np
import pandas as pd
import datetime as dt
import pdb
import sys
import os

sys.path.append(os.path.abspath(
    "F:\\YI RONG\\BU\\academic\\CS 566 ANALYSIS OF ALGORITHMS\\Project\\CS566_Project\\TradeAssistant.py"))
import TradeAssistant as ta


class Graph_MST():

    def __init__(self, num_vertex):
    # initialize the vertex number and the graph
        self.v = num_vertex
        self.graph = []

    def addEdge(self, u, v, w):
    # intent: add edge to self.graph
    # Pre1: u and v are non-negative integers, w is a non-negative number
    # e.g. [u, v, w] = [0, 1, 0.5], in which u is node 0, v is node 1 and their edge weight is 0.5
    # Pre2: u != v
    # Post1: self.graph is a list of edges, which is defined as [u, v, w]

        self.graph.append([u, v, w])

    def findParent(self, parent, i):
    # intent: find root parent node of node i
    # Pre1: parent is a list of |V| integers and its range is among [0, len(parent))
    # e.g. parent[i] is a node index, which is the parent node of node i
    # Pre2: i is among [0, len(parent))
    # Post1: for final return_result, parent[return_result] = return_result

        # Sa: if parent of node i is not i
        # XOR
        # i is returned
        if parent[i] == i:
            return i

        # Sb: find parent of node parent[i]
        return self.findParent(parent, parent[i])

    def union(self, parent, rank, x, y):
    # intent: union two nodes through setting one as the other's parent
    # Pre1: parent is a list of integers and its range is among [0, len(parent))
    # Pre2: rank a list of integers and its range is among [0, len(parent))
    # e.g. rank[2] = 3 means the node 2 has 3 nodes as children
    # Pre3: x, y are integers among [0, len(parent)) and x != y
    # Post1: x_root becomes the parent of y_root xor y_root becomes the parent of x_root

        # Sa: find the root parent node for node x and y
        x_root = self.findParent(parent, x)
        y_root = self.findParent(parent, y)

        # Sb: compare rank[x_root] and rank[y_root] and the bigger one becomes the parent of the other one
        # XOR
        # if two ranks are equal, set y_root as the parent
        if rank[x_root] > rank[y_root]:
            parent[y_root] = x_root
            rank[x_root] += 1
        else:
            parent[x_root] = y_root
            rank[y_root] += 1

    def kruskalMST(self):
    # intent: find MST for an undirected graph using Kruskal
    # Pre1: self.graph is a list of edges and each edge is defined as [u, v, w]
    # Post1: return_result is a tree, containing (num_vertex - 1) edges
    # Post2: all vertices of self.graph are included in return_result
    # Post3: edges in return_result are sorted

        return_result = []
        parent = [i for i in range(self.v)] # Initializing parent, each vertex is the root parent of itself
        rank = [0] * self.v # Initializing rank with zeros

        i = 0  # index for each edge in self.graph
        e = 0  # index for checking how many edges invovled in result

        # Sa: all edges in self.graph is sorted based on its w
        self.graph = sorted(self.graph, key=lambda edge: edge[2])

        # Sb: there are (num_vertex - 1) edges in return_result
        # XOR
        # return_result is a forest of parts of an self.graph MST,
        # add an edge with minimal w that doesn’t create a cycle to return_result
        while e < self.v - 1:

            u, v, w = self.graph[i]
            i += 1

            u_root = self.findParent(parent, u)
            v_root = self.findParent(parent, v)

            # if there is not cycle, add the minimal w edge to return_result
            if u_root != v_root:
                e += 1
                return_result.append([u, v, w])
                self.union(parent, rank, u_root, v_root)

        # Sc: return_result is one tree with all nodes of self.graph
        return return_result


class portfolio:

    def __init__(self, L_symbol, start, end):
        self.L_symbol = L_symbol
        self.start = start
        self.end = end
        obj_ta = ta.TradeAssistant(L_symbol=L_symbol, start=start, end=end)
        L_data = obj_ta.get_raw_data()
        self.dict_raw_data = L_data[0]
        self.L_ts = L_data[1]
        self.num_symbol = len(L_symbol)

    def get_dict_rt(self):
        dict_rt = {}
        for key in self.dict_raw_data.keys():
            dict_rt[key] = self.dict_raw_data[key].pct_change()

        return dict_rt

    def get_dist_graph(self):
        df_rt = pd.DataFrame(self.get_dict_rt())
        df_corr = df_rt.corr()

        def _corr_to_dist(corr):
            return np.sqrt(2 * (1 - corr))

        df_dist = df_corr.apply(_corr_to_dist)

        return df_dist

    def get_MST(self):

        g = Graph_MST(len(self.L_ts))

        df_dist_graph = self.get_dist_graph()

        for u in range(self.num_symbol):
            for v in range(u + 1, self.num_symbol):
                g.addEdge(u, v, df_dist_graph.iloc[u, v])

        mst_result = g.kruskalMST()

        return mst_result

    def print_mst(self):
        mst = self.get_MST()
        print("The correlated stock network is listed below:")
        for u, v, dist in mst:
            print(self.L_symbol[u] + " -- " + self.L_symbol[v] + " == %.5f" % (dist))

if __name__ == "__main__":
    L_symbol = ["TSLA", "FB", "V", "JPM", "WORK", "UNH", "BABA", "MCD", "NKE"]
    port = portfolio(L_symbol=L_symbol, start=dt.datetime(2018, 1, 1), end=dt.datetime(2020, 7, 1))
    port.print_mst()
