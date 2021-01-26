"""
Implemented algorithm
Dijkstra shortest path tree for finding the minimal cost path for trading a set of currencies
This project can be used to develop trading strategy in the currency market
author: Yi Rong
updated on 1/25/21
References: https://www.muzeumkremnica.sk/_img/Documents/_PUBLIK_NBS_FSR/Biatec/Rok2013/07-2013/05_biatec13-7_resovsky_EN.pdf
"""

import datetime as dt
import math
import pdb
import sys
import os

sys.path.append(os.path.abspath("TradeAssistant.py"))
import TradeAssistant as ta


class Graph_SPT():

    def __init__(self, num_vertex):
    # initialize the vertex number and the graph
        self.v = num_vertex
        self.graph = [[0] * num_vertex for _ in range(num_vertex)]

    def minDist(self, dist, sptSet):
    # intent: search and add the shortest distance edge out of SPT
    # Pre1: dist is a list of |V| non-negative numbers
    # e.g. dist[1] = 2 means the shortest distance from source to node 1 is 2
    # Pre2: sptSet is a list of |V| booleans
    # e.g. sptSet[1] = False means node 1 is not in SPT
    # Post1: min_index is the node has the shortest distance to SPT
    # Post2: min_index is added to SPT


        min_index = -1 # Initializing node min_index
        min_ = float("Inf") # Initializing the shortest distance

        # Sa: compare all the outside nodes' distances to SPT and add the shortest one
        for v in range(self.v):
            if dist[v] < min_ and sptSet[v] == False:
                min_ = dist[v]
                min_index = v

        return min_index

    def DijkstraSPT(self, src):
    # intent: find SPT for a directed self.graph from source src using Dijkstra
    # Pre1: src is an integer among [0, |V|)
    # Pre2: self.graph is a matrix and each element is a non-negative number
    # e.g. self.graph[2][3] = 1 means the distance from node 2 to 3 is 1
    # Post1: return_result is dist, which contains the shortest distance from src to each node
    # Post2: all nodes's distances are included in return_result

        # Sa: The distance of each node in dist = cost of cheapest path to it from src
        dist = [float("Inf")] * self.v # Initialize dist with infinite distance
        sptSet = [False] * self.v # Initialize sptSet with False
        dist[src] = 0 # Initialize dist at src node with 0

        for _ in range(self.v):

            # Sb: node u has the shortest distance to SPT
            u = self.minDist(dist, sptSet)

            sptSet[u] = True

            # Sc: update dist[v] if any node v can minimize dist[v] with (dist[u] + weight(u, v))
            for v in range(self.v):
                # pdb.set_trace()
                if self.graph[u][v] != 0 and sptSet[v] == False and (dist[u] + self.graph[u][v] < dist[v]):
                    dist[v] = dist[u] + self.graph[u][v]

        # Sd: return_result contains all nodes reachable from src
        return_result = dist
        return return_result


class currency:

    def __init__(self, L_symbol, date, source):
        self.src = source
        self.L_symbol = L_symbol
        self.date = date
        self.num_symbol = len(L_symbol)

        self.L_fx_symbol = []
        for i in L_symbol:
            for j in L_symbol:
                if i != j:
                    self.L_fx_symbol.append(i + j + "=X")

        obj_ta = ta.TradeAssistant(L_symbol=self.L_fx_symbol, start=date, end=date)
        L_data = obj_ta.get_raw_data()
        self.dict_raw_data = L_data[0]

    def get_SPT(self):

        g = Graph_SPT(self.num_symbol)

        for i, cur1 in enumerate(self.L_symbol):
            for j, cur2 in enumerate(self.L_symbol):
                if i != j:
                    g.graph[i][j] = 1 - math.log10(self.dict_raw_data[cur1 + cur2 + "=X"].values[-1])
        #pdb.set_trace()
        spt_result = g.DijkstraSPT(self.src)

        return spt_result

    def print_spt(self):
        spt = self.get_SPT()
        print("The minimal cost path for " + self.L_symbol[self.src] + " are listed: ")
        print("To:  Cost")
        for i in range(self.num_symbol):
            print(self.L_symbol[i], "{:.5f}".format(spt[i]))


if __name__ == "__main__":
    L_symbol = ["JPY", "AUD", "CAD", "GBP", "USD", "HKD", "CHF"]
    cur = currency(L_symbol=L_symbol, date=dt.datetime(2020, 7, 30), source=0)
    cur.print_spt()
