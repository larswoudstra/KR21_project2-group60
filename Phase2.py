from BNReasoner import BNReasoner
import os
import time
import numpy as np
from scipy import stats
from scipy.stats import levene

import matplotlib.pyplot as plt
import random
import pathlib
import networkx as nx

from typing import Union
from BayesNet import BayesNet
import BNReasoner
import os
import random

from typing import List, Tuple, Dict
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.readwrite import XMLBIFReader
import math
import itertools
import pandas as pd
from copy import deepcopy

def CreateOutput(path):

    # create output file
    f = open("output_1.txt", "w+")

    for filename in os.listdir(path):

        # open baysian network
        open_file = os.path.join(path, filename)
        GRAPH = nx.read_gpickle(open_file)
        reasoner = BNReasoner.BNReasoner(net=GRAPH)

        # pick random variables for Q and for e
        variables = reasoner.bn.get_all_variables()
        node_amount = len(variables)
        pick_variables_e = random.sample(variables, 2)
        variables.remove(pick_variables_e[0])
        variables.remove(pick_variables_e[1])
        pick_variables_Q = random.sample(variables, 2)

        # create order
        random_order_MAP = reasoner.RandomOrder(reasoner, [pick_variables_Q[0], pick_variables_Q[1]])
        random_order_MPE = reasoner.RandomOrder(reasoner, [])
        min_degree_order_MAP = reasoner.MinDegreeOrder(reasoner, [pick_variables_Q[0], pick_variables_Q[1]])
        min_degree_order_MPE = reasoner.MinDegreeOrder(reasoner, [])
        min_fill_order_MAP = reasoner.MinFillOrder(reasoner, [pick_variables_Q[0], pick_variables_Q[1]])
        min_fill_order_MPE = reasoner.MinFillOrder(reasoner, [])
        print('orders zijn created')

        # calculate the time per heuristic for MAP and MPE
        start = time.time()
        reasoner.MAP(pick_variables_Q, {pick_variables_e[0]: True, pick_variables_e[1]: False}, random_order_MAP)
        end = time.time()
        time_random_MAP = end - start
        start = time.time()
        reasoner.MPE({pick_variables_e[0]: True, pick_variables_e[1]: False}, random_order_MPE)
        end = time.time()
        time_random_MPE = end-start
        print('random is klaar')

        start = time.time()
        reasoner.MAP(pick_variables_Q, {pick_variables_e[0]: True, pick_variables_e[1]: False}, min_degree_order_MAP)
        end = time.time()
        time_mindegree_MAP = end - start
        start = time.time()
        reasoner.MPE({pick_variables_e[0]: True, pick_variables_e[1]: False}, min_degree_order_MPE)
        end = time.time()
        time_mindegree_MPE = end-start
        print('mindegree is klaar')

        start = time.time()
        reasoner.MAP(pick_variables_Q, {pick_variables_e[0]: True, pick_variables_e[1]: False}, min_fill_order_MAP)
        end = time.time()
        time_minfill_MAP = end - start
        start = time.time()
        reasoner.MPE({pick_variables_e[0]: True, pick_variables_e[1]: False}, min_fill_order_MPE)
        end = time.time()
        time_minfill_MPE = end-start
        print('minfill is klaar')

        print(f' node: {node_amount} randomMAP {time_random_MAP} timedegreeMAP {time_mindegree_MAP} timefillMAP {time_minfill_MAP} randomMPE {time_random_MPE} timedegreeMPE {time_mindegree_MPE} timefillMPE {time_minfill_MPE}')
        # write to output file
        f.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(node_amount, time_random_MAP, time_mindegree_MAP, time_minfill_MAP, time_random_MPE, time_mindegree_MPE, time_minfill_MPE))
    f.close()

def CreateGraph(output):

    # abstract nodes
    nodes = []
    [nodes.append(x.split(' ')[0]) for x in open(output).readlines()]

    # abstract random order
    random = []
    [random.append(x.split(' ')[1]) for x in open(output).readlines()]

    # abstract MinDegree order
    min_degree = []
    [min_degree.append(x.split(' ')[2]) for x in open(output).readlines()]

    # abstract MinFill
    min_fill = []
    [min_fill.append(x.split(' ')[3]) for x in open("output.txt").readlines()]

    # plot data
    plt.plot(nodes, random, label = "Random Order")
    plt.plot(nodes, min_degree, label = "MinDegree Order")
    plt.plot(nodes, min_fill, label = "MinFill Order")
    plt.legend()
    plt.xlabel('Number of Nodes of Bayseian Network')
    plt.ylabel('Running time [seconds]')

    plt.show()

def mean(list):
    list_test = [float(x) for x in list]

    return sum(list_test) / len(list_test)

def t_test(value1, value2):
    return stats.ttest_ind(value1, value2, equal_var=True)

"""
MAIN
"""

output = r'C:\Users\Stans\KR\KR21_project2\output_1.txt'
lines = open(output).readlines()

tm5 = []
tm10 = []
tm15 = []
tm25 = []
more25 = []

for line in lines:
    if int(line.split(' ')[0]) <= 5:
        tm5.append(line)
    elif int(line.split(' ')[0]) > 5 and int(line.split(' ')[0]) <= 10:
        tm10.append(line)
    elif int(line.split(' ')[0]) > 10 and int(line.split(' ')[0]) <= 15:
        tm15.append(line)
    elif int(line.split(' ')[0]) > 15 and int(line.split(' ')[0]) <= 25:
        tm25.append(line)
    elif int(line.split(' ')[0]) > 25:
        more25.append(line)

"""
TM 5
"""

random_MAP_5= []
min_degree_MAP_5 = []
min_fill_MAP_5 = []
random_MPE_5= []
min_degree_MPE_5 = []
min_fill_MPE_5 = []


for i in range(len(tm5)):
    random_MAP_5.append(tm5[i].split(" ")[1])
    min_degree_MAP_5.append(tm5[i].split(" ")[2])
    min_fill_MAP_5.append(tm5[i].split(" ")[3])
    random_MPE_5.append(tm5[i].split(" ")[4])
    min_degree_MPE_5.append(tm5[i].split(" ")[5])
    min_fill_MPE_5.append(tm5[i].split(" ")[6])

"""
TM 10
"""

random_MAP_10= []
min_degree_MAP_10 = []
min_fill_MAP_10 = []
random_MPE_10= []
min_degree_MPE_10 = []
min_fill_MPE_10 = []

for i in range(len(tm10)):
    random_MAP_10.append(tm10[i].split(" ")[1])
    min_degree_MAP_10.append(tm10[i].split(" ")[2])
    min_fill_MAP_10.append(tm10[i].split(" ")[3])
    random_MPE_10.append(tm10[i].split(" ")[4])
    min_degree_MPE_10.append(tm10[i].split(" ")[5])
    min_fill_MPE_10.append(tm10[i].split(" ")[6])

"""
TM 15
"""

random_MAP_15= []
min_degree_MAP_15 = []
min_fill_MAP_15 = []
random_MPE_15= []
min_degree_MPE_15 = []
min_fill_MPE_15 = []

for i in range(len(tm15)):
    random_MAP_15.append(tm15[i].split(" ")[1])
    min_degree_MAP_15.append(tm15[i].split(" ")[2])
    min_fill_MAP_15.append(tm15[i].split(" ")[3])
    random_MPE_15.append(tm15[i].split(" ")[4])
    min_degree_MPE_15.append(tm15[i].split(" ")[5])
    min_fill_MPE_15.append(tm15[i].split(" ")[6])

"""
TM 25
"""
random_MAP_25= []
min_degree_MAP_25 = []
min_fill_MAP_25 = []
random_MPE_25= []
min_degree_MPE_25 = []
min_fill_MPE_25 = []

for i in range(len(tm25)):
    random_MAP_25.append(tm25[i].split(" ")[1])
    min_degree_MAP_25.append(tm25[i].split(" ")[2])
    min_fill_MAP_25.append(tm25[i].split(" ")[3])
    random_MPE_25.append(tm25[i].split(" ")[4])
    min_degree_MPE_25.append(tm25[i].split(" ")[5])
    min_fill_MPE_25.append(tm25[i].split(" ")[6])

"""
25 +
"""
random_MAP_more= []
min_degree_MAP_more = []
min_fill_MAP_more = []
random_MPE_more= []
min_degree_MPE_more = []
min_fill_MPE_more = []

for i in range(len(more25)):
    random_MAP_more.append(more25[i].split(" ")[1])
    min_degree_MAP_more.append(more25[i].split(" ")[2])
    min_fill_MAP_more.append(more25[i].split(" ")[3])
    random_MPE_more.append(more25[i].split(" ")[4])
    min_degree_MPE_more.append(more25[i].split(" ")[5])
    min_fill_MPE_more.append(more25[i].split(" ")[6])

"""
Test for equal variances
"""
# print(random_MAP_5)
# print(levene([float(x) for x in random_MAP_5], [float(x) for x in min_degree_MAP_5], [float(x) for x in min_fill_MAP_5]))
# print(levene([float(x) for x in random_MPE_5], [float(x) for x in min_degree_MPE_5], [float(x) for x in min_fill_MPE_5]))
# print("\n")
# print(levene([float(x) for x in random_MAP_10], [float(x) for x in min_degree_MAP_10], [float(x) for x in min_fill_MAP_10]))
# print(levene([float(x) for x in random_MPE_10], [float(x) for x in min_degree_MPE_10], [float(x) for x in min_fill_MPE_10]))
# print("\n")
# print(levene([float(x) for x in random_MAP_15], [float(x) for x in min_degree_MAP_15], [float(x) for x in min_fill_MAP_15]))
# print(levene([float(x) for x in random_MPE_15], [float(x) for x in min_degree_MPE_15], [float(x) for x in min_fill_MPE_15]))
# print("\n")
# print(levene([float(x) for x in random_MAP_25], [float(x) for x in min_degree_MAP_25], [float(x) for x in min_fill_MAP_25]))
# print(levene([float(x) for x in random_MPE_25], [float(x) for x in min_degree_MPE_25], [float(x) for x in min_fill_MPE_25]))
# print("\n")
# print(levene([float(x) for x in random_MAP_more], [float(x) for x in min_degree_MAP_more], [float(x) for x in min_fill_MAP_more]))
# print(levene([float(x) for x in random_MPE_more], [float(x) for x in min_degree_MPE_more], [float(x) for x in min_fill_MPE_more]))
# print("\n")

"""
T-test
"""
# print(t_test([float(x) for x in random_MAP_5], [float(x) for x in min_degree_MAP_5]))
# print(t_test([float(x) for x in random_MAP_5], [float(x) for x in min_fill_MAP_5]))
# print(t_test([float(x) for x in min_degree_MAP_5], [float(x) for x in min_fill_MAP_5]))
# print(t_test([float(x) for x in random_MPE_5], [float(x) for x in min_degree_MPE_5]))
# print(t_test([float(x) for x in random_MPE_5], [float(x) for x in min_fill_MPE_5]))
# print(t_test([float(x) for x in min_degree_MPE_5], [float(x) for x in min_fill_MPE_5]))
# print("\n")
#
# print(t_test([float(x) for x in random_MAP_10], [float(x) for x in min_degree_MAP_10]))
# print(t_test([float(x) for x in random_MAP_10], [float(x) for x in min_fill_MAP_10]))
# print(t_test([float(x) for x in min_degree_MAP_10], [float(x) for x in min_fill_MAP_10]))
# print(t_test([float(x) for x in random_MPE_10], [float(x) for x in min_degree_MPE_10]))
# print(t_test([float(x) for x in random_MPE_10], [float(x) for x in min_fill_MPE_10]))
# print(t_test([float(x) for x in min_degree_MPE_10], [float(x) for x in min_fill_MPE_10]))
# print("\n")
#
# print(t_test([float(x) for x in random_MAP_15], [float(x) for x in min_degree_MAP_15]))
# print(t_test([float(x) for x in random_MAP_15], [float(x) for x in min_fill_MAP_15]))
# print(t_test([float(x) for x in min_degree_MAP_15], [float(x) for x in min_fill_MAP_15]))
# print(t_test([float(x) for x in random_MPE_15], [float(x) for x in min_degree_MPE_15]))
# print(t_test([float(x) for x in random_MPE_15], [float(x) for x in min_fill_MPE_15]))
# print(t_test([float(x) for x in min_degree_MPE_15], [float(x) for x in min_fill_MPE_15]))
# print("\n")
#
# print(t_test([float(x) for x in random_MAP_25], [float(x) for x in min_degree_MAP_25]))
# print(t_test([float(x) for x in random_MAP_25], [float(x) for x in min_fill_MAP_25]))
# print(t_test([float(x) for x in min_degree_MAP_25], [float(x) for x in min_fill_MAP_25]))
# print(t_test([float(x) for x in random_MPE_25], [float(x) for x in min_degree_MPE_25]))
# print(t_test([float(x) for x in random_MPE_25], [float(x) for x in min_fill_MPE_25]))
# print(t_test([float(x) for x in min_degree_MPE_25], [float(x) for x in min_fill_MPE_25]))
# print("\n")
#
# print(t_test([float(x) for x in random_MAP_more], [float(x) for x in min_degree_MAP_more]))
# print(t_test([float(x) for x in random_MAP_more], [float(x) for x in min_fill_MAP_more]))
# print(t_test([float(x) for x in min_degree_MAP_more], [float(x) for x in min_fill_MAP_more]))
# print(t_test([float(x) for x in random_MPE_more], [float(x) for x in min_degree_MPE_more]))
# print(t_test([float(x) for x in random_MPE_more], [float(x) for x in min_fill_MPE_more]))
# print(t_test([float(x) for x in min_degree_MPE_more], [float(x) for x in min_fill_MPE_more]))
# print("\n")

"""
AVERAGE TIME
"""
random_MAP_5= mean(random_MAP_5)
min_degree_MAP_5 = mean(min_degree_MAP_5)
min_fill_MAP_5 = mean(min_fill_MAP_5)
random_MPE_5= mean(random_MPE_5)
min_degree_MPE_5 = mean(min_degree_MPE_5)
min_fill_MPE_5 = mean(min_fill_MPE_5)

random_MAP_10= mean(random_MAP_10)
min_degree_MAP_10 = mean(min_degree_MAP_10)
min_fill_MAP_10 = mean(min_fill_MAP_10)
random_MPE_10= mean(random_MPE_10)
min_degree_MPE_10 = mean(min_degree_MPE_10)
min_fill_MPE_10 = mean(min_fill_MPE_10)

random_MAP_15= mean(random_MAP_15)
min_degree_MAP_15 = mean(min_degree_MAP_15)
min_fill_MAP_15 = mean(min_fill_MAP_15)
random_MPE_15= mean(random_MPE_15)
min_degree_MPE_15 = mean(min_degree_MPE_15)
min_fill_MPE_15 = mean(min_fill_MPE_15)

random_MAP_25= mean(random_MAP_25)
min_degree_MAP_25 = mean(min_degree_MAP_25)
min_fill_MAP_25 = mean(min_fill_MAP_25)
random_MPE_25= mean(random_MPE_25)
min_degree_MPE_25 = mean(min_degree_MPE_25)
min_fill_MPE_25 = mean(min_fill_MPE_25)

random_MAP_more = mean(random_MAP_more)
min_degree_MAP_more = mean(min_degree_MAP_more)
min_fill_MAP_more = mean(min_fill_MAP_more)
random_MPE_more= mean(random_MPE_more)
min_degree_MPE_more = mean(min_degree_MPE_more)
min_fill_MPE_more = mean(min_fill_MPE_more)

"""
BARPLOTS
"""

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize=(8, 4))

# set height of bar
random = [random_MAP_5, random_MAP_10, random_MAP_15, random_MAP_25, random_MAP_more]
mindegree = [min_degree_MAP_5, min_degree_MAP_10, min_degree_MAP_15, min_degree_MAP_25, min_degree_MAP_more]
minfill = [min_fill_MAP_5, min_fill_MAP_10, min_fill_MAP_15, min_fill_MAP_25, min_fill_MAP_more]


# Set position of bar on X axis
br1 = np.arange(len(random))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, random, color='lightgray', width=barWidth,
        edgecolor='grey', label='Random')
plt.bar(br2, mindegree, color='darkgray', width=barWidth,
        edgecolor='grey', label='MinDegree')
plt.bar(br3, minfill, color='gray', width=barWidth,
        edgecolor='grey', label='MinFill')


# Adding Xticks
plt.title('Mean run time MAP')
plt.xlabel('Number of Nodes', fontsize=10)
plt.ylabel('Run Time (seconds)', fontsize=10)
plt.xticks([r + barWidth for r in range(5)],
           ['0 - 5', '5 - 10', '10 - 15', '15 - 25', '25 +'])

plt.legend(loc = "upper left")
plt.yscale('log')
plt.ylim(10**-2, 10**3)
plt.style.use('grayscale')
plt.show()

"""
MPE
"""
#
# # set width of bar
# barWidth = 0.25
# fig = plt.subplots(figsize=(8, 4))
#
# # set height of bar
# random = [random_MPE_5, random_MPE_10, random_MPE_15, random_MPE_25, random_MPE_more]
# mindegree = [min_degree_MPE_5, min_degree_MPE_10, min_degree_MPE_15, min_degree_MPE_25, min_degree_MPE_more]
# minfill = [min_fill_MPE_5, min_fill_MPE_10, min_fill_MPE_15, min_fill_MPE_25, min_fill_MPE_more]
#
#
# # Set position of bar on X axis
# br1 = np.arange(len(random))
# br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
#
# # Make the plot
# plt.bar(br1, random, color='lightgray', width=barWidth,
#         edgecolor='grey', label='Random')
# plt.bar(br2, mindegree, color='darkgray', width=barWidth,
#         edgecolor='grey', label='MinDegree')
# plt.bar(br3, minfill, color='gray', width=barWidth,
#         edgecolor='grey', label='MinFill')
#
#
# # Adding Xticks
# plt.title('Mean run time MPE')
# plt.xlabel('Number of Nodes', fontsize=10)
# plt.ylabel('Run Time (seconds)', fontsize=10)
# plt.xticks([r + barWidth for r in range(5)],
#            ['0 - 5', '5 - 10', '10 - 15', '15 - 25', '25 +'])
#
# plt.legend(loc = "upper left")
# plt.yscale('log')
# #plt.ylim(10**-2, 10**6)
# plt.style.use('grayscale')
# plt.show()
