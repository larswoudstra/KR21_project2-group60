from BNReasoner import BNReasoner
import time
import numpy as np
from scipy import stats
from scipy.stats import levene
import random
import networkx as nx
import os
import matplotlib.pyplot as plt


def CreateOutput(path):
    """
    Creates an output file for all the Baysian networks to be tested
    :param path: Takes in a path to the directory containing all the baysian network files to be tested
    :return: a .txt output file with on each line; number of nodes, random_MAP time, random_MPE time, min-degree_MAP time
    min_degree_MPE time, min_fill_MAP time and min_fill_MPE time
    """

    # create output file
    f = open("output_1.txt", "w+")

    # loop through directory
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

        # create orders
        random_order_MAP = reasoner.RandomOrder(reasoner, [pick_variables_Q[0], pick_variables_Q[1]])
        random_order_MPE = reasoner.RandomOrder(reasoner, [])
        min_degree_order_MAP = reasoner.MinDegreeOrder(reasoner, [pick_variables_Q[0], pick_variables_Q[1]])
        min_degree_order_MPE = reasoner.MinDegreeOrder(reasoner, [])
        min_fill_order_MAP = reasoner.MinFillOrder(reasoner, [pick_variables_Q[0], pick_variables_Q[1]])
        min_fill_order_MPE = reasoner.MinFillOrder(reasoner, [])

        # calculate the time per heuristic for MAP and MPE
        start = time.time()
        reasoner.MAP(pick_variables_Q, {pick_variables_e[0]: True, pick_variables_e[1]: False}, random_order_MAP)
        end = time.time()
        time_random_MAP = end - start
        start = time.time()
        reasoner.MPE({pick_variables_e[0]: True, pick_variables_e[1]: False}, random_order_MPE)
        end = time.time()
        time_random_MPE = end-start

        start = time.time()
        reasoner.MAP(pick_variables_Q, {pick_variables_e[0]: True, pick_variables_e[1]: False}, min_degree_order_MAP)
        end = time.time()
        time_mindegree_MAP = end - start
        start = time.time()
        reasoner.MPE({pick_variables_e[0]: True, pick_variables_e[1]: False}, min_degree_order_MPE)
        end = time.time()
        time_mindegree_MPE = end-start

        start = time.time()
        reasoner.MAP(pick_variables_Q, {pick_variables_e[0]: True, pick_variables_e[1]: False}, min_fill_order_MAP)
        end = time.time()
        time_minfill_MAP = end - start
        start = time.time()
        reasoner.MPE({pick_variables_e[0]: True, pick_variables_e[1]: False}, min_fill_order_MPE)
        end = time.time()
        time_minfill_MPE = end-start

        # write to output file
        f.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(node_amount, time_random_MAP, time_mindegree_MAP, time_minfill_MAP, time_random_MPE, time_mindegree_MPE, time_minfill_MPE))
    f.close()

def mean(list):
    """
    Caclulates the average of a list
    :param list: A list with all the computational running times
    :return: An average value of the computational running time
    """

    # transpose running time
    list_test = [float(x) for x in list]

    return sum(list_test) / len(list_test)

def t_test(value1, value2):
    """
    Non Parametrical Welch Two Sample T-Test
    :param value1: list of values of group1
    :param value2: list of values of group2
    :return: T-statistic with corresponding p value
    """
    return stats.ttest_ind(value1, value2, equal_var=True)

def sd(input):
    """
    Calculates the standard deviation
    :param input: list of computational running time values
    :return: standard deviation
    """
    # transpose values
    list_test = [float(x) for x in input]

    return np.std(list_test)

def test_normalized(input):
    """
    Check for the assumption of normality with the Shapiro-Wilk test
    :param input: a list with computational running time values
    :return: True if normally distributed with corresponding statistics, False if not normally distributed
    """
    list_test = [float(x) for x in input]

    if stats.shapiro(list_test)[1] < 0.05:
        return False, stats.shapiro(list_test)[1]
    else:
        return True, stats.shapiro(list_test)[1]

"""
MAIN
"""

output = r'\output_1.txt'
lines = open(output).readlines()

tm5 = []
tm10 = []
tm15 = []
tm25 = []
more25 = []

# Create lists for all different sizes with data
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

# Create lists with all the data for each group
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

# Create lists with all the data for each group
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

# Create lists with all the data for each group
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

# Create lists with all the data for each group
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

# Create lists with all the data for each group
for i in range(len(more25)):
    random_MAP_more.append(more25[i].split(" ")[1])
    min_degree_MAP_more.append(more25[i].split(" ")[2])
    min_fill_MAP_more.append(more25[i].split(" ")[3])
    random_MPE_more.append(more25[i].split(" ")[4])
    min_degree_MPE_more.append(more25[i].split(" ")[5])
    min_fill_MPE_more.append(more25[i].split(" ")[6])

"""
Normally distributed
"""

# Test the assumption of a normal distribution
print(test_normalized(random_MAP_5))
print(test_normalized(random_MAP_10))
print(test_normalized(random_MAP_15))
print(test_normalized(random_MAP_25))
print(test_normalized(random_MAP_more))
print(test_normalized(random_MPE_5))
print(test_normalized(random_MPE_10))
print(test_normalized(random_MPE_15))
print(test_normalized(random_MPE_25))
print(test_normalized(random_MPE_more))


"""
Test for equal variances
"""

# Test for the assumption of equal variances
print(levene([float(x) for x in random_MAP_5], [float(x) for x in min_degree_MAP_5], [float(x) for x in min_fill_MAP_5]))
print(levene([float(x) for x in random_MPE_5], [float(x) for x in min_degree_MPE_5], [float(x) for x in min_fill_MPE_5]))
print("\n")
print(levene([float(x) for x in random_MAP_10], [float(x) for x in min_degree_MAP_10], [float(x) for x in min_fill_MAP_10]))
print(levene([float(x) for x in random_MPE_10], [float(x) for x in min_degree_MPE_10], [float(x) for x in min_fill_MPE_10]))
print("\n")
print(levene([float(x) for x in random_MAP_15], [float(x) for x in min_degree_MAP_15], [float(x) for x in min_fill_MAP_15]))
print(levene([float(x) for x in random_MPE_15], [float(x) for x in min_degree_MPE_15], [float(x) for x in min_fill_MPE_15]))
print("\n")
print(levene([float(x) for x in random_MAP_25], [float(x) for x in min_degree_MAP_25], [float(x) for x in min_fill_MAP_25]))
print(levene([float(x) for x in random_MPE_25], [float(x) for x in min_degree_MPE_25], [float(x) for x in min_fill_MPE_25]))
print("\n")
print(levene([float(x) for x in random_MAP_more], [float(x) for x in min_degree_MAP_more], [float(x) for x in min_fill_MAP_more]))
print(levene([float(x) for x in random_MPE_more], [float(x) for x in min_degree_MPE_more], [float(x) for x in min_fill_MPE_more]))
print("\n")

"""
T-test
"""

# Compare means for all experimental conditions
print(t_test([float(x) for x in random_MAP_5], [float(x) for x in min_degree_MAP_5]))
print(t_test([float(x) for x in random_MAP_5], [float(x) for x in min_fill_MAP_5]))
print(t_test([float(x) for x in min_degree_MAP_5], [float(x) for x in min_fill_MAP_5]))
print(t_test([float(x) for x in random_MPE_5], [float(x) for x in min_degree_MPE_5]))
print(t_test([float(x) for x in random_MPE_5], [float(x) for x in min_fill_MPE_5]))
print(t_test([float(x) for x in min_degree_MPE_5], [float(x) for x in min_fill_MPE_5]))
print("\n")

print(t_test([float(x) for x in random_MAP_10], [float(x) for x in min_degree_MAP_10]))
print(t_test([float(x) for x in random_MAP_10], [float(x) for x in min_fill_MAP_10]))
print(t_test([float(x) for x in min_degree_MAP_10], [float(x) for x in min_fill_MAP_10]))
print(t_test([float(x) for x in random_MPE_10], [float(x) for x in min_degree_MPE_10]))
print(t_test([float(x) for x in random_MPE_10], [float(x) for x in min_fill_MPE_10]))
print(t_test([float(x) for x in min_degree_MPE_10], [float(x) for x in min_fill_MPE_10]))
print("\n")

print(t_test([float(x) for x in random_MAP_15], [float(x) for x in min_degree_MAP_15]))
print(t_test([float(x) for x in random_MAP_15], [float(x) for x in min_fill_MAP_15]))
print(t_test([float(x) for x in min_degree_MAP_15], [float(x) for x in min_fill_MAP_15]))
print(t_test([float(x) for x in random_MPE_15], [float(x) for x in min_degree_MPE_15]))
print(t_test([float(x) for x in random_MPE_15], [float(x) for x in min_fill_MPE_15]))
print(t_test([float(x) for x in min_degree_MPE_15], [float(x) for x in min_fill_MPE_15]))
print("\n")

print(t_test([float(x) for x in random_MAP_25], [float(x) for x in min_degree_MAP_25]))
print(t_test([float(x) for x in random_MAP_25], [float(x) for x in min_fill_MAP_25]))
print(t_test([float(x) for x in min_degree_MAP_25], [float(x) for x in min_fill_MAP_25]))
print(t_test([float(x) for x in random_MPE_25], [float(x) for x in min_degree_MPE_25]))
print(t_test([float(x) for x in random_MPE_25], [float(x) for x in min_fill_MPE_25]))
print(t_test([float(x) for x in min_degree_MPE_25], [float(x) for x in min_fill_MPE_25]))
print("\n")

print(t_test([float(x) for x in random_MAP_more], [float(x) for x in min_degree_MAP_more]))
print(t_test([float(x) for x in random_MAP_more], [float(x) for x in min_fill_MAP_more]))
print(t_test([float(x) for x in min_degree_MAP_more], [float(x) for x in min_fill_MAP_more]))
print(t_test([float(x) for x in random_MPE_more], [float(x) for x in min_degree_MPE_more]))
print(t_test([float(x) for x in random_MPE_more], [float(x) for x in min_fill_MPE_more]))
print(t_test([float(x) for x in min_degree_MPE_more], [float(x) for x in min_fill_MPE_more]))
print("\n")

"""
Metrics
"""
# Calculate the minimum, maximum, average and standard deviation for all experimental conditions
print(min(random_MAP_more))
print(min(random_MPE_more))
print(max(random_MAP_more))
print(max(random_MPE_more))
print(mean(random_MAP_more))
print(mean(random_MPE_more))
print(sd(random_MAP_more))
print(sd(random_MPE_more))

print(min(min_degree_MAP_more))
print(min(min_degree_MPE_more))
print(max(min_degree_MAP_more))
print(max(min_degree_MPE_more))
print(mean(min_degree_MAP_more))
print(mean(min_degree_MPE_more))
print(sd(min_degree_MAP_more))
print(sd(min_degree_MPE_more))

print(min(min_fill_MAP_25))
print(min(min_fill_MPE_25))
print(max(min_fill_MAP_25))
print(max(min_fill_MPE_25))
print(mean(min_fill_MAP_25))
print(mean(min_fill_MPE_25))
print(sd(min_fill_MAP_25))
print(sd(min_fill_MPE_25))

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

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize=(8, 4))

# set height of bar
random = [random_MPE_5, random_MPE_10, random_MPE_15, random_MPE_25, random_MPE_more]
mindegree = [min_degree_MPE_5, min_degree_MPE_10, min_degree_MPE_15, min_degree_MPE_25, min_degree_MPE_more]
minfill = [min_fill_MPE_5, min_fill_MPE_10, min_fill_MPE_15, min_fill_MPE_25, min_fill_MPE_more]

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
plt.title('Mean run time MPE')
plt.xlabel('Number of Nodes', fontsize=10)
plt.ylabel('Run Time (seconds)', fontsize=10)
plt.xticks([r + barWidth for r in range(5)],
           ['0 - 5', '5 - 10', '10 - 15', '15 - 25', '25 +'])

plt.legend(loc = "upper left")
plt.yscale('log')
#plt.ylim(10**-2, 10**6)
plt.style.use('grayscale')
plt.show()





