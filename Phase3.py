from BNReasoner import BNReasoner
import time
import matplotlib.pyplot as plt
import random
import pathlib
import networkx as nx

from typing import Union
from BayesNet import BayesNet
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

# display graph
reasoner = BNReasoner('use_case.BIFXML')
reasoner.bn.draw_structure()
variables = reasoner.bn.get_all_variables()
print(variables)
node_amount = len(variables)

# create a prior for stressed
Min_Fill_Order_Prior = reasoner.MinFillOrder(reasoner, ['Stressed'])
prior_stressed = reasoner.marginal_dist(['Stressed'], {}, Min_Fill_Order_Prior)
print('Prior Stressed: ')
display(prior_stressed)

# create a prior for pandemic
reasoner = BNReasoner('use_case.BIFXML')
Min_Fill_Order_Prior = reasoner.MinFillOrder(reasoner, ['Pandemic'])
prior_pandemic = reasoner.marginal_dist(['Pandemic'], {}, Min_Fill_Order_MAP)
print('Prior Pandemic: ')
display(prior_pandemic)

# create posterior distribution Pr(Stressed | Pandemic = True)
reasoner = BNReasoner('use_case.BIFXML')
Min_Fill_Order_Prior = reasoner.MinFillOrder(reasoner, ['Stressed'])
posterior_stressed_true = reasoner.marginal_dist(['Stressed'], {'Pandemic': True}, Min_Fill_Order_Prior)
print('Marginal distribution for stressed given pandemic = True: ')
display(posterior_stressed_true)

# create posterior distribution Pr(Stressed | Pandemic = False)
reasoner = BNReasoner('use_case.BIFXML')
Min_Fill_Order_Prior = reasoner.MinFillOrder(reasoner, ['Stressed'])
posterior_stressed_false = reasoner.marginal_dist(['Stressed'], {'Pandemic': False}, Min_Fill_Order_Prior)
print('Marginal distribution for stressed given pandemic = False: ')
display(posterior_stressed_false)

# create posterior distribution Pr(Busy | Pandemic = True)
reasoner = BNReasoner('use_case.BIFXML')
Min_Fill_Order_Prior = reasoner.MinFillOrder(reasoner, ['Busy'])
posterior_busy = reasoner.marginal_dist(['Busy'], {'Pandemic': True}, Min_Fill_Order_Prior)
print('Marginal distribution for busy given pandemic = True: ')
display(posterior_busy)

# create MAP Pr(Stressed | Pandemic = True)
reasoner = BNReasoner('use_case.BIFXML')
Min_Fill_Order_MAP = reasoner.MinFillOrder(reasoner, ['Stressed'])
MAP = reasoner.MAP(['Stressed'], {'Pandemic': True}, Min_Fill_Order_MAP)
print('Most likely instance for stressed when Pandemic = True ')
display(MAP)

# create MPE Pr(Stressed = True, Pandemic = True)
reasoner = BNReasoner('use_case.BIFXML')
Min_Fill_Order_MPE = reasoner.MinFillOrder(reasoner, [])
MPE = reasoner.MPE({'Stressed': True, 'Pandemic': True}, Min_Fill_Order_MPE)
print('Most likely instantiations when Stressed = True and Pandemic = True')
display(MPE)
