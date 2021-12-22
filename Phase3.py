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


reasoner = BNReasoner(r'use_case.BIFXML')

variables = reasoner.bn.get_all_variables()
print(variables)
node_amount = len(variables)

# create order
Min_Fill_Order_Prior = reasoner.MinFillOrder(reasoner, ['Stressed'])
Min_Fill_Order_MAP = reasoner.MinFillOrder(reasoner, ['Pandemic'])
Min_Fill_Order_MPE = reasoner.MinFillOrder(reasoner, [])

# Calculate instances
prior_stressed = reasoner.marginal_dist(['Stressed'], {}, Min_Fill_Order_Prior)
prior_pandemic = reasoner.marginal_dist(['Pandemic'], {}, Min_Fill_Order_MAP)
posterior = reasoner.marginal_dist(['Stressed'], {'Pandemic': True}, Min_Fill_Order_Prior)
MAP = reasoner.MAP(['Pandemic'], {'Stressed': True}, Min_Fill_Order_MAP)
MPE = reasoner.MPE({'Stressed': True, 'Pandemic': True}, Min_Fill_Order_MPE)

# Print
print(f'Prior Stressed: {prior_stressed}')
print(f'Prior Pandemic: {prior_pandemic}')
print(f'Posterior: \n {posterior}')
print(f'MAP: \n {MAP}')
print(f'MPE: \n {MPE}')



















