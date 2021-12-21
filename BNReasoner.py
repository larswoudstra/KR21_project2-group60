from typing import Union
from BayesNet import BayesNet
import copy
from itertools import product

class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    @staticmethod
    def adjency_list(NXgraph) -> dict:
        '''
        :Return: dictionary with connection for all variables
        '''
        nodes = list(NXgraph.nodes)

        adj_list_1 = dict()
        adj_list_2 = dict()
        for n in nodes:
            adj_list_1[n] = list(NXgraph.neighbors(n))
            for l in adj_list_1.values():
                for elem in l:
                    adj_list_2[n] = list(NXgraph.neighbors(elem))

        adj_list = {key: adj_list_1[key] + adj_list_2[key] for key in adj_list_2}
        for key, value  in adj_list.items():
            print(f'Variabe {key} is dependent on {value}')
        return adj_list

    def dsep(self, X:list, Y:list, Z:list) -> bool:
        '''
        Variables in X are independent from Y given Z
        :Param X: list of variables
        :Param Y: list of variables
        :Param Z: list of variables
        :Return: Either True d-speration or False not d-speration
        '''
        test = copy.deepcopy(self.bn)
        test = test.structure.to_undirected()
        # Deleting Z's children
        for i in Z:
            for j in list(test.neighbors(i)):
                test.remove_edge(i,j)

        # Base Case
        if X == Y:
            return False

        #checking d-speration
        adj_dict = BNReasoner.adjency_list(test)
        for variable,adjency_list in adj_dict.items():
            for var_x, var_y in zip(X,Y):

                if var_x == variable:
                    if var_y in adjency_list:
                        return print(f'{False} They are not dsep')
                    else:
                        return print(f'{True} They are dsep')

    def sumOutVars(self, cpt: pd.core.frame.DataFrame, Z:list) -> pd.core.frame.DataFrame:
        '''
        Removes every variable in Z from given cpt. Returns the newly formed cpt.
        '''
        new_cpt = cpt

        # go over every variable in Z
        for variable in Z:

            # check if value is in the given cpt and the cpt is not only consisting of the variable itself
            if variable in new_cpt and len(new_cpt.columns) != 1:

                # remove variable from the cpt by dropping both true and false instances
                false_cpt = new_cpt[(new_cpt[variable] == False)].drop(variable, axis=1)
                true_cpt = new_cpt[(new_cpt[variable] == True)].drop(variable, axis=1)

                # get remaining variables and sum their probability
                Y = [col for col in true_cpt.columns if col != 'p']
                new_cpt = pd.concat([false_cpt, true_cpt]).groupby(Y)['p'].sum().reset_index()

        return new_cpt

    def MultiplyFactors(self, X: list) -> pd.core.frame.DataFrame:
        '''
        X is a list of cpts that you want to multiply.
        Returns a factor of multiplied cpts.
        '''
        # factor is starting cpt
        factor = X[0]

        # multiply this starting cpt with all other cpts in the list
        for index in range(1, len(X)):
            x = X[index]

            # only multiply when there are matching variables
            column_x = [col for col in x.columns if col != 'p']
            column_factor = [col for col in factor.columns if col != 'p']
            match = list(set(column_x) & set(column_factor))

            if len(match) != 0:
                df_mul = pd.merge(x, factor, how='left', on=match)
                df_mul['p'] = (df_mul['p_x'] * df_mul['p_y'])
                df_mul.drop(['p_x', 'p_y'],inplace=True, axis = 1)

                factor = df_mul

        return factor

    def network_pruning(self, Q: list, E: dict) -> None:
        '''
        Node- and edge-prune the Bayesian network.
        :param Q: query variable.
        :param E: evidence variable with its truth value.
        '''

        # edge pruning
        for var, truth_val in zip(E.keys(), E.values()):

            cpt = self.bn.get_cpt(var)
            cpt_update = self.bn.get_compatible_instantiations_table(pd.Series({var: truth_val}), cpt)
            self.bn.update_cpt(var, cpt_update)

            # check whether node in evidence has children of which we can prune edges
            if self.bn.get_children(var) == []:
                pass
            else:
                for child in self.bn.get_children(var):

                    # prune edge between evident node and its child
                    self.bn.del_edge((var, child))

                    # update CPT
                    cpt = self.bn.get_cpt(child)
                    cpt_update = self.bn.get_compatible_instantiations_table(pd.Series({var: truth_val}), cpt)
                    self.bn.update_cpt(child, cpt_update)

        # leaf node pruning
        stop_pruning = False

        while not stop_pruning:

            stop_pruning = True

            for variable in self.bn.get_all_variables():

                # leaf node should not be directly influencing Q or E (i.e. has no children)
                if self.bn.get_children(variable) == []:

                    # leaf node should not be in Q or E
                    if variable not in set(Q) and variable not in set(E.keys()):

                        # delete leaf node and make sure to run over all vars again to detect new leaf nodes
                        self.bn.del_var(variable)
                        stop_pruning = False


    def RandomOrder(self, BN, Q):
        """
        Returns a list with a random order of variables from all the variables
        in the BN, except the ones in the query (Q).
        """

        interaction = BN.bn.get_interaction_graph()
        degree = dict((interaction.degree()))

        # delete all variables off Q
        if Q:
            for variable in Q:
                del degree[variable]

        random_order = random.sample(list(degree), len(list(degree)))

        return random_order

    def MinDegreeOrder(self, BN, Q):
        """
        Returns an ordered list based upon the variable with the least amount
        of neighbors in the BN
        """

        # create interaction graph
        interaction = BN.bn.get_interaction_graph()
        degree = dict((interaction.degree()))
        order = []

        # delete all variables of Q
        if Q:
          for variable in Q:
              del degree[variable]
        degree = list(degree.items())

        for i in range(len(degree)):

            # check smallest width
            node = self.smallest_degree(degree)
            self.connect_neighbors(interaction, node)

            # delete edges from node
            interaction.remove_node(list(node)[0])
            order.append((list(node.keys())[0]))

            degree.remove(list(node.items())[0])

        return order

    def MinFillOrder(self, BN, Q):
        """
        Returns an ordered list based upon the variable that causes the least
        to be filled in edges when deleted
        """

        # create interaction graph
        interaction = BN.bn.get_interaction_graph()
        degree = dict((interaction.degree()))
        order = []

        # delete all variables of Q
        if Q:
          for variable in Q:
              del degree[variable]
        degree = list(degree.items())

        for i in range(len(degree)):

            # check node whose eliminations adds smallest number of edges
            node = self.smallest_edges(interaction, degree)
            self.connect_neighbors(interaction, node)

            interaction.remove_node(list(node)[0])
            order.append((list(node.keys())[0]))

            degree = dict(degree)
            del degree[str(list(node.keys())[0])]
            degree = list(degree.items())

        return order

    def smallest_edges(self, interaction, degree):

        smallest_edges = {}

        for i in range(len(degree)):
            if i == 0:
                smallest_edges[degree[i][0]] = self.compute_edges(interaction, degree[i][0])

            elif self.compute_edges(interaction, degree[i][0]) < smallest_edges.get(list(smallest_edges.keys())[0]):
                smallest_edges = {}
                smallest_edges[degree[i][0]] = self.compute_edges(interaction, degree[i][0])

        return smallest_edges

    def compute_edges(self, interaction, node):

        edges = 0

        neighbors = list(interaction.neighbors(node))

        for i in range(len(neighbors)):
            neighbors_i = list(interaction.neighbors(neighbors[i]))
            if i+1 == len(neighbors):
                break
            for j in range(i+1, len(neighbors)):
                if neighbors[j] not in neighbors_i:
                    edges += 1

        return edges


    def smallest_degree(self, degree):
        minimum = {}

        for i in range(len(degree)):
            if i == 0:
                minimum[degree[i][0]] = degree[i][1]
            elif degree[i][1] < minimum.get(list(minimum.keys())[0]):
                minimum = {}
                minimum[degree[i][0]] = degree[i][1]

        return minimum

    def connect_neighbors(self, interaction, node):

        neighbors = list(interaction.neighbors(list(node)[0]))

        for i in range(len(neighbors)):
            neighbors_i = list(interaction.neighbors(neighbors[i]))
            if i+1 == len(neighbors):
                break
            for j in range(i+1, len(neighbors)):
                if neighbors[j] not in neighbors_i:
                    interaction.add_edge(neighbors[i], neighbors[j])

    def maximizeOut(self, cpt: pd.DataFrame, var: str):
        '''
        Maximize out over a variable for a given conditional probability table; rows with max values are returned.
        :param variable: var to maximize out on.
        :param cpt: conditional probability table.
        '''
        new_cpt = cpt

        # check if value is in the given cpt and the cpt is not only consisting of the variable itself
        if var in new_cpt and len(new_cpt.columns) != 1:

            # remove variable from the cpt by dropping both true and false instances
            false_cpt = new_cpt[(new_cpt[var] == False)].drop(var, axis=1)
            true_cpt = new_cpt[(new_cpt[var] == True)].drop(var, axis=1)

            # get remaining variables and keep the maximum
            Y = [col for col in true_cpt.columns if col != 'p']
            new_cpt = pd.concat([false_cpt, true_cpt]).groupby(Y)['p'].max().reset_index()

        return new_cpt

    def marginal_dist(self, Q: list, E: dict, var: list, MAP = False, MPE = False) -> dict:
        '''
        Calculate the marginal distribution of Q given evidence E.
        :Param Q: list of variables in Q
        :Param E: list of variables in the evidence
        :Param var: ordered list of variables not in Q
        :Return: marginal distribution
        '''

        # first, prune the network based on the query and the evidence:
        self.network_pruning(Q, E)

        # get the probability of the evidence
        evidence_factor = 1
        for variable in E:
            cpt = self.bn.get_cpt(variable)
            evidence_factor *= self.bn.get_cpt(variable)['p'].sum()

        # get all cpts in which the variable occurs
        S = self.bn.get_all_cpts()

        factor = 0

        # loop over every variable not in Q
        for variable in var:
            factor_var = {}

            for cpt_var in S:

                if variable in S[cpt_var]:
                    factor_var[cpt_var] = S[cpt_var]

            # apply chain rule and eliminate all variables
            if len(factor_var) >= 2:
                multiplied_cpt = self.MultiplyFactors(list(factor_var.values()))

                new_cpt = self.sumOutVars(multiplied_cpt, [variable])

                for factor_variable in factor_var:
                    del S[factor_variable]

                factor +=1
                S["factor "+str(factor)] = new_cpt

            # when there is only one cpt, don't multiply
            elif len(factor_var) == 1:
                new_cpt = self.sumOutVars(list(factor_var.values())[0], [variable])

                for factor_variable in factor_var:
                    del S[factor_variable]

                factor +=1
                S["factor "+str(factor)] = new_cpt

        if len(S) > 1:
            marginal_dist = self.MultiplyFactors(list(S.values()))
        else:
            marginal_dist = list(S.values())[0]

        marginal_dist['p'] = marginal_dist['p'].div(evidence_factor)
        return marginal_dist

    def MAP(self, Q: list, E: dict, var: list) -> dict:
        '''
        Calculate MAP of variables in Q given evidence E.
        :Param Q: list of variables in Q
        :Param E: list of variables in the evidence
        :Param var: ordered list of variables not in Q
        :Return: MAP
        '''

        # first, prune the network based on the query and the evidence:
        self.network_pruning(Q, E)

        # get the probability of the evidence
        evidence_factor = 1
        for variable in E:
            cpt = self.bn.get_cpt(variable)
            evidence_factor *= self.bn.get_cpt(variable)['p'].sum()

        # get all cpts in which the variable occurs
        S = self.bn.get_all_cpts()

        factor = 0

        # loop over every variable not in Q
        for variable in var:
            factor_var = {}

            for cpt_var in S:

                if variable in S[cpt_var]:
                    factor_var[cpt_var] = S[cpt_var]

            # apply chain rule and eliminate all variables
            if len(factor_var) >= 2:
                multiplied_cpt = self.MultiplyFactors(list(factor_var.values()))

                new_cpt = self.sumOutVars(multiplied_cpt, [variable])

                for factor_variable in factor_var:
                    del S[factor_variable]

                factor +=1
                S["factor "+str(factor)] = new_cpt

            # when there is only one cpt, don't multiply
            elif len(factor_var) == 1:
                new_cpt = self.sumOutVars(list(factor_var.values())[0], [variable])

                for factor_variable in factor_var:
                    del S[factor_variable]

                factor +=1
                S["factor "+str(factor)] = new_cpt

        if len(S) > 1:
            MAP = self.MultiplyFactors(list(S.values()))
        else:
            MAP = list(S.values())[0]

        return MAP.iloc[MAP['p'].argmax()]

    def MPE(self, E: dict, var: list):
        '''
        Calculate the MPE given evidence E.
        :Param E: list of variables in the evidence
        :Param var: ordered list of all variables in the BN
        :Return: MPE
        '''
        Q = []


        # first, prune the network based on the query and the evidence:
        self.network_pruning(Q, E)

        # get the probability of the evidence
        evidence_factor = 1
        for variable in E:
            cpt = self.bn.get_cpt(variable)
            evidence_factor *= self.bn.get_cpt(variable)['p'].sum()

        # get all cpts in which the variable occurs
        S = self.bn.get_all_cpts()

        factor = 0

        # loop over every variable not in Q
        for variable in var:
            factor_var = {}

            for cpt_var in S:

                if variable in S[cpt_var]:
                    factor_var[cpt_var] = S[cpt_var]

            # apply chain rule and eliminate all variables
            if len(factor_var) >= 2:
                new_cpt = self.MultiplyFactors(list(factor_var.values()))

                for factor_variable in factor_var:
                    del S[factor_variable]

                factor +=1
                S["factor "+str(factor)] = new_cpt

        if len(S) > 1:
            MPE = self.MultiplyFactors(list(S.values()))

        else:
            MPE = list(S.values())[0]

        MPE = MPE.iloc[MPE['p'].astype(float).argmax()]

        return MPE
