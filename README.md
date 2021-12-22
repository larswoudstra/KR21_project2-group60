# PGM Group 60

This project was developed by: 
[Stans Paulussen](https://canvas.vu.nl/groups/193220/users/169653), 

[Lars Woudstra](https://canvas.vu.nl/groups/193220/users/175984)

[Kelly Spaans](https://canvas.vu.nl/groups/193220/users/177742)

[Sergio Alejandro Gutierrez Maury](https://canvas.vu.nl/groups/188476/users/159566) 

For the course Knowledge and Representation at the Vrije Universiteit Amsterdam (VU).


## BNReasoner Class

For this project we developed a Bayesian Network Reasoner class, using the Bayesian Network class, developed by [Erman Acar](https://canvas.vu.nl/courses/55684/users/64587 "Author's name").
Here we give a brief description of the main methods in this class:



 - dsep(X,Y,Z): This method accepts three lists of variables and return a boolean value True or False, if variables in X are d-separated from Y given Z.
 
 - sumOutVars(cpt, Z): Accepts a pandas  pandas Data Frame type cpt table and a list of variables Z. It sums the probabilities and removes every variable in Z from the cpt and returns a newly made cpt. 
 
 - MultiplyFactor(X): Accepts a list of pandas Data Frame type cpts, which you want to multiply and returns a factor with multiplied cpts in a pandas Data Frame type.
 
 - network_pruning(Q, E): Accepts a list of variables to perform a Query (Q), and dictionary with Evidence (E), and it performs a node- and edge-pruning, it returns a newly pruned Bayesian Network. 
 
 - MinDegreeOrder(): Returns an ordered variable list with in which the variables are ordered based on the least number of neighbors.
 
 - MAP(Q,E,var): Accepts a list of variables to perform a Query (Q), a dictionary with Evidence (E), and a list of ordered variables (var) which do not appear in Q. Returns a dictionary with Most a Posteriori estimate queries.
 
 - MEP(E, var): Accepts a dictionary with Evidence (E), and a list of ordered variables (var). Returns a dictionary with Most Probable Explantions.

Note: The functions network_pruning, marginal_dist, MAP and MEP update the reasoner that is created. Every time before running these functions, a new reasoner object should be created. 
