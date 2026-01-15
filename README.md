# Problem

For a given set of nodes $N$, located on a 2D space (long, lat) with a weight (population) 
$N \ni n_i = (x_i, y_i, w_i )$ and a demand matrix spanned between the nodes
$Q_{ij}$.

> Interpret as cities in Poland with their locations and a demand matrix of people willing to travel among them

find an _optimal_ subgraph $G^* \subset G$ connecting it. 

> Interpet as optimal High Speed Rail network between the cities.

Optimal in a sense of _Loss_ , composed of: 

* Benefits
* Costs

Costs are link additive constant, proportional to distance $c_{ij}$ - for instance cost of bulgind 100km railway between Warsaw and Łódź

$C(G')=\sum_{a \in G'}c_{ij}$

Benefits are measured with some classic transport-based metric, for instance accesibility or passenger hours:

$B(G') = Q \otimes T(G')
