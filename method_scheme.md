# Lemma
Maximum and minimum values can be approximated with their smooth counterparts. The smooth counterpart is differentiable and therefore applicable for NN applications. [Wikipedia](https://en.wikipedia.org/wiki/Smooth_maximum)
We approximate the softmin function with LogSumExp (LSE) for $x=(x_1, \ldots, x_n)$ as:
$$\mathrm{softmin}_{\lambda}(x) = -\lambda \mathrm{log} \sum_{i \leq n} \mathrm{exp}(-x_i/\lambda)$$
In the limit, $\mathrm{softmin}_{\lambda}(x) \xrightarrow{\lambda\rightarrow 0} \mathrm{min}(x)$. However, for the NN applications the $\lambda$ will not be very close to $0$.

# Lemma
Iterative approaches are differentiable. We can use Long Short-Term Memory (LSTM) network for iterative calculations.
See [PyTorch Tutorial](https://docs.pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html), [PyTorch Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).

# Problem definition
We seek a binary adjacency matrix $A_f$ of the undirected graph $G$. To achieve this, we will work on the the continuous version $A=[a_{ij}]_{i,j\leq n}: a_{i, j} \in [0, 1]$. Distance between nodes is expressed via the distance (weight) matrix $D$.
To ensure that avoid null division, we introduce $W = [w_{ij}] := \frac{d_{ij}}{a_{ij}+\delta}$. For a large $w_{ij}$ (small $a_{ij}$, the path will be effectively removed from the shortest path calculation. Let $X$ denote a vector with node features (population).

| Symbol | Definition |
| --- | --- |
| G | original, undirected graph, weighted nodes and edges |
| A | soft adjacency matrix (entries in $[0, 1]$) |
| D | distance between nodes - original weight matrix |
| W | amended weight matrix, where unlikely links are assigned huge weights |

# Shortest path
We will apply Floyd–Warshall algorithm ([wiki](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm)). Thankfully, due to the nature of our problem, we can easily fix a low maximal number of intermediary points and otherwise return a huge constant.

#### Algorithm
We consider the graph $G$ with nodes $V = (1, \ldots, n)$. Consider a function $\mathrm{shortestPath}(i, j, k)$ between nodes $i$ and $j$ using vertices from the set $\{1, \ldots, k\}$. Note, the shortest path is naturally given by $\mathrm{shortestPath}(i, j, N := |V|)$, which we'll find recursively. 
Note also that $\mathrm{shortestPath}(i, j, k) \leq \mathrm{shortestPath}(i, j, k-1)$ (greater flexibility).


