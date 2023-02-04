Samieh Baniasadi


400422045


# Density-based clustering
## Theory
Density-based clustering is a type of clustering algorithm that seeks to find clusters of dense areas in a dataset. It is different from other clustering methods such as k-means and hierarchical clustering, which use the distances between data points to determine clusters. Density-based clustering identifies clusters by analyzing the number of data points within a given neighborhood. Points that are close together and have many nearby neighbors are considered dense and form clusters. The algorithm is commonly used in data mining and machine learning for identifying patterns and structures in large datasets. Examples of density-based clustering algorithms include DBSCAN, OPTICS, and HDBSCAN.

Density-based clustering is a method for grouping data points in a dataset such that the number of points within a certain area is maximized and the distances between points are minimized. It is a type of unsupervised learning, as it does not rely on pre-existing labels to form clusters. The algorithm works by defining a neighborhood around each data point, and grouping points together if they are within each other's neighborhoods. The result is a partition of the data into clusters based on density. This method has proven useful for identifying patterns and structures in complex and noisy datasets.

## Mathematical
Density-based clustering can be mathematically formalized as an optimization problem, where the objective is to maximize the density of data points within a cluster and minimize the distances between data points in different clusters. This can be achieved using various metrics, such as the number of data points within a neighborhood or the distances between data points and the cluster centroids. The optimization problem can be solved using techniques such as gradient descent, Markov Chain Monte Carlo, or heuristics such as the k-means algorithm. The resulting clusters can be represented as a set of parameters, such as the centroids or covariance matrices, that describe the properties of each cluster.

# Principle method
DBSCAN
## Theory
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised machine-learning algorithm for clustering. It groups together points in a dataset based on the density of the points, where points are considered to be in the same cluster if they are close to each other and are surrounded by a sufficient number of other points. The algorithm identifies dense areas of the dataset (called clusters) and separates sparse or noisy regions (called outliers).

## Mathematical
DBSCAN is based on the concept of density. It uses two parameters, epsilon (ε) and a minimum number of points (MinPts), to define the density of a cluster.

Epsilon (ε): The maximum distance between two points in the same cluster.

MinPts: The minimum number of points required to form a dense region.

The algorithm works by starting from a random point and checking if there are at least MinPts within a distance of ε. If there are, these points are considered part of the same cluster. The process then continues by checking the neighbors of these points, and so on, until all points in the cluster have been identified.

The algorithm assigns each point in the dataset one of three labels: core point, border point, or noise point.

- Core point: A point that has at least MinPts within a distance of ε.

- Border point: A point that is within a distance of ε from a core point but does not have at least MinPts within a distance of ε.

- Noise point: A point that is not a core point or a border point.

## Algorithm
The DBSCAN algorithm can be outlined as follows:

1. Choose a random point in the dataset and check if there are at least MinPts within a distance of ε.

2. If there are, consider all these points as part of the same cluster and mark them as visited.

3. For each of the visited points, find its neighbors that are within a distance of ε and repeat step 2 for those points.

4. Continue the process until all points in the cluster have been visited and marked as part of the same cluster.

5. Repeat the process from step 1 for all unvisited points in the dataset.

6. Finally, label each point in the dataset as a core point, border point, or noise point.

Note: The algorithm can be optimized by using a data structure such as a k-d tree or an R-tree to reduce the time complexity of finding the neighbors.

# State-of-the-art method
DBSCAN++
## Theory
In DBSCAN++ method, a step towards a fast
and scalable DBSCAN. DBSCAN++ is based on the observation that they only need to compute the density estimates for a subset `m` of the `n` data points, where `m` can be much smaller than `n`, to cluster properly. To choose these `m` points, they provide two simple strategies: uniform and greedy K-center-based sampling. The resulting procedure has `O(mn)` worst-case runtime.

they show the trade-off between computational cost and estimation rates. Interestingly, up to a certain point, we can enjoy the same minimax-optimal estimation rates attained by DBSCAN while making m (instead of the larger n) empirical density queries, thus leading to a sub-quadratic procedure. In some cases, we saw that our method of limiting the number of core points can act as a regularization, thus reducing the sensitivity of classical DBSCAN to its parameters.

## Mathematical
Definition 1. (Level-set) The λ-level-set of f is defined as Lf (λ) := {x ∈ X : f(x) ≥ λ}.

Definition 2. d(x, A) := inf x'∈A |x − x'|, B(C, r) :={x ∈ X : d(x, C) ≤ r}

## Hyperparameter Settings
They state the hyperparameter settings in terms of n, the sample size, and the desired density level λ for statistical consistency guarantees to hold. Define Cδ,n =16 log(2/δ) √log n, where δ, 0 < δ < 1, is a confidence parameter that will be used later (i.e. guarantees will hold with probability at least 1 − δ).
ε = ( minPts / n · vD · (λ − λ · C^2 / √minPts)^1/D

Definition 3 (Hausdorff Distance) dHaus(A, A') = max{sup x∈A d(x, A'), sup (x'∈A') d(x', A)}.

Lemma (Noise points). For any dataset, if N0 and N1 are the noise points found by DBSCAN and DBSCAN++ then as long as they have the same settings of ε and k, we have that N0 ⊆ N1.
Proof. Noise points are those that are further than ε distance away from a core point. The result follows since DBSCAN++ core points are a subset of that of DBSCAN.


## Algorithm
They need samples X = {x1, ..., xn} drawn from a distribution F over R^D. We now define core-points,
which are essentially points with high empirical density defined concerning the two hyperparameters of DBSCAN, minPts and ε. The latter is also known as the bandwidth.

Definition 1. `Let ε > 0 and minPts be a positive integer.
Then x ∈ X is a core-point if |B(x, ε) ∩ X| ≥ minPts,
where B(x, ε) := {x': |x − x'| ≤ ε}`.

### Uniform Initialization
DBSCAN++, Algorithm proceeds as follows:
First, it chooses a subset S of m uniformly sampled points from the dataset X. Then, it computes the empirical density of points in S w.r.t. the entire dataset. That is, a point x ∈ S is a core point if |B(x, ε) ∩ X| ≥ minPts. From here, DBSCAN++ builds a similar neighborhood graph G of core-points in S and finds the connected components in G. Finally, it clusters the rest of the unlabeled points to their closest core-points. Thus, since it only recovers a fraction of the core-points, it requires expensive density estimation queries on only m of the n samples.

### K-Center Initialization
Instead of uniformly choosing the subset of m points at random, they use K-center which aims at finding the subset of size m that minimizes the maximum distance of any point in X to its closest point in that subset. In other words, it tries to find the most efficient covering of the sample points. they use the greedy initialization method for approximating K-center, which repeatedly picks the farthest point from any point currently in the set. This process continues until we have a total of m points. This method gives a 2-approximation to the K-center problem.

# Data
I worked on the UCI seeds dataset. The examined group comprised kernels belonging to three different varieties of wheat: Kama, Rosa and Canadian, 70 elements each, randomly selected for the experiment. High-quality visualization of the internal kernel structure was detected using a soft X-ray technique. Studies were conducted using combined harvested wheat grain originating from experimental fields, explored at the Institute of Agrophysics of the Polish Academy of Sciences in Lublin.

The data set can be used for the tasks of classification and cluster analysis.

## Dimension
To construct the data, seven geometric parameters of wheat kernels were measured:

1. area A,
2. perimeter P,
3. compactness C = 4*pi*A/P^2,
4. length of kernel,
5. width of kernel,
6. asymmetry coefficient
7. length of kernel groove.

All of these parameters were real-valued continuous.

