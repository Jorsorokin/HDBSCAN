# HDBSCAN
A MATLAB implementation of the Hierarchical Density-based Clustering for Applications with Noise, ([HDBSCAN](http://joss.theoj.org/papers/b5c5dd4b7491890b711c06225dcc9649)), clustering algorithm. 

The HDBSCAN algorithm creates a nested hierarchy of density-based clusters, discovered in a non-parametric way from the input data. The hierarchies are akin to Single Linkage Clustering, however in HDBSCAN, an optimal clustering scheme is automatically inferred from the cluster hierarchy. The optimal clustering is analogous to a single run of the [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) algorithm, but with possibly varying epsilon-values (see the role of epsilon in DBSCAN) for any given branch of the hierarchy. Thus, information from local neighborhoods is used to optimally cut the hierarchy at varying levels.

This MATLAB implementation of the HDBSCAN algorithm was created with peformance in mind, and is inspired by the excellent [python version](http://hdbscan.readthedocs.io/en/latest/). While this version is not as fast as the python implementation (in which highly optimized C code was compiled for iterating through the hierarchy), it is extremely easy to use, requires no dependencies on external toolboxes, and is currently the only MATLAB-based HDBSCAN algorithm.

See the [docs](docs) for interfacing and running HDBSCAN with your own data.

You are free to use/distribute the code, but please keep a reference to this original code base and author (Jordan Sorokin). 

## Dependencies
- MATLAB version r2015a or greater
- bfs.m and mst_prim.m, courtesy of David Gleich (included in the repo)

## References
- Campello et al. (2013): Density-Based Clustering Based on Hierarchical Density Estimates.
- Campello et al. (2015): Hierarchical density estimates for data clustering, visualization, and outlier detection  

## Known Issues
- Prediction of new points is an approximation, as the cluster hierarchy is not modified with new points
- Hierarchy update with new labels has heuristics in place to deal with new clusters arising from previously labeled outliers


