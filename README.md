# Label Propagation with MPI

## What is this project?
This is an implementation of parallel label propagation algorithm with MPI.
- [Label Propagation](https://en.wikipedia.org/wiki/Label_propagation_algorithm)
- [Open MPI](https://www.open-mpi.org/)

## Problem
Given a graph `G` and outputs the connected components of `G`, i.e. the vertex labels identifying the component to which each vertex belongs. The input file for the graph consist of a set of `(i, j)` space-separated tuples corresponding to the edges of the graph. Note that the input files are sorted in increasing `(i, j)` order.

### Input Dataset
For the input files, the first line contains the number of nodes and the number of edges (space separated). The following lines contain the edges of the graph. The graphs are undirected (i.e., if `(i, j)` is present in the input file, then `(j, i)` will be present as well). The graphs also do not contain any self-referential edges (i.e., there is no i such that (i, i) is in our input file). Finally, the file names indicate the number of vertices in the graph (i.e., 12.txt has 12 vertices). There are 50 components for 1000.txt, 500 components for both 10000.txt and 100000.txt, 5000 components for 1000000.txt. 

### Output Labels
The output file contain `n` lines, each line should contain the component label for node `i` on the `i`-th line. As an example, if we have 4 nodes with 2 edges of (0, 1) and (2, 3), then your file output, using the above algorithm, should be as follows.
```
1
1
3
3
```

## Approach
- Step 1: read input graph and distribute nodes to all processes
  - The root process (rank 0) reads the input graph from a file.
  - The nodes of the graph are divided approximately equally among all processes. This division is communicated to all processes.
  - The graph's adjacency list is converted into a Compressed Sparse Row (CSR) format for efficient storage and access. The CSR data structure includes two arrays: `global_row_ptr`(row pointers) and `global_col_ind` (column indices), which are distributed to all processes according to the division of nodes.

- Step 2: analyze non local edges
  - For each edge in the assigned portion of the graph, the process calculates whether the second node (target node of an edge) is local (i.e., managed by the same process) or non-local (managed by another process).
  - This determination is made by dividing the node index by the number of nodes per process. The quotient gives the process number to which the node is assigned.
  - If the calculated process number exceeds the total number of processes, it is adjusted to target the last process. This adjustment accounts for any imbalance in node distribution, ensuring that the last process handles any overflow.

- Step 3: determine the responsible process for non-local nodes
  - For each non-local edge identified in Step 2, the process calculates the responsible process for the second node using the method described above.
  - This step ensures that each process knows exactly where to send information about non-local nodes, making the use of efficient inter-process communication.

- Step 4: figure out which process to send what data to which process
  - Once all non-local nodes have been identified and their responsible processes determined, the process collects the first nodes of those edges (i.e., the local nodes that have edges to non-local nodes) into an array.
  - This array represents the set of local nodes whose information needs to be shared with other processes because they are connected to non-local nodes.
  - The process also prepares another array containing the labels of these local nodes, as these labels will need to be compared with the labels of their non-local neighbors during the label propagation step.
  - Based on the information collected, the process calculates send counts (how many nodes' data each process needs to send) and displacements (the starting indices in the send buffer for each process).
  - An `MPI_Alltoallv` (all to all) communication is performed to exchange this data among processes. Each process sends out data about its local nodes that are connected to non-local nodes, and receives similar data from other processes.
  
- Step 5: label propagation
  - After exchanging local node information, each process performs the label propagation algorithm on its assigned portion of the graph.
  - The process iterates through its CSR column indices. For each local node, it compares its label with the labels of its neighbors. If a neighbor is non-local, the process uses the label received from the corresponding process during the MPI_Alltoallv communication in Step 4.
  - The goal is to update each node's label to the smallest label observed among its neighbors, promoting label consistency across the graph.
  - This step may involve multiple iterations, with each process exchanging updated labels after each iteration, until no labels change or a predefined convergence criterion is met.
  - The use of `MPI_Alltoallv` for exchanging labels ensures that each process receives the most up-to-date labels for non-local nodes it is interested in, allowing for accurate label updates across process boundaries.

- Step 6: output
  - After convergence, the processes collect and combine their label data.
  - The root process gathers the labels from all processes using MPI_Gatherv, which collects variable amounts of data from each process.
  - The root process then outputs the final labels to a file.

- Compile
  ```bash
  make lpa
  ```
- Run
  ```bash
  mpirun -np <p> ./lpa <graph> <labels>
  ```
  - `p` is the number of processes
  - `graph` is the input file
  - `labels` is the output file

