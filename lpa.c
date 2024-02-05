#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_FILENAME_LENGTH 256

/**
* @brief Write a vector of labels to a file.
*
* @param filename The name of the file to write to.
* @param labels The array of labels.
* @param nlabels How many labels to write.
*/
static void print_labels(
    char const * const filename,
    unsigned const * const labels,
    size_t const nlabels)
{
  size_t i;
  FILE * fout;

  /* open file */
  if((fout = fopen(filename, "w")) == NULL) {
    fprintf(stderr, "error opening '%s'\n", filename);
    abort();
  }

  /* write labels to fout */
  for(i = 0; i < nlabels; ++i) {
    fprintf(fout, "%u\n", labels[i]);
  }

  fclose(fout);
}

/**
* @brief Output the seconds elapsed steps 2-5. This excludes input and
*        output time. This should be wallclock time, not CPU time.
*
* @param seconds Seconds spent sorting.
*/
static void print_time25(
    double const seconds)
{
  printf("2-5 Time: %0.04fs\n", seconds);
}

/**
* @brief Output the seconds elapsed for step 5. This excludes input and
*        output time. This should be wallclock time, not CPU time.
*
* @param seconds Seconds spent sorting.
*/
static void print_time5(
    double const seconds)
{
  printf("5 Time: %0.04fs\n", seconds);
}

/**
 * @brief Check if any process has changed
 * @param all_changed Array of all processes' changed status
 * @param p Number of processes
 * @return int 1 if any process has changed, 0 otherwise
*/
int has_changed(int* all_changed, int p) {
  for (int i = 0; i < p; i++) {
    if (all_changed[i] == 1) {
      return 1;
    }
  }
  return 0;
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if (argc != 3) {
    if (rank == 0) {
      printf("Usage: %s <graph> <labels>\n", argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  // input
  char input_graph[MAX_FILENAME_LENGTH];
  strncpy(input_graph, argv[1], MAX_FILENAME_LENGTH);

  // output
  char output_labels[MAX_FILENAME_LENGTH];
  strncpy(output_labels, argv[2], MAX_FILENAME_LENGTH);

  // ==================== step 1: read input graph and distribute nodes to all processes ====================

  int n_nodes = 0;                  // number of nodes for this process
  int* global_row_ptr = NULL;       // CSR row pointer
  int* global_col_ind = NULL;       // CSR column indices
  int* row_ptr_sc = NULL;           // sendcounts for scattering CSR row pointer
  int* row_ptr_disp = NULL;         // displacements for scattering CSR row pointer
  int* col_ind_sc = NULL;           // sendcounts for scattering CSR column indices
  int* col_ind_disp = NULL;         // displacements for scattering CSR column indices

  int start = 0;                    // starting node for this process
  int end = 0;                      // ending node for this process
  int nnpp = 0;                     // number of nodes per process

  if (rank == 0) {
    // read input graph
    FILE* f = fopen(input_graph, "r");

    if (f == NULL) {
      fprintf(stderr, "Could not open file %s\n", input_graph);
      MPI_Finalize();
      return 1;
    }

    int all_n_nodes, all_n_edges;
    fscanf(f, "%d %d", &all_n_nodes, &all_n_edges);

    // send number of nodes each process will receive
    nnpp = all_n_nodes / p;
    row_ptr_sc = malloc(p * sizeof(int));
    row_ptr_disp = malloc(p * sizeof(int));
    for (int i = 0; i < p-1; i++) {
      row_ptr_sc[i] = nnpp;
      row_ptr_disp[i] = i * nnpp;
      start = i * nnpp;
      end = start + nnpp - 1;
      MPI_Send(&start, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      MPI_Send(&end, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    int last_nnpp = nnpp + all_n_nodes % p;
    row_ptr_sc[p-1] = last_nnpp;
    row_ptr_disp[p-1] = (p-1) * nnpp;

    start = (p-1) * nnpp;
    end = start + last_nnpp - 1;
    MPI_Send(&start, 1, MPI_INT, p-1, 0, MPI_COMM_WORLD);
    MPI_Send(&end, 1, MPI_INT, p-1, 0, MPI_COMM_WORLD);

    // CSR structure
    global_row_ptr = malloc((all_n_nodes + 1) * sizeof(int));
    global_col_ind = malloc(all_n_edges * sizeof(int));
    col_ind_sc = malloc(p * sizeof(int));
    int* n_edge_per_node = malloc(all_n_nodes * sizeof(int));

    for (int i = 0; i < all_n_nodes + 1; i++) {
      global_row_ptr[i] = 0;
    }
    for (int i = 0; i < p; i++) {
      col_ind_sc[i] = 0;
    }
    for (int i = 0; i < all_n_nodes; i++) {
      n_edge_per_node[i] = 0;
    }
    for (int i = 0; i < all_n_edges; i++) {
      int u, v;
      fscanf(f, "%d %d", &u, &v);
      n_edge_per_node[u]++;
      global_col_ind[i] = v;

      int proc = u / nnpp;
      if (proc == p) {
        proc--;
      }
      col_ind_sc[proc]++;
    }
    for (int i = 0; i < all_n_nodes; i++) {
      global_row_ptr[i+1] = global_row_ptr[i] + n_edge_per_node[i];
    }

    col_ind_disp = malloc(p * sizeof(int));
    for (int i = 0; i < p; i++) {
      col_ind_disp[i] = 0;
    }
    int cumsum = 0;
    for (int i = 0; i < p; i++) {
      col_ind_disp[i] = cumsum;
      cumsum += col_ind_sc[i];
    }
    
    fclose(f);
  }
  MPI_Bcast(&nnpp, 1, MPI_INT, 0, MPI_COMM_WORLD);                                  // average number of nodes per process
  MPI_Scatter(row_ptr_sc, 1, MPI_INT, &n_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);     // number of nodes for this process

  // start and end nodes for this process
  MPI_Recv(&start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(&end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // CSR row pointer and column indices for this process
  int* row_ptr = (int *) malloc((n_nodes + 1) * sizeof(int));
  MPI_Scatterv(global_row_ptr, row_ptr_sc, row_ptr_disp, MPI_INT, row_ptr, n_nodes, MPI_INT, 0, MPI_COMM_WORLD);

  int n_col_ind = 0;                                                                // number of column indices for this process
  MPI_Scatter(col_ind_sc, 1, MPI_INT, &n_col_ind, 1, MPI_INT, 0, MPI_COMM_WORLD);
  row_ptr[n_nodes] = row_ptr[0] + n_col_ind;    // set last element of row_ptr to be the number of column indices for this process

  int* col_ind = (int *) malloc(n_col_ind * sizeof(int));
  MPI_Scatterv(global_col_ind, col_ind_sc, col_ind_disp, MPI_INT, col_ind, n_col_ind, MPI_INT, 0, MPI_COMM_WORLD);

  // modify row_ptr to be relative to this process
  int first = row_ptr[0];
  for (int i = 0; i < n_nodes; i++) {
    row_ptr[i] -= first;
  }
  row_ptr[n_nodes] = n_col_ind;

  // clean up
  if (rank == 0) {
    free(global_row_ptr);
    free(global_col_ind);
    free(row_ptr_sc);
    free(row_ptr_disp);
    free(col_ind_sc);
    free(col_ind_disp);
  }

  // =================================================================================================
  //                        step 2: analyze non local edges
  //                        step 3: determine which process are non local nodes on
  //                        step 4: figure out which process to send what data to which process
  // =================================================================================================

  double start_time = MPI_Wtime();

  // count remote nodes that is belongs to each process
  int n_remote_nodes = 0;
  int* n_remote_nodes_arr = malloc(p * sizeof(int));                        // sendcounts for sending the local nodes to other processes
  for (int i = 0; i < p; i++) {
    n_remote_nodes_arr[i] = 0;
  }
  for (int i = 0; i < n_col_ind; i++) {
    int v = col_ind[i];
    if (v < start || v > end) {           // remote node
      int remote_proc = v / nnpp;
      if (remote_proc == p) {
        remote_proc--;
      }
      n_remote_nodes_arr[remote_proc]++;
      n_remote_nodes++;
    }
  }

  int** local_nodes_to_send_2d = malloc(p * sizeof(int*));                  // local nodes to send
  int* local_nodes_to_send_arr = malloc(n_remote_nodes * sizeof(int*));
  for (int i = 0; i < p; i++) {
    local_nodes_to_send_2d[i] = malloc(n_remote_nodes_arr[i] * sizeof(int));
    for (int j = 0; j < n_remote_nodes_arr[i]; j++) {
      local_nodes_to_send_2d[i][j] = 0;
    }
  }
  int* indices = malloc(p * sizeof(int));                                   // indices for local_nodes_to_send_2d per process
  for (int i = 0; i < p; i++) {
    indices[i] = 0;
  }
  for (int i = 0; i < n_nodes; i++) {
    for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
      int v = col_ind[j];
      if (v < start || v > end) {         // remote node
        int remote_proc = v / nnpp;
        if (remote_proc == p) {
          remote_proc--;
        }
        local_nodes_to_send_2d[remote_proc][indices[remote_proc]] = i + start;
        indices[remote_proc]++;
      }
    }
  }
  // flatten local_nodes_to_send_2d into 1d array
  int index = 0;
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < n_remote_nodes_arr[i]; j++) {
      local_nodes_to_send_arr[index] = local_nodes_to_send_2d[i][j];
      index++;
    }
  }

  // clean up
  free(indices);
  for (int i = 0; i < p; i++) {
    free(local_nodes_to_send_2d[i]);
  }
  free(local_nodes_to_send_2d);

  int* senddispls = malloc(p * sizeof(int));                          // displacements for sending nodes to other processes
  for (int i = 0; i < p; i++) {
    senddispls[i] = 0;
  }
  int cumsum = 0;
  for (int i = 0; i < p; i++) {
    senddispls[i] = cumsum;
    cumsum += n_remote_nodes_arr[i];
  }

  int* local_labels = malloc(n_nodes * sizeof(int));                  // local labels
  int* remote_labels = malloc(n_remote_nodes * sizeof(int));          // remote labels
  int* remote_nodes = malloc(n_remote_nodes * sizeof(int));           // remote nodes in this process
  int* local_labels_to_send = malloc(n_remote_nodes * sizeof(int));   // local labels that will be sent to other processes

  for (int i = 0; i < n_nodes; i++) {
    local_labels[i] = i + start;
  }
  for (int i = 0; i < n_remote_nodes; i++) {
    remote_labels[i] = 0;
  }
  for (int i = 0; i < n_remote_nodes; i++) {
    remote_nodes[i] = 0;
  }
  for (int i = 0; i < n_remote_nodes; i++) {
    local_labels_to_send[i] = local_labels[local_nodes_to_send_arr[i] - start];
  }

  // all to all for get the remote nodes (use it to find remote labels)
  MPI_Alltoallv(local_nodes_to_send_arr, n_remote_nodes_arr, senddispls, MPI_INT, remote_nodes, n_remote_nodes_arr, senddispls, MPI_INT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  // ==================== step 5: label propagation ====================

  double start_time5 = MPI_Wtime();

  int* all_changed = malloc(p * sizeof(int));                       // array of all processes' changed status
  int my_changed = 0;                                               // this process' changed status
  int* temp = malloc(n_nodes * sizeof(int));                        // temporary array for updating local labels

  int* senddispls_with_end = malloc((p+1) * sizeof(int));           // displacements for sending nodes to other processes but with end
  for (int i = 0; i < p; i++) {
    senddispls_with_end[i] = senddispls[i];
  }
  senddispls_with_end[p] = n_remote_nodes;

  do {
    // reset
    my_changed = 0;
    for (int i = 0; i < n_nodes; i++) {
      temp[i] = local_labels[i];
    }
    for (int i = 0; i < n_remote_nodes; i++) {
      remote_labels[i] = 0;
    }

    // all to all to get remote labels
    MPI_Alltoallv(local_labels_to_send, n_remote_nodes_arr, senddispls, MPI_INT, remote_labels, n_remote_nodes_arr, senddispls, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // iterating through the local nodes and their edges
    for (int i = 0; i < n_nodes; i++) {
      temp[i] = local_labels[i];
      for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
        int v = col_ind[j];                   // neighbour
        if (v < start || v > end) {           // remote node
          int remote_proc = v / nnpp;         // v's process
          if (remote_proc == p) {
            remote_proc--;
          }

          // find index of label of v in remote_labels
          int index = 0;                      
          for (int k = senddispls_with_end[remote_proc]; k < senddispls_with_end[remote_proc+1]; k++) {
            if (remote_nodes[k] == v) {
              index = k;
              break;
            }
          }

          // remote neighbor's label is smaller
          if (temp[i] > remote_labels[index]) {
            temp[i] = remote_labels[index];
            my_changed = 1;
          }
        } else {                              // local node
          // local neighbor's label is smaller
          if (temp[i] > local_labels[v - start]) {
            temp[i] = local_labels[v - start];
            my_changed = 1;
          }
        }
      }
    }

    // update local label
    for (int i = 0; i < n_nodes; i++) {
      local_labels[i] = temp[i];
    }
    // update local labels to send
    for (int i = 0; i < n_remote_nodes; i++) {
      local_labels_to_send[i] = local_labels[local_nodes_to_send_arr[i] - start];
    }

    // gather all processes' changed status
    MPI_Allgather(&my_changed, 1, MPI_INT, all_changed, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

  } while (has_changed(all_changed, p));      // while any process has changed

  MPI_Barrier(MPI_COMM_WORLD);
  double end_time5 = MPI_Wtime();

  // output time taken
  if (rank == 0) {
    print_time25(end_time5 - start_time);
    print_time5(end_time5 - start_time5);
  }

  // ==================== step 6: output ====================

  int* all_labels = NULL;                           // array of all processes' labels
  int* all_labels_rc = NULL;                        // array of all processes' labels' receive counts
  int* all_labels_disp = NULL;                      // array of all processes' labels' displacements
  if (rank == 0) {
    all_labels_rc = malloc(p * sizeof(int));
    all_labels_disp = malloc(p * sizeof(int));
  }
  MPI_Gather(&n_nodes, 1, MPI_INT, all_labels_rc, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int total_n_labels = 0;
  if (rank == 0) {
    for (int i = 0; i < p; i++) {
      all_labels_disp[i] = total_n_labels;
      total_n_labels += all_labels_rc[i];
    }
    all_labels = malloc(total_n_labels * sizeof(int));
  }

  // gather local labels from all processes
  MPI_Gatherv(local_labels, n_nodes, MPI_INT, all_labels, all_labels_rc, all_labels_disp, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    // output gathered labels
    print_labels(output_labels, all_labels, total_n_labels);

    // clean up
    free(all_labels);
    free(all_labels_rc);
    free(all_labels_disp);
  }

  // clean up
  free(row_ptr);
  free(col_ind);
  free(local_labels);
  free(remote_labels);
  free(senddispls);
  free(n_remote_nodes_arr);
  free(local_nodes_to_send_arr);
  free(remote_nodes);
  free(local_labels_to_send);
  free(all_changed);
  free(temp);
  free(senddispls_with_end);

  MPI_Finalize();
  return 0;
}