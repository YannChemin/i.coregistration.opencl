/* SPDX-License-Identifier: Unlicense */

#ifndef I_COREGISTRATION_OPENCL_MAXFLOW_H
#define I_COREGISTRATION_OPENCL_MAXFLOW_H

/* Minimum s-t cut of a graph with non-negative capacities, for binary
 * labellings with submodular pairwise terms (Kolmogorov and Zabih,
 * 2004), by Dinic's blocking-flow algorithm. Nodes are 0..n-1; the
 * source and the sink are implicit. */
struct maxflow;

struct maxflow *maxflow_new(int n, int edge_hint);
void maxflow_free(struct maxflow *g);

/* Energy terms: unary cost e0 when node i takes label 0 (source side),
 * e1 when it takes label 1 (sink side); pairwise costs e00, e01, e10, e11
 * of labels (i, j), with e01 + e10 >= e00 + e11. */
void maxflow_unary(struct maxflow *g, int i, double e0, double e1);
void maxflow_pairwise(struct maxflow *g, int i, int j, double e00, double e01,
                      double e10, double e11);

/* Minimize the energy; returns its minimum. */
double maxflow_solve(struct maxflow *g);

/* Label of node i in the minimum: 0 (source side) or 1 (sink side). */
int maxflow_label(const struct maxflow *g, int i);

#endif /* I_COREGISTRATION_OPENCL_MAXFLOW_H */
