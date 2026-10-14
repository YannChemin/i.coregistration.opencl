/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Minimum s-t cut for binary labellings: the graph
 *               construction of Kolmogorov and Zabih (2004) for
 *               submodular pairwise energies, solved by Dinic's
 *               blocking-flow maximum flow (Dinic, 1970).
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <string.h>

#include <grass/gis.h>

#include "maxflow.h"

struct maxflow {
    int n, src, snk;
    int ne, cap_e;
    int *head, *next, *to;
    double *cap;
    double *unary; /* Accumulated e1 - e0 per node. */
    double constant;
    int *level, *iter, *queue, *stack;
    unsigned char *side; /* 1 when on the source side after solve. */
};

struct maxflow *maxflow_new(int n, int edge_hint)
{
    struct maxflow *g = G_calloc(1, sizeof(*g));

    g->n = n;
    g->src = n;
    g->snk = n + 1;
    g->cap_e = 2 * (edge_hint + n) + 16;
    g->head = G_malloc((n + 2) * sizeof(int));
    for (int i = 0; i < n + 2; i++)
        g->head[i] = -1;
    g->next = G_malloc(g->cap_e * sizeof(int));
    g->to = G_malloc(g->cap_e * sizeof(int));
    g->cap = G_malloc(g->cap_e * sizeof(double));
    g->unary = G_calloc(n, sizeof(double));
    return g;
}

void maxflow_free(struct maxflow *g)
{
    if (!g)
        return;
    G_free(g->head);
    G_free(g->next);
    G_free(g->to);
    G_free(g->cap);
    G_free(g->unary);
    G_free(g->level);
    G_free(g->iter);
    G_free(g->queue);
    G_free(g->stack);
    G_free(g->side);
    G_free(g);
}

/* Edge u -> v of capacity c and its reverse of capacity rc. */
static void add_edge(struct maxflow *g, int u, int v, double c, double rc)
{
    if (g->ne + 2 > g->cap_e) {
        g->cap_e *= 2;
        g->next = G_realloc(g->next, g->cap_e * sizeof(int));
        g->to = G_realloc(g->to, g->cap_e * sizeof(int));
        g->cap = G_realloc(g->cap, g->cap_e * sizeof(double));
    }
    g->to[g->ne] = v;
    g->cap[g->ne] = c;
    g->next[g->ne] = g->head[u];
    g->head[u] = g->ne++;
    g->to[g->ne] = u;
    g->cap[g->ne] = rc;
    g->next[g->ne] = g->head[v];
    g->head[v] = g->ne++;
}

void maxflow_unary(struct maxflow *g, int i, double e0, double e1)
{
    g->constant += e0;
    g->unary[i] += e1 - e0;
}

void maxflow_pairwise(struct maxflow *g, int i, int j, double e00, double e01,
                      double e10, double e11)
{
    /* E = A + (C - A) x_i + (D - C) x_j + (B + C - A - D) (1 - x_i) x_j
     * with A..D = e00, e01, e10, e11: the last term is the edge i -> j,
     * cut when i is on the source side and j on the sink side. */
    double w = e01 + e10 - e00 - e11;

    g->constant += e00;
    g->unary[i] += e10 - e00;
    g->unary[j] += e11 - e10;
    if (w < 0)
        G_fatal_error("maxflow: non-submodular pairwise term");
    if (w > 0)
        add_edge(g, i, j, w, 0);
}

static int bfs(struct maxflow *g)
{
    int qh = 0, qt = 0, u, e;

    for (u = 0; u < g->n + 2; u++)
        g->level[u] = -1;
    g->level[g->src] = 0;
    g->queue[qt++] = g->src;
    while (qh < qt) {
        u = g->queue[qh++];
        for (e = g->head[u]; e >= 0; e = g->next[e])
            if (g->cap[e] > 1e-12 && g->level[g->to[e]] < 0) {
                g->level[g->to[e]] = g->level[u] + 1;
                g->queue[qt++] = g->to[e];
            }
    }
    return g->level[g->snk] >= 0;
}

/* One augmenting path in the level graph, found depth first without
 * recursion (the stack holds the edges of the current path). */
static double augment(struct maxflow *g)
{
    int depth = 0, u = g->src;

    for (;;) {
        if (u == g->snk) {
            double f = INFINITY;
            int k;

            for (k = 0; k < depth; k++)
                f = fmin(f, g->cap[g->stack[k]]);
            for (k = 0; k < depth; k++) {
                g->cap[g->stack[k]] -= f;
                g->cap[g->stack[k] ^ 1] += f;
            }
            return f;
        }
        while (g->iter[u] >= 0) {
            int e = g->iter[u], v = g->to[e];

            if (g->cap[e] > 1e-12 && g->level[v] == g->level[u] + 1)
                break;
            g->iter[u] = g->next[e];
        }
        if (g->iter[u] < 0) {
            /* Dead end: retreat and skip the edge that led here. */
            if (depth == 0)
                return 0;
            g->level[u] = -1;
            depth--;
            u = g->to[g->stack[depth] ^ 1];
            g->iter[u] = g->next[g->iter[u]];
            continue;
        }
        g->stack[depth++] = g->iter[u];
        u = g->to[g->iter[u]];
    }
}

double maxflow_solve(struct maxflow *g)
{
    double flow = 0, constant = g->constant;
    int i, u, qh = 0, qt = 0, e;

    for (i = 0; i < g->n; i++) {
        double c = g->unary[i];

        /* Cost c x_i: source -> i when c > 0, else c + (-c)(1 - x_i). */
        if (c > 0)
            add_edge(g, g->src, i, c, 0);
        else if (c < 0) {
            constant += c;
            add_edge(g, i, g->snk, -c, 0);
        }
    }
    g->level = G_malloc((g->n + 2) * sizeof(int));
    g->iter = G_malloc((g->n + 2) * sizeof(int));
    g->queue = G_malloc((g->n + 2) * sizeof(int));
    g->stack = G_malloc((g->n + 2) * sizeof(int));
    while (bfs(g)) {
        double f;

        memcpy(g->iter, g->head, (g->n + 2) * sizeof(int));
        while ((f = augment(g)) > 0)
            flow += f;
    }
    /* The source side: nodes reachable in the residual graph. */
    g->side = G_calloc(g->n + 2, 1);
    g->side[g->src] = 1;
    g->queue[qt++] = g->src;
    while (qh < qt) {
        u = g->queue[qh++];
        for (e = g->head[u]; e >= 0; e = g->next[e])
            if (g->cap[e] > 1e-12 && !g->side[g->to[e]]) {
                g->side[g->to[e]] = 1;
                g->queue[qt++] = g->to[e];
            }
    }
    return constant + flow;
}

int maxflow_label(const struct maxflow *g, int i) { return !g->side[i]; }
