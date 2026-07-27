"""
influence_graph.py
==================
The NETWORK half of the influential-users work: Chan (2026) chapter 5
("network and label analysis") ported to RetailRadar's own reply graph,
plus the drawing primitives the dashboard needs.

WHY A SEPARATE MODULE (and why no networkx)
-------------------------------------------
`influence.py` builds the store (who said what, how good were they).
`influence_ml.py` learns "can we spot a good voice before reading their
record". This module answers the third, purely descriptive question the
thesis spends a whole chapter on: *what does the crowd's conversation
graph actually look like, and do the influential nodes sit anywhere
special in it?* Everything here is pandas / numpy / scipy only, because
the dashboard imports it and the desk's Windows box must not need a
graph library to draw a picture. The notebook cross-checks the two
non-trivial routines (Louvain modularity, betweenness) against networkx
when it happens to be installed - see NB05.

WHAT IS PORTED FROM THE THESIS
------------------------------
5.1 network statistics  : n, m, <k>, density, L, D, C, small-world sigma,
                          closeness, betweenness  -> `network_stats`
5.1 degree distribution : linear + log-log tails  -> `degree_distribution`
5.2 communities         : Louvain modularity Q, community count,
                          inter-community edge share -> `louvain`,
                          `community_report`
5.3 label analysis      : edge and node homophily, overall and per class,
                          centrality-by-class, attributes-by-class
                          -> `homophily`, `by_class`
5.x figures             : ego networks and the k-core backbone
                          -> `ego_subgraph`, `kcore_subgraph`,
                             `spring_layout`

CONVENTIONS THAT ARE CHOICES (recorded, not hidden)
---------------------------------------------------
* SCOPE: nodes are the store's authors; an edge exists when one author
  replied to another. Reply pairs where either endpoint is not a store
  author are dropped - those users have no features, so they cannot be
  scored or drawn. This shortens paths relative to the full conversation
  graph, which is stated wherever a path statistic is reported.
* WEIGHT: edge weight = number of distinct reply records between the two
  authors, capped at MAX_EDGE_W (the same cap influence.py already uses,
  so audience counts and graph weights agree).
* DIRECTION: undirected. A reply is an interaction; the thesis treats it
  the same way for its structural statistics.
* SAMPLING: path length, closeness and betweenness are estimated from
  PIVOT_SAMPLE random source nodes (seeded, so the number is stable
  between runs). Exact all-pairs on ~12.5k nodes is 150M shortest paths -
  the estimate is reported WITH its sample size, never as an exact value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path

from analytics.influence import MAX_EDGE_W, PAGERANK_D

# --- module constants (round, and each one only affects a REPORTED
# --- estimate's precision, never a decision) ---------------------------
PIVOT_SAMPLE = 400        # source nodes for sampled path/centrality stats
LOUVAIN_PASSES = 20       # max local-moving sweeps per Louvain level
LOUVAIN_TOL = 1e-7        # stop a level when modularity gain < this
POWER_ITERS = 100         # power iterations for pagerank / eigenvector
POWER_TOL = 1e-10
LAYOUT_ITERS = 200        # Fruchterman-Reingold iterations for drawings
GRAPH_SEED = 42           # project seed, reused so drawings are stable


# ---------------------------------------------------------------------------
# the graph object
# ---------------------------------------------------------------------------
@dataclass
class Graph:
    """A weighted undirected graph as (names, symmetric CSR adjacency).

    Deliberately tiny: every routine below takes a Graph and returns
    pandas, so nothing in the codebase has to know about sparse matrices.
    """
    names: np.ndarray                      # node label per row of A
    A: sparse.csr_matrix                   # symmetric, zero diagonal
    idx: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self.idx:
            self.idx = {n: i for i, n in enumerate(self.names)}

    # --- basic quantities everything else is derived from --------------
    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def degree(self) -> np.ndarray:
        """Number of distinct neighbours (unweighted degree)."""
        binary = self.A.copy()
        binary.data = np.ones_like(binary.data)
        return np.asarray(binary.sum(axis=1)).ravel().astype(int)

    @property
    def strength(self) -> np.ndarray:
        """Sum of incident edge weights (weighted degree)."""
        return np.asarray(self.A.sum(axis=1)).ravel().astype(float)

    @property
    def m(self) -> int:
        """Number of undirected edges."""
        return int(self.degree.sum() // 2)

    def series(self, values: np.ndarray, name: str) -> pd.Series:
        return pd.Series(values, index=pd.Index(self.names, name="author"),
                         name=name)

    def subgraph(self, keep: np.ndarray) -> "Graph":
        """`keep` is a boolean mask or an array of node names."""
        if keep.dtype == bool:
            rows = np.flatnonzero(keep)
        else:
            rows = np.array([self.idx[k] for k in keep if k in self.idx])
        return Graph(names=self.names[rows],
                     A=self.A[rows][:, rows].tocsr())


def build_graph(edges: pd.DataFrame,
                nodes: pd.Index | np.ndarray | None = None,
                max_weight: int = MAX_EDGE_W) -> Graph:
    """Board-scoped weighted reply graph.

    HOW: (1) keep reply records whose BOTH endpoints are store authors,
    (2) drop self-replies (an author replying to themselves is not an
    interaction between two people), (3) collapse the remaining records
    into unordered pairs and count them -> weight, capped at
    `max_weight`, (4) write the counts symmetrically into a CSR matrix.
    """
    if nodes is None:
        nodes = pd.Index(pd.unique(pd.concat(
            [edges["replier"], edges["author"]], ignore_index=True)))
    names = np.asarray(pd.Index(nodes).dropna().unique())
    idx = {n: i for i, n in enumerate(names)}
    A = sparse.csr_matrix((len(names), len(names)), dtype=float)
    if len(edges) and len(names):
        e = edges[["replier", "author"]].dropna()
        e = e[e["replier"].isin(idx) & e["author"].isin(idx)]
        e = e[e["replier"] != e["author"]]
        if len(e):
            i = e["replier"].map(idx).to_numpy()
            j = e["author"].map(idx).to_numpy()
            lo, hi = np.minimum(i, j), np.maximum(i, j)
            pair = pd.DataFrame({"lo": lo, "hi": hi})
            w = pair.groupby(["lo", "hi"]).size().clip(upper=max_weight)
            lo = w.index.get_level_values(0).to_numpy()
            hi = w.index.get_level_values(1).to_numpy()
            val = w.to_numpy(dtype=float)
            A = sparse.coo_matrix(
                (np.concatenate([val, val]),
                 (np.concatenate([lo, hi]), np.concatenate([hi, lo]))),
                shape=(len(names), len(names))).tocsr()
            A.setdiag(0.0)
            A.eliminate_zeros()
    return Graph(names=names, A=A, idx=idx)


def edge_list(g: Graph) -> pd.DataFrame:
    """The graph's unique undirected edges as (u, v, weight) names."""
    upper = sparse.triu(g.A, k=1).tocoo()
    return pd.DataFrame({"u": g.names[upper.row], "v": g.names[upper.col],
                         "weight": upper.data})


# ---------------------------------------------------------------------------
# 5.1  centralities
# ---------------------------------------------------------------------------
def pagerank(g: Graph, damping: float = PAGERANK_D) -> pd.Series:
    """Weighted PageRank by power iteration - the same algorithm and the
    same damping influence.py already uses, so the dashboard's 'reach'
    number cannot disagree with the store's."""
    n = g.n
    if n == 0:
        return pd.Series(dtype=float, name="pagerank")
    strength = g.strength
    dangling = strength == 0
    P = g.A.multiply(1.0 / np.where(dangling, 1.0, strength)[:, None]).T
    P = sparse.csr_matrix(P)
    r = np.full(n, 1.0 / n)
    for _ in range(POWER_ITERS):
        new = damping * (P @ r + r[dangling].sum() / n) + (1 - damping) / n
        if np.abs(new - r).sum() < POWER_TOL:
            r = new
            break
        r = new
    return g.series(r, "pagerank")


def eigenvector_centrality(g: Graph) -> pd.Series:
    """Leading eigenvector of the weighted adjacency (power iteration).
    Thesis 5.1 reports it beside PageRank: PageRank rewards being replied
    to at all, eigenvector rewards being replied to by well-connected
    people."""
    n = g.n
    if n == 0:
        return pd.Series(dtype=float, name="eigenvector")
    x = np.full(n, 1.0 / np.sqrt(n))
    for _ in range(POWER_ITERS):
        y = g.A @ x
        norm = np.linalg.norm(y)
        if norm == 0:
            break
        y /= norm
        if np.abs(y - x).sum() < POWER_TOL:
            x = y
            break
        x = y
    return g.series(np.abs(x), "eigenvector")


def clustering_coefficient(g: Graph) -> pd.Series:
    """Local clustering C_i = 2*triangles_i / (k_i*(k_i-1)) on the
    UNWEIGHTED graph. HOW the triangle count is obtained without loops:
    the number of closed triples through i is the i-th diagonal entry of
    A^3 / 2, and diag(A^3) = row-sums of (A@A) elementwise-times A - much
    cheaper than forming A^3."""
    B = g.A.copy()
    B.data = np.ones_like(B.data)
    tri = np.asarray(((B @ B).multiply(B)).sum(axis=1)).ravel() / 2.0
    k = g.degree.astype(float)
    denom = k * (k - 1) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(denom > 0, tri / denom, 0.0)
    return g.series(c, "clustering")


def core_number(g: Graph) -> pd.Series:
    """k-core decomposition (peeling): repeatedly remove the lowest-degree
    node, recording the degree it had when removed. The k-core backbone in
    the thesis's figures is `core_number >= k`."""
    A = g.A.copy()
    A.data = np.ones_like(A.data)
    A = A.tolil()
    deg = np.asarray(A.sum(axis=1)).ravel().astype(int)
    core = np.zeros(g.n, dtype=int)
    alive = np.ones(g.n, dtype=bool)
    A = A.tocsr()
    level = 0
    for _ in range(g.n):
        if not alive.any():
            break
        d = np.where(alive, deg, np.iinfo(np.int32).max)
        v = int(np.argmin(d))
        level = max(level, int(deg[v]))
        core[v] = level
        alive[v] = False
        nb = A.indices[A.indptr[v]:A.indptr[v + 1]]
        for u in nb:
            if alive[u]:
                deg[u] -= 1
    return g.series(core, "core_number")


def _pivots(g: Graph, sample: int, seed: int) -> np.ndarray:
    """Random source nodes for the sampled path statistics. Isolated
    nodes are excluded: they have no finite distance to anyone and would
    only add NaNs to the estimate."""
    live = np.flatnonzero(g.degree > 0)
    if len(live) == 0:
        return live
    rng = np.random.default_rng(seed)
    if len(live) <= sample:
        return live
    return np.sort(rng.choice(live, size=sample, replace=False))


def path_stats(g: Graph, sample: int = PIVOT_SAMPLE,
               seed: int = GRAPH_SEED) -> dict:
    """Sampled average shortest-path length, eccentricity-based diameter
    lower bound, and mean closeness. Hop counts, unweighted - the thesis
    reports the same (an interaction is one hop regardless of how many
    replies it carried).

    Closeness uses the Wasserman-Faust correction (reachable fraction
    times inverse mean distance) so nodes in small components are not
    flattered - the same choice networkx makes by default.
    """
    piv = _pivots(g, sample, seed)
    if len(piv) == 0:
        return {"avg_path_len": np.nan, "diameter_lb": np.nan,
                "mean_closeness": np.nan, "pivots": 0}
    D = shortest_path(g.A, method="D", unweighted=True, indices=piv)
    finite = np.isfinite(D) & (D > 0)
    lengths = D[finite]
    n = g.n
    reach = finite.sum(axis=1)
    tot = np.where(finite, D, 0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        clos = np.where(tot > 0,
                        (reach / np.maximum(tot, 1e-12))
                        * (reach / max(n - 1, 1)), 0.0)
    ecc = np.where(finite.any(axis=1), np.where(finite, D, 0).max(axis=1), 0)
    return {"avg_path_len": float(lengths.mean()) if lengths.size else np.nan,
            "diameter_lb": int(ecc.max()) if len(ecc) else np.nan,
            "mean_closeness": float(clos.mean()),
            "pivots": int(len(piv)),
            "closeness": g.series(np.nan * np.ones(n), "closeness"),
            "closeness_pivots": pd.Series(clos, index=g.names[piv],
                                          name="closeness")}


def betweenness(g: Graph, sample: int = PIVOT_SAMPLE,
                seed: int = GRAPH_SEED) -> pd.Series:
    """Brandes betweenness from a random sample of source nodes, scaled
    to the full-source estimate and normalised by (n-1)(n-2)/2 so the
    number is comparable with the thesis's 0.0009.

    HOW Brandes works, in two sweeps per source s:
      forward  - BFS from s, recording each node's distance and its
                 number of shortest paths sigma (a node's sigma is the
                 sum of its predecessors' sigmas);
      backward - walk the BFS layers in reverse, pushing each node's
                 dependency delta back to its predecessors in proportion
                 to sigma. Summing delta over all sources gives
                 betweenness.
    """
    piv = _pivots(g, sample, seed)
    n = g.n
    bc = np.zeros(n)
    if len(piv) == 0 or n < 3:
        return g.series(bc, "betweenness")
    indptr, indices = g.A.indptr, g.A.indices
    for s in piv:
        sigma = np.zeros(n)
        sigma[s] = 1.0
        dist = np.full(n, -1, dtype=np.int64)
        dist[s] = 0
        order = []
        preds = [[] for _ in range(n)]
        frontier = [int(s)]
        while frontier:
            nxt = []
            for v in frontier:
                order.append(v)
                for w in indices[indptr[v]:indptr[v + 1]]:
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        nxt.append(int(w))
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        preds[w].append(v)
            frontier = nxt
        delta = np.zeros(n)
        for w in reversed(order):
            coeff = (1.0 + delta[w]) / sigma[w] if sigma[w] else 0.0
            for v in preds[w]:
                delta[v] += sigma[v] * coeff
            if w != s:
                bc[w] += delta[w]
    scale = (n / len(piv)) / ((n - 1) * (n - 2))    # /2 for undirected x2
    return g.series(bc * scale, "betweenness")


def centrality_table(g: Graph, sample: int = PIVOT_SAMPLE,
                     seed: int = GRAPH_SEED,
                     bc: pd.Series | None = None) -> pd.DataFrame:
    """Every 5.1 centrality in one frame, indexed by author. Closeness is
    only defined on the sampled pivots, so it is left NaN elsewhere and
    the notebook says so in the caption. Pass `bc` to reuse an already
    computed betweenness (it is by far the most expensive column)."""
    ps = path_stats(g, sample, seed)
    out = pd.concat([
        g.series(g.degree, "degree"),
        g.series(g.strength, "weighted_degree"),
        pagerank(g), eigenvector_centrality(g),
        clustering_coefficient(g), core_number(g),
        bc if bc is not None else betweenness(g, sample, seed),
    ], axis=1)
    out["closeness"] = ps["closeness_pivots"].reindex(out.index)
    return out


# ---------------------------------------------------------------------------
# 5.2  communities (own Louvain - no networkx dependency)
# ---------------------------------------------------------------------------
def as_labels(g: Graph, comm) -> np.ndarray:
    """Coerce a partition to a POSITIONAL array aligned to `g.names`.

    Every routine in here indexes a partition by matrix row number, but
    `louvain` hands back a Series keyed by author (which is what the
    notebook and the dashboard want to read). Rather than make callers
    remember which form each function needs, both are accepted and
    normalised here exactly once: a Series is reindexed onto the graph's
    own node order, anything else is taken as already positional.
    """
    if isinstance(comm, pd.Series):
        return comm.reindex(g.names).to_numpy()
    return np.asarray(comm)


def modularity(g: Graph, comm, resolution: float = 1.0) -> float:
    """Newman-Girvan Q for a partition of the WEIGHTED graph:
    Q = sum_c [ w_in(c)/W - gamma*(strength(c)/(2W))^2 ],
    where W is total edge weight. Q ~ 0 means "no better than a random
    graph with the same degrees"; the thesis reports 0.34."""
    comm = as_labels(g, comm)
    W = g.A.sum() / 2.0
    if W <= 0:
        return 0.0
    coo = sparse.triu(g.A, k=1).tocoo()
    same = comm[coo.row] == comm[coo.col]
    w_in = pd.Series(coo.data[same]).groupby(comm[coo.row[same]]).sum()
    strength = pd.Series(g.strength).groupby(comm).sum()
    q = 0.0
    for c, tot in strength.items():
        q += w_in.get(c, 0.0) / W - resolution * (tot / (2 * W)) ** 2
    return float(q)


def _louvain_one_level(A: sparse.csr_matrix, resolution: float,
                       order: np.ndarray) -> np.ndarray:
    """One Louvain level: greedy local moving until no single-node move
    raises modularity. The gain of moving node i into community c is
      dQ = w(i -> c)/W - gamma * strength(i) * strength(c) / (2 W^2)
    i.e. "edges I actually have there" minus "edges I'd expect there by
    chance". `order` is a fixed permutation so the result is
    deterministic for a given seed.
    """
    n = A.shape[0]
    strength = np.asarray(A.sum(axis=1)).ravel()
    W = strength.sum() / 2.0
    comm = np.arange(n)
    if W <= 0:
        return comm
    ctot = strength.copy()
    indptr, indices, data = A.indptr, A.indices, A.data
    for _ in range(LOUVAIN_PASSES):
        moved = 0
        for i in order:
            nb = indices[indptr[i]:indptr[i + 1]]
            wt = data[indptr[i]:indptr[i + 1]]
            if len(nb) == 0:
                continue
            ci = comm[i]
            ctot[ci] -= strength[i]
            links = {}
            for j, w in zip(nb, wt):
                if j == i:
                    continue          # a super-node's own internal weight
                links[comm[j]] = links.get(comm[j], 0.0) + w
            best_c, best_gain = ci, (links.get(ci, 0.0) / W
                                     - resolution * strength[i] * ctot[ci]
                                     / (2 * W * W))
            for c, w in links.items():
                gain = (w / W
                        - resolution * strength[i] * ctot[c] / (2 * W * W))
                if gain > best_gain + LOUVAIN_TOL:
                    best_c, best_gain = c, gain
            ctot[best_c] += strength[i]
            if best_c != ci:
                comm[i] = best_c
                moved += 1
        if moved == 0:
            break
    return comm


def louvain(g: Graph, resolution: float = 1.0,
            seed: int = GRAPH_SEED) -> pd.Series:
    """Multi-level Louvain: local moving, then collapse each community
    into a super-node and repeat, until a level stops improving. Returns
    a 0-based community id per author (largest community = 0)."""
    if g.n == 0:
        return pd.Series(dtype=int, name="community")
    rng = np.random.default_rng(seed)
    A = g.A.tocsr()
    mapping = np.arange(g.n)
    while True:
        order = rng.permutation(A.shape[0])
        comm = _louvain_one_level(A, resolution, order)
        _, comm = np.unique(comm, return_inverse=True)
        mapping = comm[mapping]
        if comm.max() + 1 == A.shape[0]:
            break
        M = sparse.csr_matrix(
            (np.ones(A.shape[0]), (comm, np.arange(A.shape[0]))),
            shape=(comm.max() + 1, A.shape[0]))
        # NOTE the diagonal is deliberately KEPT: a super-node's self-loop
        # carries its community's internal weight, which is what keeps the
        # next level's degrees (and therefore its null model) correct.
        # Zeroing it is the classic bug - it makes every later level merge
        # everything, because the chance-expectation term collapses.
        A2 = (M @ A @ M.T).tocsr()
        A2.eliminate_zeros()
        if A2.shape[0] == A.shape[0]:
            break
        A = A2
    # relabel by size so "community 0" is always the biggest
    sizes = pd.Series(mapping).value_counts()
    rank = {c: r for r, c in enumerate(sizes.index)}
    return g.series(np.array([rank[c] for c in mapping]), "community")


def community_report(g: Graph, comm: pd.Series) -> dict:
    """Thesis 5.2's three numbers: modularity, community count, and the
    share of edges that cross communities (their 67% - a high value means
    the communities are loose interest clusters, not silos)."""
    c = comm.reindex(g.names).to_numpy()
    coo = sparse.triu(g.A, k=1).tocoo()
    cross = float((c[coo.row] != c[coo.col]).mean()) if coo.nnz else np.nan
    sizes = comm.value_counts()
    return {"modularity": modularity(g, c),
            "n_communities": int(comm.nunique()),
            "inter_community_edge_share": cross,
            "largest_community_share": float(sizes.iloc[0] / len(comm))
            if len(sizes) else np.nan,
            "communities_ge_10": int((sizes >= 10).sum())}


# ---------------------------------------------------------------------------
# 5.1  the headline statistics table
# ---------------------------------------------------------------------------
def network_stats(g: Graph, sample: int = PIVOT_SAMPLE,
                  seed: int = GRAPH_SEED,
                  comm: pd.Series | None = None,
                  bc: pd.Series | None = None) -> pd.Series:
    """Thesis Table 5.1 recomputed on our graph.

    small-world sigma = (C/C_rand) / (L/L_rand) with the standard
    Erdos-Renyi references C_rand = <k>/n and L_rand = ln n / ln <k>.
    sigma >> 1 is the small-world signature (their 17.25): tight local
    clustering with global shortcuts.
    """
    n, m = g.n, g.m
    k = g.degree
    kbar = float(k.mean()) if n else np.nan
    density = (2 * m / (n * (n - 1))) if n > 1 else np.nan
    C = float(clustering_coefficient(g).mean()) if n else np.nan
    ps = path_stats(g, sample, seed)
    L = ps["avg_path_len"]
    c_rand = kbar / n if n else np.nan
    l_rand = (np.log(n) / np.log(kbar)) if (n > 1 and kbar > 1) else np.nan
    sigma = ((C / c_rand) / (L / l_rand)
             if all(np.isfinite([C, c_rand, L, l_rand]))
             and c_rand > 0 and l_rand > 0 and L > 0 else np.nan)
    ncomp, lab = connected_components(g.A, directed=False)
    giant = pd.Series(lab).value_counts().iloc[0] / n if n else np.nan
    if comm is None:
        comm = louvain(g, seed=seed)
    if bc is None:
        bc = betweenness(g, sample, seed)
    cr = community_report(g, comm)
    return pd.Series({
        "nodes": n, "edges": m, "mean_degree": kbar, "median_degree":
        float(np.median(k)) if n else np.nan, "max_degree":
        int(k.max()) if n else 0, "isolated_nodes": int((k == 0).sum()),
        "density": density, "components": int(ncomp),
        "giant_component_share": float(giant),
        "avg_clustering": C, "avg_path_len_sampled": L,
        "diameter_lower_bound": ps["diameter_lb"],
        "mean_closeness_sampled": ps["mean_closeness"],
        "mean_betweenness_sampled": float(bc.mean()),
        "small_world_sigma": sigma,
        "modularity_Q": cr["modularity"],
        "n_communities": cr["n_communities"],
        "inter_community_edge_share": cr["inter_community_edge_share"],
        "path_pivots": ps["pivots"],
    }, name="value")


def degree_distribution(g: Graph) -> pd.DataFrame:
    """P(k) for the linear and log-log panels of thesis Figure 5.1. A
    heavy right tail (a few users replied to by hundreds) is what makes
    'influence' a meaningful word here at all."""
    k = g.degree
    vc = pd.Series(k).value_counts().sort_index()
    out = pd.DataFrame({"degree": vc.index.astype(int),
                        "count": vc.to_numpy()})
    out["p"] = out["count"] / out["count"].sum()
    out["ccdf"] = out["p"][::-1].cumsum()[::-1]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5.3  label analysis
# ---------------------------------------------------------------------------
def homophily(g: Graph, labels: pd.Series) -> dict:
    """Two standard measures, both in the thesis:

    EDGE homophily  = share of edges whose endpoints share a label. It is
      dominated by the majority class in an imbalanced problem.
    NODE homophily   = average over nodes of "fraction of my neighbours
      with my label". Reported per class, because that is where the
      thesis's real finding lives: their high-predictive nodes scored
      0.08 (their neighbours are almost all ordinary users) against 0.93
      for low-predictive nodes. Low positive-class homophily is exactly
      why a neighbourhood-averaging model struggles - it is a diagnosis,
      not a bug.
    """
    y = labels.reindex(g.names).fillna(0).to_numpy()
    coo = sparse.triu(g.A, k=1).tocoo()
    edge_h = float((y[coo.row] == y[coo.col]).mean()) if coo.nnz else np.nan
    B = g.A.copy()
    B.data = np.ones_like(B.data)
    same = np.zeros(g.n)
    for cls in np.unique(y):
        mask = (y == cls).astype(float)
        got = np.asarray(B @ mask).ravel()
        same += np.where(y == cls, got, 0.0)
    k = g.degree.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        node_h = np.where(k > 0, same / np.maximum(k, 1), np.nan)
    node_h = pd.Series(node_h, index=g.names)
    per_class = {f"node_homophily_class_{int(c)}":
                 float(node_h[y == c].mean(skipna=True))
                 for c in np.unique(y)}
    return {"edge_homophily": edge_h,
            "node_homophily": float(np.nanmean(node_h)),
            **per_class,
            "node_homophily_series": node_h}


def by_class(tab: pd.DataFrame, label: str | pd.Series = "y",
             cols: list | None = None) -> pd.DataFrame:
    """Thesis Tables 5.2/5.3: mean of each attribute split by class, with
    the ratio. The ratio column is the useful one - it says which
    attributes a classifier could plausibly separate on.

    `label` is either a column of `tab` or a Series to align onto its
    index - the second form is the common one, because the centrality
    table is built from the graph while the labels live in the node table.
    """
    if isinstance(label, pd.Series):
        tab = tab.assign(_y=label.reindex(tab.index))
        label_col = "_y"
    else:
        label_col = label
    cols = cols or [c for c in tab.columns
                    if c != label_col
                    and pd.api.types.is_numeric_dtype(tab[c])]
    grp = tab.dropna(subset=[label_col]).groupby(label_col)[cols].mean().T
    grp.columns = [f"class_{int(c)}" for c in grp.columns]
    if {"class_0", "class_1"} <= set(grp.columns):
        grp["ratio_1_over_0"] = grp["class_1"] / grp["class_0"].replace(
            0, np.nan)
    return grp.sort_values(grp.columns[-1], ascending=False)


# ---------------------------------------------------------------------------
# drawings: subgraph extraction + layout (pure numpy, dashboard-safe)
# ---------------------------------------------------------------------------
def ego_subgraph(g: Graph, center: str, radius: int = 1,
                 max_nodes: int = 150) -> Graph:
    """The neighbourhood around one author, out to `radius` hops. If the
    hop ball is bigger than `max_nodes`, the highest-strength neighbours
    are kept (so the picture shows the busy part of the ego network
    rather than a random slice) - and the caller is expected to say so."""
    if center not in g.idx:
        return g.subgraph(np.array([], dtype=object))
    seen = {g.idx[center]}
    frontier = {g.idx[center]}
    for _ in range(max(radius, 0)):
        nxt = set()
        for v in frontier:
            nxt.update(g.A.indices[g.A.indptr[v]:g.A.indptr[v + 1]].tolist())
        frontier = nxt - seen
        seen |= frontier
    rows = np.array(sorted(seen))
    if len(rows) > max_nodes:
        keep_strength = g.strength[rows]
        centre_row = g.idx[center]
        order = rows[np.argsort(-keep_strength)]
        rows = np.array(sorted(set(order[:max_nodes]) | {centre_row}))
    return g.subgraph(g.names[rows])


def kcore_subgraph(g: Graph, k: int | None = None, min_nodes: int = 40,
                   max_nodes: int = 400) -> tuple[Graph, int]:
    """The k-core BACKBONE: the DEEPEST core that still has at least
    `min_nodes` members (walking k down from the maximum until the
    picture has something in it). Returns (subgraph, k). This is the
    thesis's way of drawing a 12k-node graph honestly - instead of
    thinning at random, it shows the densely interconnected heart of it,
    and every node in a k-core provably has >= k neighbours inside it."""
    core = core_number(g)
    if k is None:
        k = max(int(core.max()), 1)
        while k > 1 and int((core >= k).sum()) < min_nodes:
            k -= 1
    keep = core[core >= k].index.to_numpy()
    if len(keep) > max_nodes:
        keep = (g.series(g.strength, "s").loc[keep]
                .sort_values(ascending=False).head(max_nodes).index
                .to_numpy())
    return g.subgraph(keep), int(k)


def spring_layout(g: Graph, seed: int = GRAPH_SEED,
                  iterations: int = LAYOUT_ITERS) -> pd.DataFrame:
    """Fruchterman-Reingold force layout, vectorised, seeded.

    HOW: every pair of nodes repels with force k^2/d, every edge attracts
    with d^2/k (k = ideal spacing = sqrt(area/n)); positions move a little
    each step and the step size "cools" linearly to zero, which is what
    makes the picture settle instead of oscillating. O(n^2) per iteration,
    so this is for drawings of a few hundred nodes - never the full graph.
    """
    n = g.n
    if n == 0:
        return pd.DataFrame(columns=["x", "y"])
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-0.5, 0.5, size=(n, 2))
    if n == 1:
        return pd.DataFrame(pos, index=g.names, columns=["x", "y"])
    A = g.A.toarray()
    A = A / A.max() if A.max() > 0 else A
    k = np.sqrt(1.0 / n)
    t = 0.1
    dt = t / (iterations + 1)
    for _ in range(iterations):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        dist = np.clip(dist, 0.01, None)
        # the diagonal must be FINITE and non-zero: diff is exactly zero
        # there, so both forces vanish anyway, whereas an infinity would
        # turn 0*inf into NaN and poison the whole layout.
        np.fill_diagonal(dist, 1.0)
        rep = (k * k / dist)[:, :, None] * diff / dist[:, :, None]
        att = (A * dist / k)[:, :, None] * diff / dist[:, :, None]
        disp = (rep - att).sum(axis=1)
        length = np.clip(np.linalg.norm(disp, axis=1, keepdims=True), 0.001,
                         None)
        pos += disp / length * np.minimum(length, t)
        t -= dt
    span = np.ptp(pos, axis=0)
    pos = (pos - pos.mean(axis=0)) / np.where(span > 0, span, 1.0)
    return pd.DataFrame(pos, index=pd.Index(g.names, name="author"),
                        columns=["x", "y"])


def map_frames(g: Graph, board: pd.DataFrame,
               comm: pd.Series | None = None,
               seed: int = GRAPH_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Everything a plotly scatter needs, computed here so the dashboard
    stays a view: a node frame (x, y, size basis, community, store
    attributes) and an edge frame with per-edge endpoint coordinates."""
    pos = spring_layout(g, seed=seed)
    if comm is None:
        comm = louvain(g, seed=seed)
    nodes = pos.join(board.set_index("author"), how="left")
    nodes["community"] = comm.reindex(nodes.index).fillna(-1).astype(int)
    nodes["degree_here"] = g.series(g.degree, "d").reindex(nodes.index)
    nodes["strength_here"] = g.series(g.strength, "s").reindex(nodes.index)
    el = edge_list(g)
    el = el.join(pos.rename(columns={"x": "x0", "y": "y0"}), on="u")
    el = el.join(pos.rename(columns={"x": "x1", "y": "y1"}), on="v")
    return nodes.reset_index(), el


# ---------------------------------------------------------------------------
# "what are the influential users saying" - the dashboard's other half
# ---------------------------------------------------------------------------
def recent_calls(calls: pd.DataFrame, authors: list | np.ndarray | None = None,
                 days: int = 30, asof: pd.Timestamp | None = None,
                 ) -> pd.DataFrame:
    """The store's calls for the given authors inside a trailing window.
    Text-free by construction: `calls` never holds post bodies, only the
    extracted (ticker, direction, stance, kind) tuple."""
    if not len(calls):
        return calls
    c = calls.copy()
    c["date"] = pd.to_datetime(c["date"], errors="coerce")
    asof = pd.Timestamp(asof) if asof is not None else c["date"].max()
    c = c[c["date"].between(asof - pd.Timedelta(days=days), asof)]
    if authors is not None:
        c = c[c["author"].isin(list(authors))]
    return c.sort_values("date", ascending=False)


def _direction_sign(direction: pd.Series) -> np.ndarray:
    """+1 long / -1 short, whichever way the store spells it. The store
    writes direction as int (+1/-1); this also accepts the words in case
    a future extractor version changes the encoding, so the dashboard can
    never silently read every short as a long."""
    if pd.api.types.is_numeric_dtype(direction):
        return np.where(direction.to_numpy() < 0, -1.0, 1.0)
    s = direction.astype(str).str.upper().str.strip()
    return np.where(s.isin(["SHORT", "SELL", "BEAR", "BEARISH", "-1"]),
                    -1.0, 1.0)


def direction_label(direction) -> str:
    """The human word for one direction value - used by the dashboard so
    "LONG"/"SHORT" is spelled in exactly one place."""
    sign = _direction_sign(pd.Series([direction]))[0]
    return "LONG" if sign > 0 else "SHORT"


def _weighted_calls(calls: pd.DataFrame, board: pd.DataFrame,
                    authors: list | np.ndarray | None = None,
                    days: int = 30, asof: pd.Timestamp | None = None,
                    weight_col: str = "composite") -> pd.DataFrame:
    """The one place the influence weighting is applied.

    Every "what is the panel pushing" view below is the same four columns
    grouped differently, so they are computed ONCE here.  Two of them are
    the whole arithmetic of this module:

        den = w * |stance|        the BACKING a call carries
        num = w * |stance| * dir  the same backing, signed by direction

    w is the author's board score, |stance| their conviction (0-1) and dir
    is +1 long / -1 short.  Any influence-weighted net direction is then
    sum(num) / sum(den) over whatever slice you care about - per ticker,
    per week, per author - which is why every caller can be three lines
    long and none of them can disagree with the others about the weighting.

    Returns the windowed calls with `w`, `conv`, `dir_num`, `num`, `den`
    added, or an EMPTY frame carrying those same columns so callers can
    group without an existence check.
    """
    c = recent_calls(calls, authors, days, asof)
    if not len(c):
        cols = list(getattr(calls, "columns", [])) or [
            "author", "date", "ticker", "direction", "stance", "kind"]
        return pd.DataFrame(columns=list(dict.fromkeys(
            list(cols) + ["w", "conv", "dir_num", "num", "den"])))
    w = board.set_index("author")[weight_col].reindex(c["author"]).fillna(0.0)
    c = c.assign(w=w.to_numpy(), conv=c["stance"].abs().to_numpy(),
                 dir_num=_direction_sign(c["direction"]))
    return c.assign(num=c["w"] * c["conv"] * _direction_sign(c["direction"]),
                    den=c["w"] * c["conv"].to_numpy())


def suggestion_digest(calls: pd.DataFrame, board: pd.DataFrame,
                      authors: list | np.ndarray | None = None,
                      days: int = 30, asof: pd.Timestamp | None = None,
                      weight_col: str = "composite") -> pd.DataFrame:
    """Per ticker: what the selected voices are currently suggesting.

    The consensus number is an INFLUENCE-WEIGHTED net direction:
        consensus = sum_i w_i * dir_i * |stance_i| / sum_i w_i * |stance_i|
    with w_i the author's composite score and dir_i in {+1, -1}. It sits
    in [-1, +1]: +1 = every influential voice in the window is long this
    name with full conviction, -1 = every one is short. Weighting by
    composite is the point of the whole influence store - a call from
    someone with a record counts for more than a call from a first-timer.
    """
    c = _weighted_calls(calls, board, authors, days, asof, weight_col)
    if not len(c):
        return pd.DataFrame(columns=["ticker", "n_calls", "n_authors",
                                     "longs", "shorts", "consensus",
                                     "weighted_voices", "last_date"])
    g = c.groupby("ticker")
    out = pd.DataFrame({
        "n_calls": g.size(),
        "n_authors": g["author"].nunique(),
        "longs": g["dir_num"].apply(lambda s: int((s > 0).sum())),
        "shorts": g["dir_num"].apply(lambda s: int((s < 0).sum())),
        "consensus": g["num"].sum() / g["den"].sum().replace(0, np.nan),
        "weighted_voices": g["den"].sum(),
        "last_date": g["date"].max(),
    })
    return (out.reset_index()
            .sort_values(["weighted_voices", "n_calls"], ascending=False)
            .reset_index(drop=True))


def ticker_voices(calls: pd.DataFrame, board: pd.DataFrame,
                  authors: list | np.ndarray | None = None,
                  days: int = 30, asof: pd.Timestamp | None = None,
                  weight_col: str = "composite",
                  top: int = 6) -> pd.DataFrame:
    """Per ticker: WHO is behind the call, strongest voice first.

    `suggestion_digest` tells you a ticker's net direction and how much
    influence sits behind it, but a single number cannot answer the
    follow-up a PM always asks - *who, and how sure were they?*  This
    returns exactly that, preformatted for a chart hover: one line per
    contributing author, strongest first, capped at `top` names with an
    "and N more" tail so a crowded name does not produce a hover box
    taller than the screen.

    Returns one row per ticker with:
      voices      - "<br>"-joined lines, ready to drop into a hovertemplate
      top_author  - the single strongest voice (for compact labels)
      n_more      - how many contributors were cut from the list

    The direction word comes from `direction_label`, so LONG/SHORT is
    still spelled in exactly one place in this module.
    """
    c = _weighted_calls(calls, board, authors, days, asof, weight_col)
    if not len(c):
        return pd.DataFrame(columns=["ticker", "voices", "top_author",
                                     "n_more"])
    # One row per (ticker, author): their net lean on that name, how many
    # times they said it, and their usefulness score.  Summing dir_num
    # means an author who flip-flopped nets out toward zero rather than
    # being counted twice in opposite directions.
    per = (c.groupby(["ticker", "author"])
           .agg(lean=("dir_num", "sum"), n=("dir_num", "size"),
                weight=("w", "max"))
           .reset_index()
           .sort_values(["ticker", "weight"], ascending=[True, False]))
    # The hover quotes influence on the 0-100 board scale, not the raw
    # composite: 0.71 means nothing to a reader, "influence 84" says
    # "84% of the strongest record in the store" - see influence_index().
    top_w = float(board[weight_col].max() or 1.0) or 1.0
    rows = []
    for ticker, grp in per.groupby("ticker", sort=False):
        head = grp.head(top)
        lines = []
        for r in head.itertuples(index=False):
            word = "LONG" if r.lean > 0 else ("SHORT" if r.lean < 0
                                              else "MIXED")
            lines.append(f"{r.author} - {word}, {int(r.n)} call"
                         f"{'s' if int(r.n) != 1 else ''}, "
                         f"influence {100.0 * float(r.weight) / top_w:.0f}")
        n_more = int(len(grp) - len(head))
        if n_more:
            lines.append(f"...and {n_more} more")
        rows.append({"ticker": ticker, "voices": "<br>".join(lines),
                     "top_author": str(head.iloc[0]["author"]),
                     "n_more": n_more})
    return pd.DataFrame(rows)


def author_calls_wide(calls: pd.DataFrame, authors: list | np.ndarray,
                      days: int = 30, asof: pd.Timestamp | None = None,
                      per_author: int = 5) -> pd.DataFrame:
    """One row per (author, call) for the most recent `per_author` calls -
    the "what are they suggesting" table behind the dashboard's leaderboard.
    """
    c = recent_calls(calls, authors, days, asof)
    if not len(c):
        return pd.DataFrame(columns=["author", "date", "ticker",
                                     "direction", "stance", "kind"])
    keep = ["author", "date", "ticker", "direction", "stance", "kind"]
    return (c.groupby("author", sort=False).head(per_author)[keep]
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# MAKING THE NUMBERS MEAN SOMETHING (added 2026-07-27)
#
# The influence tab was accurate and unreadable, for one reason: two of the
# three numbers it showed had no unit a reader could hold.
#
#   * `composite` is MIN-MAX NORMALISED inside build_author_scores(), so the
#     strongest author scores ~1.0 BY CONSTRUCTION, not by being always
#     right.  Printed as "usefulness 0.987" it reads like a 98.7% hit rate.
#     It is not - it is a position in the field.  influence_index() states
#     that outright by rescaling to 0-100 where 100 IS the top of the field.
#   * `weighted_voices` is a sum of (score x conviction) over calls.  "3.42"
#     is unitless: it depends on how many people called the name AND how the
#     scores happen to be scaled.  backing_share() turns it into the name's
#     SHARE of all the influence-weighted conviction the room spent in the
#     window, in per cent - so "26%" means a quarter of everything the panel
#     said, weighted by who said it and how hard, went into one name.
#
# Both are monotone transforms - no ranking anywhere changes.  What changes
# is that the reader can say what the number means out loud.
#
# WHY SHARE AND NOT "x THE TYPICAL NAME" (measured, 2026-07-27)
# The first version of this divided backing by the MEDIAN name in the window,
# mirroring the euphoria detector's self-anchoring convention (A1's "2x its
# own 120d median").  It was measured on the real store and rejected, because
# the two cases are not alike.  A1 divides a name by ITS OWN history, which
# is a stable reference.  Dividing by the median NAME divides by whatever the
# middle of the cross-section happens to be - and the cross-section is a long
# tail: in the week ending 2026-06-28, 163 names were mentioned and the
# median one had a single call from a single author, so the median backing
# was 0.24 and MSFT came out at 141x.  Weekly maxima ran 141x, 41x, 2.7x,
# 14x, 26x on the 30-day view and 171x on the 90-day view: numbers that
# cannot be spoken, and that move with how many one-off tickers a given
# fetch happened to catch rather than with the crowd.
#
# Share has none of that.  It is bounded 0-100, it is additive (the names on
# a chart sum to a share of the whole, which is what "crowded" MEANS), and
# adding a hundred one-off names barely moves the denominator because they
# barely contribute to a sum - where they move a median a lot.  On the same
# store the shares read 26%, 7%, 5%, 4%, 3%, ... 0.2%, and the reference
# line is DERIVED rather than chosen: 100/n per cent is what every name
# would show if attention were spread evenly, so "above the line" means
# "more crowded than an even split".
# ---------------------------------------------------------------------------
def influence_index(board: pd.DataFrame, score_col: str = "composite",
                    ) -> pd.Series:
    """`composite` rescaled to 0-100, where 100 = the strongest record in
    the board handed in.  A RELATIVE scale, and deliberately so: composite
    is itself min-max normalised, so it was never an absolute accuracy and
    should not be dressed as one."""
    s = pd.to_numeric(board[score_col], errors="coerce")
    top = float(s.max()) if len(s) and pd.notna(s.max()) else 0.0
    if top <= 0:
        return pd.Series(0.0, index=board.index)
    return 100.0 * s / top


def backing_share(weighted: pd.Series) -> pd.Series:
    """Unitless backing -> PER CENT of the window's total backing.

    The one unit this tab uses for crowding.  `weighted` is a column of
    sum(influence x conviction) per name; the share is that divided by the
    column's own sum, so the names shown sum to a share of everything the
    room said.  See the block comment above for why this replaced the
    earlier "x the typical name" ratio.

    Zero total (nobody in the board said anything) returns zeros rather than
    NaN: on this tab an empty window means "no crowding", not "unknown"."""
    s = pd.to_numeric(weighted, errors="coerce").fillna(0.0)
    tot = float(s.sum()) if len(s) else 0.0
    if tot <= 0:
        return pd.Series(0.0, index=getattr(weighted, "index", None))
    return 100.0 * s / tot


def even_share(n_names: int) -> float:
    """The share every name would show if attention were spread evenly.

    A DERIVED reference line, not a chosen threshold: with n names in the
    window, an even split is 100/n per cent each.  Anything above it is more
    crowded than even, which is the whole question the chart asks."""
    n = int(n_names or 0)
    return 100.0 / n if n > 0 else float("nan")


def author_push_table(calls: pd.DataFrame, board: pd.DataFrame,
                      authors: list | np.ndarray | None = None,
                      days: int = 30, asof: pd.Timestamp | None = None,
                      weight_col: str = "composite",
                      per_author: int = 4) -> pd.DataFrame:
    """One row per author: their influence, and THE TICKERS THEY ARE PUSHING.

    This is the leaderboard the tab actually needs.  The old one showed a
    hit rate next to an influence score, which invites the one comparison
    that does not hold (the score is shrunk toward the crowd base rate, the
    hit rate is raw) - and it showed `latest_calls`, a pre-baked string of
    "LONG RDDT (2026-05-05) | ..." that ignores the window the user chose.

    Here the tickers are recomputed FROM the chosen window, netted per
    author so a flip-flop shows as MIXED rather than as two opposite calls,
    and ordered by how much conviction the author put behind each one.

    Returns author, n_calls, n_tickers, longs, shorts, pushing (a display
    string like "LONG GME, LONG AMC, SHORT TSLA"), top_ticker, top_dir.
    """
    cols = ["author", "n_calls", "n_tickers", "longs", "shorts", "pushing",
            "top_ticker", "top_dir"]
    c = _weighted_calls(calls, board, authors, days, asof, weight_col)
    if not len(c):
        return pd.DataFrame(columns=cols)
    per = (c.groupby(["author", "ticker"])
           .agg(lean=("dir_num", "sum"), n=("dir_num", "size"),
                conv=("conv", "sum"))
           .reset_index()
           .sort_values(["author", "conv", "n"], ascending=[True, False,
                                                            False]))
    per["word"] = np.where(per["lean"] > 0, "LONG",
                           np.where(per["lean"] < 0, "SHORT", "MIXED"))
    rows = []
    for author, grp in per.groupby("author", sort=False):
        head = grp.head(per_author)
        shown = [f"{r.word} {r.ticker}" for r in head.itertuples(index=False)]
        hidden = int(len(grp) - len(head))
        if hidden:
            shown.append(f"+{hidden} more")
        rows.append({
            "author": str(author),
            "n_calls": int(grp["n"].sum()),
            "n_tickers": int(len(grp)),
            "longs": int((grp["word"] == "LONG").sum()),
            "shorts": int((grp["word"] == "SHORT").sum()),
            "pushing": ", ".join(shown),
            "top_ticker": str(head.iloc[0]["ticker"]),
            "top_dir": str(head.iloc[0]["word"]),
        })
    return pd.DataFrame(rows, columns=cols)


def ticker_backers(calls: pd.DataFrame, board: pd.DataFrame, ticker: str,
                   authors: list | np.ndarray | None = None,
                   days: int = 30, asof: pd.Timestamp | None = None,
                   weight_col: str = "composite") -> pd.DataFrame:
    """WHO is behind one ticker, one row per person, strongest first.

    `ticker_voices` answers the same question as a hover string; this
    answers it as data, so the tab can DRAW it.  A hover is a dead end for
    a PM - it cannot be compared across people, cannot be sorted, and
    vanishes when the mouse moves.  A bar per person, length = their
    influence, colour = their direction, is the same information in a form
    you can read at a glance and screenshot into a note.

    Returns author, influence (0-100), lean (net sign), word (LONG / SHORT /
    MIXED), n_calls, conviction (mean |stance|), last_date.
    """
    cols = ["author", "influence", "lean", "word", "n_calls", "conviction",
            "last_date"]
    c = _weighted_calls(calls, board, authors, days, asof, weight_col)
    if not len(c):
        return pd.DataFrame(columns=cols)
    c = c[c["ticker"].astype(str) == str(ticker)]
    if not len(c):
        return pd.DataFrame(columns=cols)
    top_w = float(board[weight_col].max() or 1.0) or 1.0
    g = c.groupby("author")
    out = pd.DataFrame({
        "influence": 100.0 * g["w"].max() / top_w,
        "lean": g["dir_num"].sum(),
        "n_calls": g["dir_num"].size(),
        "conviction": g["conv"].mean(),
        "last_date": g["date"].max(),
    }).reset_index()
    out["word"] = np.where(out["lean"] > 0, "LONG",
                           np.where(out["lean"] < 0, "SHORT", "MIXED"))
    return (out[cols].sort_values("influence", ascending=False)
            .reset_index(drop=True))


def crowding_history(calls: pd.DataFrame, board: pd.DataFrame,
                     authors: list | np.ndarray | None = None,
                     days: int = 30, asof: pd.Timestamp | None = None,
                     weight_col: str = "composite", freq: str = "W",
                     tickers: list | None = None) -> pd.DataFrame:
    """Per (period, ticker): how much influence-weighted backing piled in.

    The tab had NO time axis, which is the single biggest thing missing
    from it.  "GME is the most-backed name" is a fact about a snapshot;
    "GME's backing has tripled over three weeks" is the thing a PM can act
    on, and the store has carried the dates all along.

    `backing` is sum(w * |stance|) inside the period - the same quantity
    the bubble chart's height shows, cut by week instead of pooled.  `tilt`
    is the influence-weighted net direction inside the period, so a name
    can be seen switching sides, not just getting louder.

    `share` is `backing` as a per cent of ALL the backing spent in the SAME
    period, which is the only way this chart can be read: raw backing rises
    and falls with how busy the week was, so an unnormalised line conflates
    "this name is being crowded into" with "everybody posted a lot that
    week".  `even` is the share an even split would give (100 / names that
    period), the line to read `share` against.

    Both are computed over every ticker in the period BEFORE the `tickers`
    filter, so restricting the chart to five names cannot move its own
    baseline - draw five lines or fifty, each week's denominator is the same.

    Weekly by default because the comment fetch runs about twice a week
    (see PARAMETER_REGISTER Class 7): a daily axis would mostly plot the
    ingestion cadence rather than the crowd.  Weekly is not a cure for a
    THIN week, though - the fetch budget leaves some weeks with a few dozen
    calls and others with thousands - so `n_calls` and `n_authors` come back
    per row and every chart drawn from this is expected to show them, rather
    than quietly dropping thin weeks behind a cut-off.
    """
    cols = ["period", "ticker", "backing", "share", "even", "tilt",
            "n_calls", "n_authors"]
    c = _weighted_calls(calls, board, authors, days, asof, weight_col)
    if not len(c):
        return pd.DataFrame(columns=cols)
    c = c.assign(period=pd.to_datetime(c["date"]).dt.to_period(freq)
                 .dt.end_time.dt.normalize())
    g = c.groupby(["period", "ticker"])
    out = pd.DataFrame({
        "backing": g["den"].sum(),
        "tilt": g["num"].sum() / g["den"].sum().replace(0, np.nan),
        "n_calls": g.size(),
        "n_authors": g["author"].nunique(),
    }).reset_index()
    tot = out.groupby("period")["backing"].transform("sum")
    out["share"] = 100.0 * out["backing"] / tot.replace(0, np.nan)
    out["even"] = 100.0 / out.groupby("period")["ticker"].transform("nunique")
    if tickers is not None:
        out = out[out["ticker"].isin(list(tickers))]
    return out[cols].sort_values(["ticker", "period"]).reset_index(drop=True)


def panel_tilt_history(calls: pd.DataFrame, board: pd.DataFrame,
                       authors: list | np.ndarray | None = None,
                       days: int = 30, asof: pd.Timestamp | None = None,
                       weight_col: str = "composite", freq: str = "W",
                       ) -> pd.DataFrame:
    """The whole panel on one line: net direction per period, and how much
    was said.  Computed over EVERY call in the window, not just the tickers
    a chart happens to draw, so it answers "is the room turning bullish?"
    without a top-N cut deciding the answer."""
    cols = ["period", "tilt", "backing", "n_calls", "n_authors", "n_tickers"]
    c = _weighted_calls(calls, board, authors, days, asof, weight_col)
    if not len(c):
        return pd.DataFrame(columns=cols)
    c = c.assign(period=pd.to_datetime(c["date"]).dt.to_period(freq)
                 .dt.end_time.dt.normalize())
    g = c.groupby("period")
    out = pd.DataFrame({
        "tilt": g["num"].sum() / g["den"].sum().replace(0, np.nan),
        "backing": g["den"].sum(),
        "n_calls": g.size(),
        "n_authors": g["author"].nunique(),
        "n_tickers": g["ticker"].nunique(),
    }).reset_index()
    return out[cols].sort_values("period").reset_index(drop=True)
