# Architecture: plato-fleet-graph

## Language Choice: Rust

### Why Rust

Graph algorithms are computationally intensive. Fleet topology analysis with
10K nodes and 50K edges means O(V*E) operations per analysis pass.

| Metric | Python (NetworkX) | Rust (adjacency lists) |
|--------|-------------------|----------------------|
| Memory per node | ~200 bytes (dict) | ~48 bytes (struct) |
| BFS 10K nodes | ~15ms | ~0.8ms |
| PageRank 10 iterations | ~200ms | ~12ms |
| Memory 10K nodes | ~2MB | ~480KB |

### Why not NetworkX

NetworkX is excellent for research. But every node is a Python dict with
hash table overhead. For fleet-scale topology (10K+ agents), this becomes
a memory bottleneck. Rust's HashMap<String, Vec<Edge>> is 4x more compact.

### Why not Neo4j

Neo4j adds: network latency (~5ms/query), query parsing (Cypher), connection
pooling, and schema management. Worth it for persistent, multi-tenant graph DB.
Our use case: transient fleet analysis — build graph, analyze, discard.

### Architecture

```
FleetGraph {
    nodes: HashMap<String, Node>
    adjacency: HashMap<String, Vec<Edge>>     // outgoing
    reverse_adj: HashMap<String, Vec<Edge>>   // incoming
}

Algorithms:
    BFS → shortest_path (unweighted)
    Dijkstra → all-pairs weighted distances
    Union-Find → connected components
    Power iteration → PageRank
    BFS depth → betweenness/closeness centrality
```

### Future: Parallel Algorithms

With rayon, we can parallelize:
- Multi-source BFS (reachability from all agents simultaneously)
- PageRank power iteration (vector operations on GPU)
- Community detection (Louvain algorithm)
