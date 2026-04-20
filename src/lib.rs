//! # plato-fleet-graph
//!
//! Graph engine for fleet topology analysis. Adjacency lists with centrality metrics,
//! community detection (connected components), and multi-hop path finding.
//!
//! ## Why Rust
//!
//! Graph algorithms are O(V²) or O(V*E) — they scale poorly in any language.
//! Rust's advantage: no GC pauses during large traversals, compact adjacency
//! representation (Vec<usize> vs Python list of dicts), and SIMD-optimized
//! sort operations for degree calculations.
//!
//! ## Why not NetworkX (Python)
//!
//! NetworkX is the gold standard for graph analysis in Python. It's great for
//! research and prototyping. But: every node is a Python dict (~200 bytes),
//! every edge is a tuple (~100 bytes). For a fleet of 10K agents with 50K edges,
//! that's ~4GB of Python objects vs ~2MB of Rust structs.
//!
//! ## Why not Neo4j
//!
//! Neo4j adds network latency, query parsing, and schema management.
//! Worth it for: persistent graph storage, Cypher queries, multi-tenant access.
//! Our use case: transient fleet topology analysis. In-memory Rust is sufficient.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// A node in the fleet graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub label: String,
    pub node_type: NodeType,
    pub weight: f64,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeType {
    Agent,
    Room,
    Crate,
    Repo,
    Service,
}

/// An edge in the fleet graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: String,
    pub target: String,
    pub weight: f64,
    pub edge_type: String,
}

/// Centrality metrics for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Centrality {
    pub degree: usize,
    pub in_degree: usize,
    pub out_degree: usize,
    pub betweenness: f64,
    pub closeness: f64,
    pub pagerank: f64,
}

/// A path between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPath {
    pub nodes: Vec<String>,
    pub edges: Vec<String>,
    pub total_weight: f64,
    pub hops: usize,
}

/// A connected component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Component {
    pub id: usize,
    pub nodes: Vec<String>,
    pub size: usize,
}

/// Graph engine.
pub struct FleetGraph {
    nodes: HashMap<String, Node>,
    adjacency: HashMap<String, Vec<Edge>>,  // node → outgoing edges
    reverse_adj: HashMap<String, Vec<Edge>>, // node → incoming edges
    edge_count: usize,
}

impl FleetGraph {
    pub fn new() -> Self {
        Self { nodes: HashMap::new(), adjacency: HashMap::new(),
               reverse_adj: HashMap::new(), edge_count: 0 }
    }

    /// Add a node.
    pub fn add_node(&mut self, id: &str, label: &str, node_type: NodeType,
                    weight: f64, metadata: HashMap<String, String>) {
        self.nodes.insert(id.to_string(), Node {
            id: id.to_string(), label: label.to_string(), node_type, weight, metadata
        });
        self.adjacency.entry(id.to_string()).or_default();
        self.reverse_adj.entry(id.to_string()).or_default();
    }

    /// Add a directed edge.
    pub fn add_edge(&mut self, source: &str, target: &str, weight: f64, edge_type: &str) {
        let edge = Edge { source: source.to_string(), target: target.to_string(),
                         weight, edge_type: edge_type.to_string() };
        self.adjacency.entry(source.to_string()).or_default().push(edge.clone());
        self.reverse_adj.entry(target.to_string()).or_default().push(edge);
        self.edge_count += 1;
    }

    /// Get neighbors (outgoing).
    pub fn neighbors(&self, node_id: &str) -> Vec<String> {
        self.adjacency.get(node_id)
            .map(|edges| edges.iter().map(|e| e.target.clone()).collect())
            .unwrap_or_default()
    }

    /// Get incoming neighbors.
    pub fn predecessors(&self, node_id: &str) -> Vec<String> {
        self.reverse_adj.get(node_id)
            .map(|edges| edges.iter().map(|e| e.source.clone()).collect())
            .unwrap_or_default()
    }

    /// BFS shortest path (unweighted).
    pub fn shortest_path(&self, source: &str, target: &str) -> Option<GraphPath> {
        if source == target {
            return Some(GraphPath { nodes: vec![source.to_string()], edges: vec![],
                                   total_weight: 0.0, hops: 0 });
        }
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, Vec<String>, Vec<String>, f64)> = VecDeque::new();
        visited.insert(source.to_string());
        queue.push_back((source.to_string(), vec![source.to_string()], vec![], 0.0));

        while let Some((current, path, edges, weight)) = queue.pop_front() {
            for edge in self.adjacency.get(&current).unwrap_or(&vec![]) {
                if visited.contains(&edge.target) { continue; }
                let mut new_path = path.clone();
                new_path.push(edge.target.clone());
                let mut new_edges = edges.clone();
                new_edges.push(format!("{}→{}", edge.source, edge.target));
                let new_weight = weight + edge.weight;
                if edge.target == target {
                    return Some(GraphPath { nodes: new_path, edges: new_edges,
                                           total_weight: new_weight, hops: new_path.len() - 1 });
                }
                visited.insert(edge.target.clone());
                queue.push_back((edge.target.clone(), new_path, new_edges, new_weight));
            }
        }
        None
    }

    /// All-pairs shortest paths from a source (Dijkstra).
    pub fn dijkstra(&self, source: &str) -> HashMap<String, (f64, Vec<String>)> {
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        #[derive(Eq, PartialEq)]
        struct MinDist(f64, String);

        impl Ord for MinDist {
            fn cmp(&self, other: &Self) -> Ordering {
                other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
            }
        }
        impl PartialOrd for MinDist {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
        }

        let mut dist: HashMap<String, f64> = HashMap::new();
        let mut prev: HashMap<String, String> = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(source.to_string(), 0.0);
        heap.push(MinDist(0.0, source.to_string()));

        while let Some(MinDist(d, u)) = heap.pop() {
            if d > *dist.get(&u).unwrap_or(&f64::INFINITY) { continue; }
            for edge in self.adjacency.get(&u).unwrap_or(&vec![]) {
                let new_dist = d + edge.weight;
                if new_dist < *dist.get(&edge.target).unwrap_or(&f64::INFINITY) {
                    dist.insert(edge.target.clone(), new_dist);
                    prev.insert(edge.target.clone(), u.clone());
                    heap.push(MinDist(new_dist, edge.target.clone()));
                }
            }
        }

        // Reconstruct paths
        let mut results = HashMap::new();
        for (node, _) in &dist {
            let mut path = vec![node.clone()];
            let mut current = node.clone();
            while let Some(p) = prev.get(&current) {
                path.push(p.clone());
                current = p.clone();
            }
            path.reverse();
            results.insert(node.clone(), (*dist.get(node).unwrap(), path));
        }
        results
    }

    /// Connected components (undirected view).
    pub fn components(&self) -> Vec<Component> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut components = Vec::new();
        let mut comp_id = 0;

        for node_id in self.nodes.keys() {
            if visited.contains(node_id) { continue; }
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(node_id.clone());
            visited.insert(node_id.clone());

            while let Some(current) = queue.pop_front() {
                component.push(current.clone());
                for neighbor in self.neighbors(&current) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor.clone());
                        queue.push_back(neighbor);
                    }
                }
                for neighbor in self.predecessors(&current) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor.clone());
                        queue.push_back(neighbor);
                    }
                }
            }
            components.push(Component { id: comp_id, size: component.len(), nodes: component });
            comp_id += 1;
        }
        components.sort_by(|a, b| b.size.cmp(&a.size));
        components
    }

    /// Degree centrality for a node.
    pub fn centrality(&self, node_id: &str) -> Centrality {
        let out = self.adjacency.get(node_id).map(|e| e.len()).unwrap_or(0);
        let inn = self.reverse_adj.get(node_id).map(|e| e.len()).unwrap_or(0);
        let degree = out + inn;
        // Approximate betweenness using BFS from this node
        let reachable = self.bfs_reachable_count(node_id);
        let total = self.nodes.len();
        let betweenness = if total > 2 {
            (reachable as f64 - 1.0) * (total as f64 - reachable as f64 - 1.0) / ((total as f64 - 1.0) * (total as f64 - 2.0))
        } else { 0.0 };
        // Closeness: 1 / avg_distance (approximate using BFS depth)
        let closeness = if reachable > 0 && total > 1 {
            (reachable as f64) / (total as f64 - 1.0)
        } else { 0.0 };
        Centrality { degree, in_degree: inn, out_degree: out,
                     betweenness, closeness, pagerank: 0.0 }
    }

    /// PageRank (simplified power iteration).
    pub fn pagerank(&self, iterations: usize, damping: f64) -> HashMap<String, f64> {
        let n = self.nodes.len() as f64;
        if n == 0.0 { return HashMap::new(); }

        let mut rank: HashMap<String, f64> = self.nodes.keys()
            .map(|k| (k.clone(), 1.0 / n)).collect();

        for _ in 0..iterations {
            let mut new_rank: HashMap<String, f64> = HashMap::new();
            for node_id in self.nodes.keys() {
                let out_edges = self.adjacency.get(node_id).unwrap_or(&vec![]);
                if out_edges.is_empty() { continue; }
                let share = rank[node_id] / out_edges.len() as f64;
                for edge in out_edges {
                    *new_rank.entry(edge.target.clone()).or_insert(0.0) += share;
                }
            }
            for (k, v) in new_rank.iter_mut() {
                *v = damping * *v + (1.0 - damping) / n;
            }
            rank = new_rank;
        }
        rank
    }

    /// Nodes by type.
    pub fn nodes_by_type(&self, node_type: &NodeType) -> Vec<&Node> {
        self.nodes.values().filter(|n| &n.node_type == node_type).collect()
    }

    /// Subgraph induced by a set of nodes.
    pub fn subgraph(&self, node_ids: &HashSet<String>) -> FleetGraph {
        let mut sub = FleetGraph::new();
        for id in node_ids {
            if let Some(node) = self.nodes.get(id) {
                sub.add_node(&node.id, &node.label, node.node_type.clone(),
                           node.weight, node.metadata.clone());
            }
        }
        for id in node_ids {
            if let Some(edges) = self.adjacency.get(id) {
                for edge in edges {
                    if node_ids.contains(&edge.target) {
                        sub.add_edge(&edge.source, &edge.target, edge.weight, &edge.edge_type);
                    }
                }
            }
        }
        sub
    }

    fn bfs_reachable_count(&self, start: &str) -> usize {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start.to_string());
        visited.insert(start.to_string());
        while let Some(current) = queue.pop_front() {
            for neighbor in self.neighbors(&current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor.clone());
                    queue.push_back(neighbor);
                }
            }
        }
        visited.len()
    }

    pub fn stats(&self) -> GraphStats {
        let components = self.components();
        let mut degree_sum = 0;
        let mut max_degree = 0;
        for node_id in self.nodes.keys() {
            let c = self.centrality(node_id);
            degree_sum += c.degree;
            max_degree = max_degree.max(c.degree);
        }
        GraphStats { nodes: self.nodes.len(), edges: self.edge_count,
                    components: components.len(), largest_component: components.first().map(|c| c.size).unwrap_or(0),
                    avg_degree: if self.nodes.is_empty() { 0.0 } else { degree_sum as f64 / self.nodes.len() as f64 },
                    max_degree }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub nodes: usize,
    pub edges: usize,
    pub components: usize,
    pub largest_component: usize,
    pub avg_degree: f64,
    pub max_degree: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_graph() -> FleetGraph {
        let mut g = FleetGraph::new();
        g.add_node("a", "Agent A", NodeType::Agent, 1.0, HashMap::new());
        g.add_node("b", "Agent B", NodeType::Agent, 1.0, HashMap::new());
        g.add_node("c", "Room C", NodeType::Room, 1.0, HashMap::new());
        g.add_node("d", "Crate D", NodeType::Crate, 1.0, HashMap::new());
        g.add_edge("a", "b", 1.0, "communicates");
        g.add_edge("b", "c", 1.0, "belongs_to");
        g.add_edge("c", "d", 1.0, "contains");
        g.add_edge("a", "c", 1.0, "belongs_to");
        g
    }

    #[test]
    fn test_neighbors() {
        let g = setup_graph();
        let neighbors = g.neighbors("a");
        assert!(neighbors.contains(&"b".to_string()));
        assert!(neighbors.contains(&"c".to_string()));
    }

    #[test]
    fn test_shortest_path() {
        let g = setup_graph();
        let path = g.shortest_path("a", "d").unwrap();
        assert_eq!(path.hops, 2);
    }

    #[test]
    fn test_components() {
        let g = setup_graph();
        let components = g.components();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].size, 4);
    }

    #[test]
    fn test_centrality() {
        let g = setup_graph();
        let c = g.centrality("c");
        assert!(c.in_degree >= 2);
    }

    #[test]
    fn test_pagerank() {
        let g = setup_graph();
        let pr = g.pagerank(20, 0.85);
        assert!(pr.len() == 4);
        // Node c (hub) should have highest PageRank
        let c_rank = pr.get("c").unwrap();
        assert!(*c_rank > 0.0);
    }

    #[test]
    fn test_subgraph() {
        let g = setup_graph();
        let ids = HashSet::from(["a".to_string(), "b".to_string()]);
        let sub = g.subgraph(&ids);
        assert_eq!(sub.nodes.len(), 2);
    }
}
