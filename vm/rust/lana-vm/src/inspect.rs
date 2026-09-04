//! LIP-002: lazy state-dist inspection.
//!
//! A read-only serializer over the existing `STATE_DIST` DAG, mirroring
//! `lana_vm_state_dist_inspect` in `vm/c/vm.c`. It reports the structural
//! summary (node count, maximum depth, APPEND/TRANSFORM/Dirac counts,
//! exact-measurement support, sampling requirements) and emits the DAG as JSON
//! or Graphviz DOT. The output is byte-identical to the C11 reference so the
//! two runtimes can be differentially compared.

use std::fmt::Write;
use std::sync::Arc;

use lana_bytecode::LanaError;

use crate::state::{self, State};
use crate::value::{DistOperand, StateDist, StateDistKind};

/// The serialization format, matching `LanaInspectFormat` in `vm/include/vm.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InspectFormat {
    Json,
    Dot,
}

/// The accumulated structural summary of a state-dist DAG.
#[derive(Default)]
struct InspectGraph {
    nodes: Vec<Arc<StateDist>>,
    heights: Vec<usize>,
    append_count: usize,
    transform_count: usize,
    dirac_count: usize,
    sampling_required: bool,
}

impl InspectGraph {
    fn node_id(&self, node: &Arc<StateDist>) -> usize {
        self.nodes
            .iter()
            .position(|existing| Arc::ptr_eq(existing, node))
            .expect("node was visited before its id is requested")
    }
}

/// Assign a stable id to every reachable node, record each node's height (max
/// edges to a leaf), and accumulate the structural counters. Shared subgraphs
/// are visited once, so the reported node count matches the DAG, not the tree
/// expansion.
fn inspect_visit(node: &Arc<StateDist>, graph: &mut InspectGraph) -> Result<usize, LanaError> {
    if let Some(existing) = graph
        .nodes
        .iter()
        .position(|candidate| Arc::ptr_eq(candidate, node))
    {
        return Ok(existing);
    }
    let id = graph.nodes.len();
    graph.nodes.push(node.clone());
    graph.heights.push(0);
    let mut height = 0usize;
    match &node.kind {
        StateDistKind::Dirac(_) => {
            graph.dirac_count += 1;
        }
        StateDistKind::Append { left, right, sigma, .. } => {
            graph.append_count += 1;
            if *sigma != 0.0 {
                graph.sampling_required = true;
            }
            match left {
                DistOperand::Inline(_) => height = height.max(1),
                DistOperand::Node(child) => {
                    let child_id = inspect_visit(child, graph)?;
                    height = height.max(graph.heights[child_id] + 1);
                }
            }
            match right {
                DistOperand::Inline(_) => height = height.max(1),
                DistOperand::Node(child) => {
                    let child_id = inspect_visit(child, graph)?;
                    height = height.max(graph.heights[child_id] + 1);
                }
            }
        }
        StateDistKind::Transform { child, .. } => {
            graph.transform_count += 1;
            let child_id = inspect_visit(child, graph)?;
            height = graph.heights[child_id] + 1;
        }
        StateDistKind::Attenuate { child, .. } => {
            graph.transform_count += 1;
            let child_id = inspect_visit(child, graph)?;
            height = graph.heights[child_id] + 1;
        }
    }
    graph.heights[id] = height;
    Ok(id)
}

fn emit_state(out: &mut String, state: &State) {
    let _ = write!(
        out,
        "{{\"p\":{},\"d_re\":{},\"d_im\":{}}}",
        lana_bytecode::format_g(state.p),
        lana_bytecode::format_g(state.d_re),
        lana_bytecode::format_g(state.d_im),
    );
}

fn emit_operand_json(out: &mut String, graph: &InspectGraph, operand: &DistOperand) {
    match operand {
        DistOperand::Inline(state) => {
            out.push_str("{\"inline\":");
            emit_state(out, &state.state);
            out.push('}');
        }
        DistOperand::Node(node) => {
            let _ = write!(out, "{{\"node\":{}}}", graph.node_id(node));
        }
    }
}

fn emit_node_json(out: &mut String, graph: &InspectGraph, id: usize) {
    let node = &graph.nodes[id];
    let _ = write!(out, "{{\"id\":{id}");
    match &node.kind {
        StateDistKind::Dirac(state) => {
            out.push_str(",\"kind\":\"dirac\",\"state\":");
            emit_state(out, &state.state);
        }
        StateDistKind::Append { left, right, sigma, .. } => {
            out.push_str(",\"kind\":\"append\",\"left\":");
            emit_operand_json(out, graph, left);
            out.push_str(",\"right\":");
            emit_operand_json(out, graph, right);
            let _ = write!(out, ",\"sigma\":{}", lana_bytecode::format_g(*sigma));
        }
        StateDistKind::Transform { child, transform_id } => {
            let _ = write!(out, ",\"kind\":\"transform\",\"transform_id\":{transform_id},\"child\":{}", graph.node_id(child));
        }
        StateDistKind::Attenuate { child, factor } => {
            let _ = write!(out, ",\"kind\":\"attenuate\",\"factor\":{},\"child\":{}", lana_bytecode::format_g(*factor), graph.node_id(child));
        }
    }
    out.push('}');
}

fn emit_json(graph: &InspectGraph, root_id: usize) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        "{{\"kind\":\"state_dist\",\"node_count\":{},\"max_depth\":{},\"append_count\":{},\"transform_count\":{},\"dirac_count\":{},\"exact_measurement\":{},\"sampling_required\":{},\"provenance\":{{}},\"root\":{},\"nodes\":[",
        graph.nodes.len(),
        graph.heights[root_id],
        graph.append_count,
        graph.transform_count,
        graph.dirac_count,
        if graph.sampling_required { "false" } else { "true" },
        if graph.sampling_required { "true" } else { "false" },
        root_id,
    );
    for id in 0..graph.nodes.len() {
        if id > 0 {
            out.push(',');
        }
        emit_node_json(&mut out, graph, id);
    }
    out.push_str("]}");
    out
}

fn emit_dot_node(out: &mut String, graph: &InspectGraph, id: usize) {
    let node = &graph.nodes[id];
    let _ = write!(out, "  n{id} [label=\"");
    match &node.kind {
        StateDistKind::Dirac(state) => {
            let _ = write!(out, "dirac\\np={}", lana_bytecode::format_g(state.state.p));
        }
        StateDistKind::Append { sigma, .. } => {
            let _ = write!(out, "append\\nsigma={}", lana_bytecode::format_g(*sigma));
        }
        StateDistKind::Transform { transform_id, .. } => {
            let name = state::transform_spec(*transform_id)
                .map(|spec| spec.name)
                .unwrap_or("?");
            let _ = write!(out, "transform\\n{name}");
        }
        StateDistKind::Attenuate { factor, .. } => {
            let _ = write!(out, "attenuate\\nfactor={}", lana_bytecode::format_g(*factor));
        }
    }
    out.push_str("\"];\n");
}

fn emit_dot_edge(out: &mut String, graph: &InspectGraph, from: usize, label: &str, operand: &DistOperand) {
    let _ = write!(out, "  n{from} -> ");
    match operand {
        DistOperand::Inline(state) => {
            let _ = write!(
                out,
                "i{from}_{label} [label=\"{label}: state(p={})\"];\n",
                lana_bytecode::format_g(state.state.p)
            );
            let _ = write!(
                out,
                "  i{from}_{label} [label=\"state(p={})\"];\n",
                lana_bytecode::format_g(state.state.p)
            );
        }
        DistOperand::Node(node) => {
            let _ = write!(out, "n{} [label=\"{label}\"];\n", graph.node_id(node));
        }
    }
}

fn emit_dot(graph: &InspectGraph) -> String {
    let mut out = String::from("digraph state_dist {\n");
    for id in 0..graph.nodes.len() {
        emit_dot_node(&mut out, graph, id);
    }
    for id in 0..graph.nodes.len() {
        let node = &graph.nodes[id];
        match &node.kind {
            StateDistKind::Append { left, right, .. } => {
                emit_dot_edge(&mut out, graph, id, "left", left);
                emit_dot_edge(&mut out, graph, id, "right", right);
            }
            StateDistKind::Transform { child, .. } => {
                emit_dot_edge(&mut out, graph, id, "child", &DistOperand::Node(child.clone()));
            }
            StateDistKind::Attenuate { child, .. } => {
                emit_dot_edge(&mut out, graph, id, "child", &DistOperand::Node(child.clone()));
            }
            StateDistKind::Dirac(_) => {}
        }
    }
    out.push_str("}\n");
    out
}

/// Serialize a state distribution's DAG structure, mirroring
/// `lana_vm_state_dist_inspect` in `vm/c/vm.c`.
pub fn inspect(distribution: &Arc<StateDist>, format: InspectFormat) -> Result<String, LanaError> {
    let mut graph = InspectGraph::default();
    let root_id = inspect_visit(distribution, &mut graph)?;
    Ok(match format {
        InspectFormat::Json => emit_json(&graph, root_id),
        InspectFormat::Dot => emit_dot(&graph),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::StateValue;

    fn dirac(p: f64) -> Arc<StateDist> {
        let mut state = State { p: 0.0, d_re: 0.0, d_im: 0.0 };
        assert_eq!(state::make_complex(p, 0.0, 0.0, &mut state), LanaError::Ok);
        Arc::new(StateDist {
            kind: StateDistKind::Dirac(StateValue { state, indexes: Default::default() }),
        })
    }

    #[test]
    fn inspect_dirac_json() {
        let json = inspect(&dirac(0.4), InspectFormat::Json).unwrap();
        assert_eq!(
            json,
            "{\"kind\":\"state_dist\",\"node_count\":1,\"max_depth\":0,\"append_count\":0,\"transform_count\":0,\"dirac_count\":1,\"exact_measurement\":true,\"sampling_required\":false,\"provenance\":{},\"root\":0,\"nodes\":[{\"id\":0,\"kind\":\"dirac\",\"state\":{\"p\":0.4,\"d_re\":0,\"d_im\":0}}]}"
        );
    }

    #[test]
    fn inspect_append_json() {
        let left = dirac(0.2);
        let right = dirac(0.3);
        let append = Arc::new(StateDist {
            kind: StateDistKind::Append {
                left: DistOperand::Node(left),
                right: DistOperand::Node(right),
                has_cached_parameters: false,
                p: 0.0,
                m_re: 0.0,
                m_im: 0.0,
                sigma: 0.0,
            },
        });
        let json = inspect(&append, InspectFormat::Json).unwrap();
        assert!(json.contains("\"node_count\":3"));
        assert!(json.contains("\"max_depth\":1"));
        assert!(json.contains("\"append_count\":1"));
        assert!(json.contains("\"dirac_count\":2"));
    }

    #[test]
    fn inspect_dot() {
        let dot = inspect(&dirac(0.4), InspectFormat::Dot).unwrap();
        assert!(dot.starts_with("digraph state_dist {\n"));
        assert!(dot.contains("n0 [label=\"dirac\\np=0.4\"];"));
        assert!(dot.ends_with("}\n"));
    }
}
