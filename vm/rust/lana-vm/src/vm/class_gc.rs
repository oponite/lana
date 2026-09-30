//! Trace heap graphs independently of execution; Arc counts identify external roots.
use super::*;
use crate::value::{ClassObject, ClassReference, Generator, ObjectValue};
use std::sync::Weak;
use std::sync::atomic::{AtomicU8, AtomicUsize};
use std::cell::{Cell, RefCell};

thread_local! {
    static DRAINING_OWNERS: Cell<bool> = const { Cell::new(false) };
    // Initialized before the primary queue, so it remains available during
    // primary TLS destruction and receives child owners released there.
    static ENDING_OWNERS: RefCell<Option<Arc<RootOwner>>> = const { RefCell::new(None) };
    static RETIRED_OWNERS: RefCell<RetiredOwners> = const { RefCell::new(RetiredOwners(None)) };
}

struct RetiredOwners(Option<Arc<RootOwner>>);
impl Drop for RetiredOwners {
    fn drop(&mut self) {
        let _drain = RootOwner::defer_retired();
        loop {
            let owner = self.0.take().or_else(|| ENDING_OWNERS.with(|head| head.borrow_mut().take()));
            let Some(owner) = owner else { break; };
            self.0 = owner.next_retired.lock().unwrap().take();
            owner.queued.store(false, Ordering::Release);
            let _ = owner.collect();
        }
    }
}

pub(super) struct OwnerDrain { previous: bool }
impl Drop for OwnerDrain {
    fn drop(&mut self) { DRAINING_OWNERS.with(|active| active.set(self.previous)); }
}

/// An embedding root. Clone this handle, rather than its borrowed Value, when
/// retaining a result beyond the lifetime of the VM that produced it.
#[derive(Debug, Clone)]
pub struct RootedValue {
    pub(crate) value: Value,
    // Drop the lease before the final strong owner reference.
    _lease: Option<Arc<RootLease>>,
    _owner: Arc<RootOwner>,
}

#[derive(Debug)]
struct RootLease {
    owner: Weak<RootOwner>,
    id: u64,
    _allocation: Option<Arc<crate::heap::Reservation>>,
}

impl Drop for RootLease {
    fn drop(&mut self) {
        let Some(owner) = self.owner.upgrade() else { return; };
        if self.id == 0 { return; }
        if let Ok(mut roots) = owner.roots.try_lock() {
            if let Some(root) = roots.values.iter_mut().find(|root| root.id == self.id) { root.id = 0; }
        }
        owner.heap.bump_mutation_epoch();
        if owner.retired.load(Ordering::Acquire) { owner.collect_retired(); }
    }
}

impl RootedValue {
    /// Inspect a retained value without exposing clonable internal graph edges.
    pub fn print(&self) -> String { self.value.print() }
    pub fn type_name(&self) -> &'static str { self.value.type_name() }
    pub fn value_type(&self) -> ValueType { self.value.value_type() }
    pub fn as_number(&self) -> f64 { self.value.as_number() }
    pub fn as_bool(&self) -> bool { self.value.as_bool() }
    pub fn as_string(&self) -> Arc<str> { self.value.as_string() }

    /// A child retained independently of its parent, using the same heap owner.
    pub fn child(&self, index: usize) -> Result<Option<Self>, LanaError> {
        if let ValueKind::Task(task) = &self.value.kind {
            if index != 0 { return Ok(None); }
            let state = task.state.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
            if !state.completed { return Ok(None); }
            if state.status != LanaError::Ok { return Err(state.status); }
            if !state.joined { return Ok(state.result_root.clone()); }
            let value = state.result.clone();
            drop(state);
            return self._owner.root(value).map(Some);
        }
        self.value.inspection_child(index).map(|value| self._owner.root(value)).transpose()
    }

    /// Render a distribution while retaining its managed owner.
    pub fn inspect_state_dist(&self, format: crate::InspectFormat) -> Result<String, LanaError> {
        match &self.value.kind {
            ValueKind::StateDist(value) => crate::inspect(value, format),
            _ => Err(LanaError::Type),
        }
    }
}

impl Drop for RootedValue {
    fn drop(&mut self) {
        self.value = Value::null();
    }
}

#[derive(Debug)]
struct RootRegistry {
    values: Buffer<RootRecord>,
    next_id: u64,
}

#[derive(Debug)]
struct RootRecord {
    id: u64,
    value: Value,
    _allocation: Arc<crate::heap::Reservation>,
    lease: Weak<RootLease>,
}

impl RootRecord {
    fn live(&self) -> bool { self.id != 0 && self.lease.strong_count() != 0 }
}

#[derive(Debug, Default)]
struct RetainedHeap {
    classes: Vec<Arc<ClassObject>>,
    _classes_reservation: Option<Arc<Mutex<crate::heap::Reservation>>>,
    cycles: Option<Buffer<Arc<crate::heap::CycleSlot>>>,
}

#[derive(Debug)]
pub(crate) struct RootOwner {
    heap: Heap,
    retired: AtomicBool,
    retained: Mutex<RetainedHeap>,
    workspace: Mutex<Workspace>,
    retention_workspace: Mutex<Workspace>,
    roots: Mutex<RootRegistry>,
    active_collection: AtomicU8,
    last_slice_work: AtomicUsize,
    admitted_nodes: AtomicUsize,
    admitted_edges: AtomicUsize,
    queued: AtomicBool,
    next_retired: Mutex<Option<Arc<RootOwner>>>,
    _allocation: crate::heap::Reservation,
}

impl RootOwner {
    fn enqueue_retired(self: &Arc<Self>) {
        if !self.queued.swap(true, Ordering::AcqRel) {
            DRAINING_OWNERS.with(|_| {});
            ENDING_OWNERS.with(|_| {});
            let queued = RETIRED_OWNERS.try_with(|head| {
                let mut head = head.borrow_mut();
                *self.next_retired.lock().unwrap() = head.0.take();
                head.0 = Some(self.clone());
            });
            if queued.is_err() {
                ENDING_OWNERS.with(|head| {
                    let mut head = head.borrow_mut();
                    *self.next_retired.lock().unwrap() = head.take();
                    *head = Some(self.clone());
                });
            }
        }
    }

    pub(super) fn defer_retired() -> OwnerDrain {
        OwnerDrain { previous: DRAINING_OWNERS.with(|active| active.replace(true)) }
    }

    pub(super) fn drain_retired(charge: &mut dyn FnMut(u64) -> Result<(), LanaError>) -> Result<(), LanaError> {
        let _drain = Self::defer_retired();
        ENDING_OWNERS.with(|_| {});
        loop {
            let owner = RETIRED_OWNERS.with(|head| {
                let mut head = head.borrow_mut();
                let owner = head.0.take()?;
                head.0 = owner.next_retired.lock().unwrap().take();
                Some(owner)
            });
            let Some(owner) = owner else { return Ok(()); };
            owner.queued.store(false, Ordering::Release);
            match owner.collect_charged(charge, &mut || {}) {
                Ok(true) => {},
                result => {
                    owner.enqueue_retired();
                    return Err(result.err().unwrap_or(LanaError::UnsupportedOperation));
                }
            }
        }
    }

    fn collect_retired(self: &Arc<Self>) {
        self.enqueue_retired();
        if DRAINING_OWNERS.with(Cell::get) { return; }
        let _ = Self::drain_retired(&mut |_| Ok(()));
    }

    pub(super) fn new(heap: &Heap) -> Arc<Self> {
        let allocation = heap.reserve(std::mem::size_of::<Self>() + 2 * std::mem::size_of::<usize>())
            .expect("default heap admits collector owner");
        let owner = Arc::new(Self { heap: heap.clone(), retired: AtomicBool::new(false),
            retained: Mutex::new(RetainedHeap::default()), workspace: Mutex::new(Workspace::new(heap)),
            retention_workspace: Mutex::new(Workspace::new(heap)),
            roots: Mutex::new(RootRegistry { values: Buffer::new(heap, 0, 0).expect("empty root registry"), next_id: 1 }),
            active_collection: AtomicU8::new(0), last_slice_work: AtomicUsize::new(0),
            admitted_nodes: AtomicUsize::new(0), admitted_edges: AtomicUsize::new(0),
            queued: AtomicBool::new(false), next_retired: Mutex::new(None), _allocation: allocation });
        heap.set_collector(&owner);
        owner
    }

    pub(crate) fn admit(&self, nodes: usize, edges: usize) -> Result<(), LanaError> {
        self.admitted_nodes.fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| old.checked_add(nodes)).map_err(|_| LanaError::Oom)?;
        if self.admitted_edges.fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| old.checked_add(edges)).is_err() {
            self.admitted_nodes.fetch_sub(nodes, Ordering::AcqRel);
            return Err(LanaError::Oom);
        }
        let result = (|| {
            let mut workspace = self.workspace.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
            let mut retention = self.retention_workspace.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
            let nodes = self.admitted_nodes.load(Ordering::Acquire);
            let edges = self.admitted_edges.load(Ordering::Acquire);
            workspace.admit(&self.heap, nodes, edges)?;
            retention.admit(&self.heap, nodes, edges)
        })();
        if result.is_err() { self.release_admission(nodes, edges); }
        result
    }

    pub(crate) fn release_admission(&self, nodes: usize, edges: usize) {
        self.admitted_nodes.fetch_sub(nodes, Ordering::AcqRel);
        self.admitted_edges.fetch_sub(edges, Ordering::AcqRel);
        if self.admitted_nodes.load(Ordering::Acquire) == 0 && self.admitted_edges.load(Ordering::Acquire) == 0 {
            for workspace in [&self.workspace, &self.retention_workspace] {
                if let Ok(mut workspace) = workspace.try_lock() {
                    if workspace.progress.is_none() && workspace.graph.nodes.is_empty() {
                        *workspace = Workspace::new(&self.heap);
                    }
                }
            }
        }
    }

    fn root(self: &Arc<Self>, value: Value) -> Result<RootedValue, LanaError> {
        if has_gc_graph(&value) {
            let mut workspace = self.retention_workspace.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
            workspace.clear();
            let result = (|| {
                let mut context = Context { heap: &self.heap, charge: &mut |_| Ok(()) };
                workspace.graph.value(&value, &mut context)?;
                for index in (0..self.heap.cycle_count()).rev() {
                    if let Some(slot) = self.heap.cycle_slot_at(index) {
                        if let Some(node) = cycle_node(&slot) { workspace.graph.add_cycle(node, false, Some(slot), &mut context)?; }
                    }
                }
                let mut index = 0;
                while index < workspace.graph.nodes.len() {
                    let node = workspace.graph.nodes[index].node.clone();
                    let start = workspace.graph.edges.len();
                    if !workspace.graph.trace(&node, &mut context)? { return Err(LanaError::UnsupportedOperation); }
                    workspace.graph.nodes[index].edges = start..workspace.graph.edges.len();
                    index += 1;
                }
                let nodes = workspace.graph.nodes.len();
                workspace.pending.reserve(nodes)?;
                for entry in &workspace.graph.nodes {
                    if let Node::Storage(storage) = &entry.node {
                        if !storage._allocation.heap().same_heap(&self.heap) { return Err(LanaError::Task); }
                        if !storage.initialized.load(Ordering::Acquire) { return Err(LanaError::UnsupportedOperation); }
                    }
                    if entry.cycle.is_none() && !matches!(entry.node, Node::Class(_) | Node::Storage(_)) {
                        return Err(LanaError::Task);
                    }
                }
                Ok(())
            })();
            workspace.clear();
            drop(workspace);
            result?;
            return self.register_root(value, 0, 0);

        }
        self.register_root(value, 0, 0)
    }

    pub(super) fn register_root(self: &Arc<Self>, value: Value, foreign: usize, extra_edges: usize) -> Result<RootedValue, LanaError> {
        if !has_gc_graph(&value) {
            return Ok(RootedValue { value, _lease: None, _owner: self.clone() });
        }
        let mut allocation = self.heap.reserve(std::mem::size_of::<RootLease>() + std::mem::size_of::<crate::heap::Reservation>() + 4 * std::mem::size_of::<usize>())?;
        allocation.admit_gc(foreign, extra_edges.checked_add(5).ok_or(LanaError::Oom)?)?;
        let allocation = Arc::new(allocation);
        let mut roots = self.roots.lock().unwrap();
        roots.values.reserve(1)?;
        let id = roots.next_id;
        roots.next_id = roots.next_id.checked_add(1).ok_or(LanaError::Oom)?;
        let lease = Arc::new(RootLease { owner: Arc::downgrade(self), id, _allocation: Some(allocation.clone()) });
        roots.values.push(RootRecord { id, value: value.clone(), _allocation: allocation, lease: Arc::downgrade(&lease) })?;
        self.heap.bump_mutation_epoch();
        drop(roots);
        Ok(RootedValue { value, _lease: Some(lease), _owner: self.clone() })
    }

    pub(super) fn collection_major(&self) -> Option<bool> {
        match self.active_collection.load(Ordering::Acquire) { 1 => Some(false), 2 => Some(true), _ => None }
    }

    #[cfg(test)]
    pub(super) fn last_slice_work(&self) -> usize { self.last_slice_work.load(Ordering::Acquire) }

    fn collect(&self) -> Result<bool, LanaError> {
        self.collect_before_mark(&mut || {})
    }

    fn collect_before_mark(&self, before_mark: &mut dyn FnMut()) -> Result<bool, LanaError> {
        self.collect_charged(&mut |_| Ok(()), before_mark)
    }

    fn collect_charged(&self, charge: &mut dyn FnMut(u64) -> Result<(), LanaError>, before_mark: &mut dyn FnMut()) -> Result<bool, LanaError> {
        let Ok(mut retained) = self.retained.try_lock() else { return Ok(false); };
        let Ok(mut routine) = self.workspace.try_lock() else { return Ok(false); };
        let Ok(mut roots) = self.roots.try_lock() else { return Ok(false); };
        if retained.classes.is_empty() && self.heap.cycle_count() == 0 && roots.values.is_empty()
            && routine.graph.nodes.is_empty() && routine.graph.pending_add.is_none() && routine.graph.pending_value.is_none() {
            self.active_collection.store(0, Ordering::Release);
            retained._classes_reservation = None;
            if let Ok(mut workspace) = self.retention_workspace.try_lock() { *workspace = Workspace::new(&self.heap); }
            before_mark();
            return Ok(true);
        }
        // Shutdown reuses workspace already admitted to the original heap.
        // A larger graph still needs checked admission before any edge changes.
        let Ok(mut workspace) = self.retention_workspace.try_lock() else { return Ok(false); };
        workspace.clear();
        let mut context = Context { heap: &self.heap, charge };
        let result = match workspace.graph.seed_from(&routine.graph, &mut context) {
            Ok(()) => collect(&mut retained.classes, &mut roots.values, &mut context,
                &mut workspace, true, &mut || { routine.clear(); before_mark(); }),
            Err(error) => { workspace.clear(); Err(error) }
        };
        self.active_collection.store(routine.progress.as_ref().map_or(0, |progress| if progress.major { 2 } else { 1 }), Ordering::Release);
        let complete = result?;
        if complete {
            if let Some(cycles) = &mut retained.cycles {
                cycles.retain(|slot| slot.node.lock().unwrap().as_ref().is_some_and(crate::heap::CycleWeak::alive));
                if cycles.is_empty() { retained.cycles = None; }
            }
            if retained.classes.is_empty() {
                retained.classes = Vec::new();
                retained._classes_reservation = None;
            }
            if self.admitted_nodes.load(Ordering::Acquire) == 0 && self.admitted_edges.load(Ordering::Acquire) == 0 {
                *routine = Workspace::new(&self.heap);
                *workspace = Workspace::new(&self.heap);
            }
        }
        Ok(complete)
    }
}

pub(super) fn has_gc_graph(value: &Value) -> bool {
    value.derivation.is_some() || value.reactive.is_some() || value.claim.is_some() || value.planned_effect.is_some() || matches!(value.kind,
        ValueKind::ClassObject(_) | ValueKind::Array(_) | ValueKind::Map(_) | ValueKind::Set(_)
        | ValueKind::Joint(_) | ValueKind::Kernel(_) | ValueKind::Network(_) | ValueKind::Possibility(_)
        | ValueKind::PathSet(_) | ValueKind::Adt(_) | ValueKind::ObjectValue(_) | ValueKind::Generator(_)
        | ValueKind::Future(_) | ValueKind::TrainingResult(_) | ValueKind::Posterior(_) | ValueKind::Dataset(_) | ValueKind::Task(_) | ValueKind::StateDist(_))
}

impl Drop for RootOwner {
    fn drop(&mut self) {
        let empty = {
            let retained = self.retained.get_mut().unwrap();
            let roots = self.roots.get_mut().unwrap();
            let workspace = self.workspace.get_mut().unwrap();
            retained.classes.is_empty() && retained.cycles.is_none() && roots.values.is_empty()
                && workspace.graph.nodes.is_empty() && workspace.progress.is_none()
        };
        if !empty { let _ = self.collect(); }
        self.heap.collect_strings();
        self.heap.clear_collector(self as *const Self);
    }
}

impl Vm<'_> {
    pub(super) fn prepare_container_write(&mut self, owner: &Value, value: &Value) -> Result<(), LanaError> {
        let stable = match &owner.kind {
            ValueKind::Array(array) => {
                let array = array.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
                if array.frozen { return Err(LanaError::UnsupportedOperation); }
                array.cycle.generation() == crate::heap::Generation::StableShared
            }
            ValueKind::Map(map) => {
                let map = map.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
                if map.frozen { return Err(LanaError::UnsupportedOperation); }
                map.cycle.generation() == crate::heap::Generation::StableShared
            }
            _ => return Err(LanaError::Type),
        };
        if stable { self.promote_shared_graph(value)?; }
        Ok(())
    }

    pub(super) fn promote_shared_graph(&mut self, value: &Value) -> Result<(), LanaError> {
        let heap = self.heap.clone();
        let owner = self.host_roots.clone();
        let mut workspace = owner.retention_workspace.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
        let Workspace { graph, .. } = &mut *workspace;
        let result = (|| {
            let mut context = Context { heap: &heap,
                charge: &mut |steps| self.charge_bounded_work(steps) };
            graph.value(value, &mut context)?;
            let mut index = 0;
            while index < graph.nodes.len() {
                let node = graph.nodes[index].node.clone();
                if !graph.trace(&node, &mut context)? { return Err(LanaError::UnsupportedOperation); }
                index += 1;
            }
            for cycle_index in 0..heap.cycle_count() {
                self.charge_bounded_work(1)?;
                if !graph.index.is_empty() {
                    if let (Some(node), Some(cycle)) = (heap.cycle_node(cycle_index), heap.cycle_slot_at(cycle_index)) {
                    let hash_slot = Graph::slot(&graph.index, node.identity());
                    if let Some((_, index)) = graph.index[hash_slot] {
                        graph.nodes[index].cycle = Some(cycle);
                    }
                    }
                }
            }
            for entry in &graph.nodes {
                if let Node::Storage(storage) = &entry.node {
                    storage.generation.fetch_max(crate::heap::Generation::StableShared as u8, Ordering::AcqRel);
                }
                if let Some(cycle) = &entry.cycle {
                    cycle.promote(crate::heap::Generation::StableShared);
                }
            }
            Ok(())
        })();
        workspace.clear();
        result
    }

    pub(super) fn rooted_result(&self) -> Result<RootedValue, LanaError> {
        self.host_roots.root(self.result.clone())
    }

    /// Retain a host-call value beyond this VM. Foreign mutable graphs require
    /// an explicit task transfer; an incomplete or locked graph is not retained.
    pub fn retain_value(&mut self, value: &Value) -> Result<RootedValue, LanaError> {
        self.retain_value_impl(value, false)
    }

    pub(super) fn retain_value_impl(&mut self, value: &Value, internal: bool) -> Result<RootedValue, LanaError> {
        if !internal && !self.constructions.is_empty() { return Err(LanaError::UnsupportedOperation); }
        let heap = self.heap.clone();
        let owner = self.host_roots.clone();
        let mut workspace = owner.retention_workspace.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
        let Workspace { graph, pending, .. } = &mut *workspace;
        let result = (|| {
            {
                let mut context = Context { heap: &heap,
                    charge: &mut |steps| self.charge_bounded_work(steps) };
                graph.value(value, &mut context)?;
                let mut index = 0;
                while index < graph.nodes.len() {
                    let node = graph.nodes[index].node.clone();
                    let start = graph.edges.len();
                    if !graph.trace(&node, &mut context)? { return Err(LanaError::UnsupportedOperation); }
                    graph.nodes[index].edges = start..graph.edges.len();
                    index += 1;
                }
            }
            // Mark local registered nodes using the existing budgeted index.
            if !graph.index.is_empty() {
                for index in 0..heap.cycle_count() {
                    self.charge_bounded_work(1)?;
                    if let Some(node) = heap.cycle_node(index) {
                        let slot = Graph::slot(&graph.index, node.identity());
                        if let Some((_, entry)) = graph.index[slot] {
                            graph.nodes[entry].arena = true;
                            graph.nodes[entry].cycle = heap.cycle_slot_at(index);
                        }
                    }
                }
            }
            for entry in &graph.nodes {
                self.charge_bounded_work(1)?;
                match &entry.node {
                    Node::Class(reference) if !Arc::ptr_eq(&reference.owner, &self.class_owner) => return Err(LanaError::Task),
                    Node::Storage(storage) if !storage.initialized.load(Ordering::Acquire) => return Err(LanaError::UnsupportedOperation),
                    Node::Array(_) | Node::Map(_) | Node::Set(_) | Node::Reactive(_)
                    | Node::Generator(_) | Node::Future(_) | Node::Effect(_) | Node::Task(_) if !entry.arena => return Err(LanaError::Task),
                    _ => {},
                }
            }
            for entry in &graph.nodes {
                if entry.cycle.is_none() && !matches!(entry.node, Node::Class(_) | Node::Storage(_)) {
                    return Err(LanaError::Task);
                }
            }
            pending.reserve(graph.nodes.len())?;
            Ok(())
        })();
        workspace.clear();
        drop(workspace);
        result?;
        self.host_roots.register_root(value.clone(), 0, 0)
    }
}

impl Drop for Vm<'_> {
    fn drop(&mut self) {

        if self.scheduler_owner {
            if let Some(scheduler) = &self.scheduler {
                scheduler.shutdown();
                // Queued children retain the scheduler. With zero workers they
                // never run, so detach the queue before dropping those children.
                let queued = std::mem::take(&mut scheduler.state.lock().unwrap().queue);
                drop(queued);
            }
        }
        let mut retained = self.host_roots.retained.lock().unwrap();
        retained.classes = std::mem::take(&mut self.class_objects);
        retained._classes_reservation = Some(self.class_objects_reservation.clone());
        retained.cycles = Some(std::mem::replace(&mut self.owned_cycles,
            Buffer::new(&self.heap, 0, 0).expect("empty collector registry")));
        drop(retained);
        self.host_roots.retired.store(true, Ordering::Release);
        // Pin managed graphs before execution roots release their final owners.
        // Retained embedding roots remain live during the subsequent marking.
        let owner = self.host_roots.clone();
        let _ = owner.collect_before_mark(&mut || {
            self.host_call_extension = None;
            self.frames.clear();
            self.result = Value::null();
            self.path_execution.clear();
            self.constructing.clear();
            self.constructions.clear();
            self.ready_futures.clear();
            self.awaiters.clear();
            self.shared_references.clear();
            self.tasks.clear();
            self.scheduler = None;
        });
        let _ = RootOwner::drain_retired(&mut |_| Ok(()));
    }
}

macro_rules! nodes {
    ($($name:ident($ty:ty)),+ $(,)?) => {
        #[derive(Debug, Clone)]
        pub(crate) enum Node { Vacant, $($name(Arc<$ty>)),+ }
        impl Node {
            fn identity(&self) -> usize {
                match self { Self::Vacant => 0, $(Self::$name(value) => Arc::as_ptr(value) as usize),+ }
            }
        fn references(&self) -> usize {
                match self { Self::Vacant => 0, $(Self::$name(value) => Arc::strong_count(value)),+ }
            }
            fn has_write_barrier(&self) -> bool {
                matches!(self, Self::Array(_) | Self::Map(_) | Self::Set(_)
                    | Self::Joint(_) | Self::Kernel(_) | Self::Network(_) | Self::Possibility(_)
                    | Self::Paths(_) | Self::Adt(_) | Self::Object(_) | Self::Training(_)
                    | Self::Posterior(_) | Self::Dataset(_) | Self::Claim(_) | Self::StateDist(_) | Self::Derivation(_)
                    | Self::Generator(_) | Self::Future(_) | Self::Reactive(_) | Self::Effect(_) | Self::Task(_))
            }
            fn is_young(&self) -> bool {
                match self {
                    Self::Storage(value) => value.generation.load(Ordering::Acquire) == crate::heap::Generation::Young as u8,
                    _ => false,
                }
            }
        }
    };
}

nodes! {
    Class(ClassReference), Storage(ClassObject), Array(Mutex<Array>), Map(Mutex<Map>),
    Set(Mutex<Set>), Joint(JointState), Kernel(FiniteKernel), Network(FiniteNetwork),
    Possibility(Possibility), Paths(PathSet), Adt(Adt), Object(ObjectValue),
    Generator(Mutex<Generator>), Future(Mutex<Future>), Training(TrainingResult),
    Posterior(Posterior), Dataset(Dataset), Reactive(Mutex<Reactive>), Claim(Claim),
    Effect(PlannedEffect), Task(Task), StateDist(StateDist), Derivation(Derivation),
}

#[derive(Debug)]
struct Entry {
    node: Node,
    cycle: Option<Arc<crate::heap::CycleSlot>>,
    incoming: usize,
    arena: bool,
    live: bool,
    edges: std::ops::Range<usize>,
    trace_cursor: usize,
    trace_outer: usize,
    trace_inner: usize,
    trace_phase: u8,
    trace_started: bool,
}

struct Context<'a> {
    heap: &'a Heap,
    charge: &'a mut dyn FnMut(u64) -> Result<(), LanaError>,
}

#[derive(Debug)]
struct Graph {
    nodes: Buffer<Entry>,
    edges: Buffer<usize>,
    index: Buffer<Option<(usize, usize)>>,
    incremental: bool,
    pending_value: Option<Value>,
    pending_component: u8,
    pending_node: Option<Node>,
    pending_add: Option<PendingAdd>,
}

#[derive(Debug)]
struct PendingAdd {
    node: Option<Node>,
    cycle: Option<Arc<crate::heap::CycleSlot>>,
    strong: bool,
    arena: bool,
    edge: bool,
    slot: usize,
    probing: bool,
    index: Option<usize>,
}

#[derive(Debug)]
struct Workspace {
    graph: Graph,
    pending: Buffer<usize>,
    progress: Option<GcProgress>,
    spare_index: Option<Buffer<Option<(usize, usize)>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GcPhase { Classes, Cycles, Roots, RetiredRoots, Trace, Classify, Mark, Promote, Nursery, Remember, Sweep, RetireRoots, CompactClasses, TrimClasses, ReleaseCounts, ReleaseCountEdges, ReleaseSeeds, ReleaseParents, ReleaseChildren, ReleaseNodes, TrimNodes, ReleaseEdges, ReleaseIndex, ReleasePending, PruneCycles, PruneOwned, Finish }

#[derive(Debug)]
enum RetiringNode {
    Joint(JointState), Kernel(FiniteKernel), Network(FiniteNetwork),
    Possibility(Possibility), Paths(PathSet), Adt(Adt), Object(ObjectValue),
    StateDist(StateDist), Derivation(Derivation),
}

impl RetiringNode {
    fn step(&mut self) -> bool {
        match self {
            Self::Joint(value) => {
                if value.values.pop().is_some() || value.names.pop().is_some() { return false; }
                if let Some(row) = value.rows.last_mut() {
                    if row.values.pop().is_none() { value.rows.pop(); }
                    return false;
                }
            }
            Self::Kernel(value) => {
                if value.output_domain.pop().is_some() || value.rows.pop().is_some() { return false; }
                if let Some(domain) = value.input_domains.last_mut() {
                    if domain.pop().is_none() { value.input_domains.pop(); }
                    return false;
                }
            }
            Self::Network(value) => { if value.nodes.pop().is_some() { return false; } }
            Self::Possibility(value) => { if value.values.pop().is_some() { return false; } }
            Self::Paths(value) => { if value.alternatives.pop().is_some() { return false; } }
            Self::Adt(value) => { if value.fields.pop().is_some() { return false; } }
            Self::Object(value) => { if value.fields.pop().is_some() { return false; } }
            Self::Derivation(value) => {
                if value.inputs.pop().is_some() || value.ad_a_deriv.take().is_some() || value.ad_b_deriv.take().is_some() { return false; }
            }
            Self::StateDist(value) => {
                let child = match &mut value.kind {
                    StateDistKind::Append { left: DistOperand::Node(_), .. } => {
                        let StateDistKind::Append { left, .. } = &mut value.kind else { unreachable!() };
                        Some(std::mem::replace(left, DistOperand::Inline(StateValue::default())))
                    }
                    StateDistKind::Append { right: DistOperand::Node(_), .. } => {
                        let StateDistKind::Append { right, .. } = &mut value.kind else { unreachable!() };
                        Some(std::mem::replace(right, DistOperand::Inline(StateValue::default())))
                    }
                    StateDistKind::Transform { .. } | StateDistKind::Attenuate { .. } => {
                        let prior = std::mem::replace(&mut value.kind, StateDistKind::Dirac(StateValue::default()));
                        drop(prior);
                        return false;
                    }
                    _ => None,
                };
                if child.is_some() { return false; }
            }
        }
        true
    }

    fn from_node(node: Node) -> Option<Self> {
        macro_rules! take {
            ($value:expr, $variant:ident) => { Arc::try_unwrap($value).ok().map(Self::$variant) };
        }
        match node {
            Node::Joint(value) => take!(value, Joint),
            Node::Kernel(value) => take!(value, Kernel),
            Node::Network(value) => take!(value, Network),
            Node::Possibility(value) => take!(value, Possibility),
            Node::Paths(value) => take!(value, Paths),
            Node::Adt(value) => take!(value, Adt),
            Node::Object(value) => take!(value, Object),
            Node::StateDist(value) => take!(value, StateDist),
            Node::Derivation(value) => take!(value, Derivation),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct GcProgress {
    phase: GcPhase,
    major: bool,
    mutation_epoch: u64,
    class_limit: usize,
    cycle_limit: usize,
    root_limit: usize,
    class_cursor: usize,
    cycle_cursor: usize,
    root_cursor: usize,
    root_edge_end: usize,
    trace_cursor: usize,
    classify_cursor: usize,
    root_seed_cursor: usize,
    mark_current: Option<usize>,
    mark_edge: usize,
    promote_cursor: usize,
    nursery_cursor: usize,
    nursery_nonempty: bool,
    remember_cursor: usize,
    remember_edge: usize,
    remember_started: bool,
    remembered: bool,
    sweep_cursor: usize,
    compact_cursor: usize,
    class_read: usize,
    class_write: usize,
    prune_cursor: usize,
    release_index: usize,
    release_node: usize,
    release_edge: usize,
    retiring: Option<RetiringNode>,
    aborted: bool,
}

impl GcProgress {
    fn new(major: bool, mutation_epoch: u64, class_limit: usize, cycle_limit: usize, root_limit: usize) -> Self {
        Self { phase: GcPhase::Classes, major, mutation_epoch, class_limit, cycle_limit, root_limit,
            class_cursor: 0, cycle_cursor: 0, root_cursor: 0, root_edge_end: 0, trace_cursor: 0,
            classify_cursor: 0, root_seed_cursor: 0, mark_current: None, mark_edge: 0,
            promote_cursor: 0, nursery_cursor: 0, nursery_nonempty: false, remember_cursor: 0,
            remember_edge: 0, remember_started: false, remembered: false,
            sweep_cursor: 0, compact_cursor: 0, class_read: 0, class_write: 0,
            prune_cursor: 0, release_index: 0, release_node: 0, release_edge: 0, retiring: None, aborted: false }
    }
}

impl Workspace {
    fn new(heap: &Heap) -> Self {
        Self {
            graph: Graph {
                nodes: Buffer::new(heap, 0, 0).expect("empty collector nodes"),
                edges: Buffer::new(heap, 0, 0).expect("empty collector edges"),
                index: Buffer::new(heap, 0, 0).expect("empty collector index"),
                incremental: false, pending_value: None, pending_component: 0, pending_node: None, pending_add: None,
            },
            pending: Buffer::new(heap, 0, 0).expect("empty collector worklist"),
            progress: None,
            spare_index: None,
        }
    }

    fn admit(&mut self, heap: &Heap, nodes: usize, edges: usize) -> Result<(), LanaError> {
        self.graph.nodes.reserve(nodes.saturating_sub(self.graph.nodes.len()))?;
        self.graph.edges.reserve(edges.saturating_sub(self.graph.edges.len()))?;
        self.pending.reserve(nodes.saturating_sub(self.pending.len()))?;
        let capacity = nodes.checked_mul(2).and_then(|n| n.max(16).checked_next_power_of_two()).ok_or(LanaError::Oom)?;
        if self.graph.index.len() < capacity && self.spare_index.as_ref().is_none_or(|index| index.len() < capacity) {
            let index = Buffer::filled(heap, capacity, None)?;
            if let Some(progress) = &mut self.progress {
                self.spare_index = Some(index);
                progress.aborted = true;
                progress.release_node = 0;
                progress.phase = GcPhase::ReleaseNodes;
            }
            else { self.graph.index = index; }
        }
        Ok(())
    }

    fn clear(&mut self) {
        // Never retain graph ownership between collections. Only the accounted
        // capacities survive, so they cannot become false external roots.
        for entry in self.graph.nodes.iter() {
            if let Some(cycle) = &entry.cycle { cycle.set_tracing(false); }
        }
        self.graph.pending_add = None;
        self.graph.pending_node = None;
        self.graph.pending_value = None;
        self.graph.nodes.clear();
        self.graph.edges.clear();
        self.graph.index.fill(None);
        self.pending.clear();
        self.progress = None;

    }
}

impl Graph {
    fn slot(index: &[Option<(usize, usize)>], identity: usize) -> usize {
        let mut slot = identity.rotate_right(4).wrapping_mul(0x9e3779b9) & (index.len() - 1);
        while index[slot].is_some_and(|(key, _)| key != identity) {
            slot = (slot + 1) & (index.len() - 1);
        }
        slot
    }

    fn add(&mut self, node: Node, strong: bool, vm: &mut Context<'_>) -> Result<usize, LanaError> {
        self.add_cycle(node, strong, None, vm)
    }

    fn add_cycle(&mut self, node: Node, strong: bool, cycle: Option<Arc<crate::heap::CycleSlot>>, vm: &mut Context<'_>) -> Result<usize, LanaError> {
        if self.incremental {
            self.queue_add(node, strong, cycle, false, false)?;
            return Ok(usize::MAX);
        }
        (vm.charge)(1)?;
        let identity = node.identity();
        if !self.index.is_empty() {
            let slot = Self::slot(&self.index, identity);
            if let Some((_, index)) = self.index[slot] {
                if self.nodes[index].cycle.is_none() { self.nodes[index].cycle = cycle; }
                self.nodes[index].incoming += usize::from(strong);
                return Ok(index);
            }
        }
        if self.nodes.len().saturating_mul(2) >= self.index.len() {
            let capacity = self.index.len().checked_mul(2).ok_or(LanaError::Oom)?.max(16);
            (vm.charge)(capacity as u64)?;
            let mut index = Buffer::filled(vm.heap, capacity, None)?;
            for (position, entry) in self.nodes.iter().enumerate() {
                let identity = entry.node.identity();
                let slot = Self::slot(&index, identity);
                index[slot] = Some((identity, position));
            }
            self.index = index;
        }
        let slot = Self::slot(&self.index, identity);
        let index = match self.index[slot] {
            Some((_, index)) => {
                if self.nodes[index].cycle.is_none() { self.nodes[index].cycle = cycle; }
                index
            }
            None => {
                self.nodes.push(Entry { node, cycle, incoming: 0, arena: false, live: false, edges: 0..0,
                    trace_cursor: 0, trace_outer: 0, trace_inner: 0, trace_phase: 0, trace_started: false })?;
                let index = self.nodes.len() - 1;
                self.index[slot] = Some((identity, index));
                index
            }
        };
        self.nodes[index].incoming += usize::from(strong);
        Ok(index)
    }

    fn edge(&mut self, node: Node, strong: bool, vm: &mut Context<'_>) -> Result<(), LanaError> {
        if self.incremental { return self.queue_add(node, strong, None, false, true); }
        let index = self.add(node, strong, vm)?;
        self.edges.push(index)
    }

    fn seed_from(&mut self, source: &Self, vm: &mut Context<'_>) -> Result<(), LanaError> {
        for entry in &source.nodes {
            if !matches!(entry.node, Node::Vacant) {
                self.add_cycle(entry.node.clone(), false, entry.cycle.clone(), vm)?;
            }
        }
        if let Some(node) = &source.pending_node { self.add(node.clone(), false, vm)?; }
        if let Some(operation) = &source.pending_add {
            if let Some(node) = &operation.node { self.add_cycle(node.clone(), false, operation.cycle.clone(), vm)?; }
        }
        if let Some(value) = &source.pending_value {
            for component in 0..5 {
                if let Some(node) = Self::value_node(value, component) { self.add(node, false, vm)?; }
            }
        }
        Ok(())
    }

    fn queue_add(&mut self, node: Node, strong: bool, cycle: Option<Arc<crate::heap::CycleSlot>>, arena: bool, edge: bool) -> Result<(), LanaError> {
        debug_assert!(self.pending_add.is_none());
        if self.index.is_empty() { return Err(LanaError::Oom); }
        let slot = node.identity().rotate_right(4).wrapping_mul(0x9e3779b9) & (self.index.len() - 1);
        self.pending_add = Some(PendingAdd { node: Some(node), cycle, strong, arena, edge,
            slot, probing: true, index: None });
        Ok(())
    }

    fn add_one(&mut self, vm: &mut Context<'_>) -> Result<(), LanaError> {
        let mut operation = self.pending_add.take().expect("pending graph insertion");
        (vm.charge)(1)?;
        if operation.probing {
            match self.index[operation.slot] {
                Some((key, _)) if key != operation.node.as_ref().unwrap().identity() => {
                    operation.slot = (operation.slot + 1) & (self.index.len() - 1);
                }
                Some((_, index)) => {
                    let entry = &mut self.nodes[index];
                    if entry.cycle.is_none() { entry.cycle = operation.cycle.take(); }
                    entry.incoming += usize::from(operation.strong);
                    entry.arena |= operation.arena;
                    operation.index = Some(index);
                    operation.probing = false;
                    if !operation.edge { return Ok(()); }
                }
                None => operation.probing = false,
            }
            self.pending_add = Some(operation);
            return Ok(());
        }
        if let Some(index) = operation.index {
            if self.edges.len() == self.edges.capacity() { return Err(LanaError::Oom); }
            self.edges.push(index)?;
            return Ok(());
        }
        // All storage must have been admitted by the mutator. Routine insertion
        // never allocates a buffer or rehashes existing entries.
        if self.nodes.len() == self.nodes.capacity() || self.nodes.len().saturating_mul(2) >= self.index.len() {
            return Err(LanaError::Oom);
        }
        let index = self.nodes.len();
        self.nodes.push(Entry { node: operation.node.take().unwrap(), cycle: operation.cycle.take(),
            incoming: usize::from(operation.strong), arena: operation.arena, live: false, edges: 0..0,
            trace_cursor: 0, trace_outer: 0, trace_inner: 0, trace_phase: 0, trace_started: false })?;
        self.index[operation.slot] = Some((self.nodes[index].node.identity(), index));
        if operation.edge {
            operation.index = Some(index);
            self.pending_add = Some(operation);
        }
        Ok(())
    }

    fn value_node(value: &Value, component: u8) -> Option<Node> {
        match component {
            0 => return value.reactive.as_ref().map(|value| Node::Reactive(value.clone())),
            1 => return value.claim.as_ref().map(|value| Node::Claim(value.clone())),
            2 => return value.planned_effect.as_ref().map(|value| Node::Effect(value.clone())),
            3 => return value.derivation.as_ref().map(|value| Node::Derivation(value.clone())),
            _ => {}
        }
        let node = match &value.kind {
            ValueKind::ClassObject(v) => Node::Class(v.clone()),
            ValueKind::Array(v) => Node::Array(v.clone()),
            ValueKind::Map(v) => Node::Map(v.clone()),
            ValueKind::Set(v) => Node::Set(v.clone()),
            ValueKind::Joint(v) => Node::Joint(v.clone()),
            ValueKind::Kernel(v) => Node::Kernel(v.clone()),
            ValueKind::Network(v) => Node::Network(v.clone()),
            ValueKind::Possibility(v) => Node::Possibility(v.clone()),
            ValueKind::PathSet(v) => Node::Paths(v.clone()),
            ValueKind::Adt(v) => Node::Adt(v.clone()),
            ValueKind::ObjectValue(v) => Node::Object(v.clone()),
            ValueKind::Generator(v) => Node::Generator(v.clone()),
            ValueKind::Future(v) => Node::Future(v.clone()),
            ValueKind::TrainingResult(v) => Node::Training(v.clone()),
            ValueKind::Posterior(v) => Node::Posterior(v.clone()),
            ValueKind::Dataset(v) => Node::Dataset(v.clone()),
            ValueKind::Task(v) => Node::Task(v.clone()),
            ValueKind::StateDist(v) => Node::StateDist(v.clone()),
            // Concurrent task/shared state is an external root. Do not subtract
            // its references or acquire its locks during a task-local collection.
            ValueKind::Capability(_) => return None,
            // These payloads and derivations cannot contain class references.
            ValueKind::Null | ValueKind::Number(_) | ValueKind::Bool(_) | ValueKind::String(_)
            | ValueKind::State(_) | ValueKind::Distribution { .. } | ValueKind::Sample(_)
            | ValueKind::Function(_) | ValueKind::Tensor(_)
            | ValueKind::NQubitState(_) | ValueKind::Povm(_) | ValueKind::Channel(_)
            | ValueKind::Observable(_) | ValueKind::Lazy { .. } | ValueKind::Regex(_)
            | ValueKind::Optimizer(_) | ValueKind::InferenceAlgorithm(_) => return None,
        };
        Some(node)
    }

    fn value(&mut self, value: &Value, vm: &mut Context<'_>) -> Result<(), LanaError> {
        if self.incremental {
            if value.reactive.is_none() && value.claim.is_none() && value.planned_effect.is_none() && value.derivation.is_none() {
                if let Some(node) = Self::value_node(value, 4) { self.edge(node, true, vm)?; }
                return Ok(());
            }
            debug_assert!(self.pending_value.is_none());
            self.pending_value = Some(value.clone());
            self.pending_component = 0;
            return Ok(());
        }
        (vm.charge)(1)?;
        for component in 0..5 {
            if let Some(node) = Self::value_node(value, component) { self.edge(node, true, vm)?; }
        }
        Ok(())
    }

    fn trace_value_component(&mut self, vm: &mut Context<'_>) -> Result<bool, LanaError> {
        let Some(value) = &self.pending_value else { return Ok(false); };
        // There are exactly five possible links. Skip absent links without
        // spending a safepoint on each scalar's empty metadata fields.
        let mut node = None;
        while self.pending_component < 5 && node.is_none() {
            node = Self::value_node(value, self.pending_component);
            self.pending_component += 1;
        }
        if let Some(node) = node { self.edge(node, true, vm)?; }
        if self.pending_component == 5 { self.pending_value = None; }
        Ok(true)
    }

    fn trace(&mut self, node: &Node, vm: &mut Context<'_>) -> Result<bool, LanaError> {
        // A reentrant host callback can retain a container lock. Defer the pass
        // rather than deadlock or guess what is reachable through that container.
        macro_rules! lock {
            ($value:expr) => { match $value.try_lock() { Ok(value) => value, Err(_) => return Ok(false) } };
        }
        match node {
            Node::Vacant => {},
            Node::Class(v) => {
                if let Some(storage) = v.object.upgrade() { self.edge(Node::Storage(storage), false, vm)?; }
            }
            Node::Storage(v) => for value in lock!(v.fields).iter().flatten() { self.value(value, vm)?; },
            Node::Array(v) => for value in &lock!(v).items { self.value(value, vm)?; },
            Node::Map(v) => for entry in &lock!(v).entries { self.value(&entry.value, vm)?; },
            Node::Set(v) => for value in &lock!(v).items { self.value(value, vm)?; },
            Node::Joint(v) => {
                for value in &v.values { self.value(value, vm)?; }
                for row in &v.rows { for value in &row.values { self.value(value, vm)?; } }
            }
            Node::Kernel(v) => {
                for domain in &v.input_domains { for value in domain { self.value(value, vm)?; } }
                for value in &v.output_domain { self.value(value, vm)?; }
            }
            Node::Network(v) => {
                self.edge(Node::Joint(v.root.clone()), true, vm)?;
                for node in &v.nodes { self.edge(Node::Kernel(node.kernel.clone()), true, vm)?; }
            }
            Node::Possibility(v) => for value in &v.values { self.value(value, vm)?; },
            Node::Paths(v) => for value in &v.alternatives { self.value(&value.result, vm)?; },
            Node::Adt(v) => for value in &v.fields { self.value(value, vm)?; },
            Node::Object(v) => for value in &v.fields { self.value(value, vm)?; },
            Node::Generator(v) => for value in &lock!(v).registers { self.value(value, vm)?; },
            Node::Future(v) => for value in &lock!(v).registers { self.value(value, vm)?; },
            Node::Task(v) => {
                let state = lock!(v.state);
                // Before join, the child heap is owned by its independent lease.
                if state.joined { self.value(&state.result, vm)?; }
            },
            Node::Training(v) => {
                self.edge(Node::Array(v.steps.clone()), true, vm)?;
                self.value(&v.data, vm)?;
            }
            Node::Posterior(v) => self.edge(Node::Array(v.steps.clone()), true, vm)?,
            Node::Dataset(v) => for value in [&v.source, &v.columns, &v.key, &v.limit, &v.other, &v.aggregate] {
                self.value(value, vm)?;
            },
            Node::Reactive(v) => {
                let value = lock!(v);
                for input in value.inputs.iter().flatten() { self.edge(Node::Reactive(input.clone()), true, vm)?; }
                for value in value.constants.iter().flatten().chain(value.current.iter()) { self.value(value, vm)?; }
                for version in &value.history { if let Some(value) = &version.value { self.value(value, vm)?; } }
            }
            Node::Claim(v) => self.value(&v.value, vm)?,
            Node::Derivation(v) => {
                for child in v.inputs.iter().chain(v.ad_a_deriv.iter()).chain(v.ad_b_deriv.iter()) {
                    self.edge(Node::Derivation(child.clone()), true, vm)?;
                }
            }
            Node::StateDist(v) => {
                for cursor in 0..2 {
                    if let Some(child) = distribution_child(v, cursor) { self.edge(Node::StateDist(child), true, vm)?; }
                }
            }
            Node::Effect(v) => {
                self.value(&v.payload, vm)?;
                for receipt in &lock!(v.state).receipts { self.value(&receipt.result, vm)?; }
            }
        }
        Ok(true)
    }

    fn trace_one(&mut self, index: usize, vm: &mut Context<'_>) -> Result<Option<bool>, LanaError> {
        let node = self.nodes[index].node.clone();
        let cursor = self.nodes[index].trace_cursor;
        match &node {
            Node::Network(value) => {
                let child = if cursor == 0 { Some(Node::Joint(value.root.clone())) }
                    else { value.nodes.get(cursor - 1).map(|node| Node::Kernel(node.kernel.clone())) };
                let Some(child) = child else { return Ok(Some(true)); };
                self.nodes[index].trace_cursor += 1;
                self.edge(child, true, vm)?;
                return Ok(Some(false));
            }
            Node::Joint(value) => {
                if self.nodes[index].trace_phase == 0 {
                    if let Some(child) = value.values.get(cursor) {
                        let child = child.clone();
                        self.nodes[index].trace_cursor += 1;
                        self.value(&child, vm)?;
                        return Ok(Some(false));
                    }
                    self.nodes[index].trace_phase = 1;
                    self.nodes[index].trace_outer = 0;
                    self.nodes[index].trace_inner = 0;
                    return Ok(Some(false));
                }
                let outer = self.nodes[index].trace_outer;
                if outer >= value.rows.len() { return Ok(Some(true)); }
                let inner = self.nodes[index].trace_inner;
                if let Some(child) = value.rows[outer].values.get(inner) {
                    let child = child.clone();
                    self.nodes[index].trace_inner += 1;
                    self.value(&child, vm)?;
                } else {
                    self.nodes[index].trace_outer += 1;
                    self.nodes[index].trace_inner = 0;
                }
                return Ok(Some(false));
            }
            Node::Kernel(value) => {
                if self.nodes[index].trace_phase == 0 {
                    let outer = self.nodes[index].trace_outer;
                    if outer < value.input_domains.len() {
                        let inner = self.nodes[index].trace_inner;
                        if let Some(child) = value.input_domains[outer].get(inner) {
                            let child = child.clone();
                            self.nodes[index].trace_inner += 1;
                            self.value(&child, vm)?;
                        } else {
                            self.nodes[index].trace_outer += 1;
                            self.nodes[index].trace_inner = 0;
                        }
                        return Ok(Some(false));
                    }
                    self.nodes[index].trace_phase = 1;
                    self.nodes[index].trace_cursor = 0;
                    return Ok(Some(false));
                }
                if let Some(child) = value.output_domain.get(cursor) {
                    let child = child.clone();
                    self.nodes[index].trace_cursor += 1;
                    self.value(&child, vm)?;
                    return Ok(Some(false));
                }
                return Ok(Some(true));
            }
            Node::Training(value) => {
                if cursor == 0 {
                    self.nodes[index].trace_cursor = 1;
                    self.edge(Node::Array(value.steps.clone()), true, vm)?;
                    return Ok(Some(false));
                }
                if cursor == 1 {
                    self.nodes[index].trace_cursor = 2;
                    self.value(&value.data, vm)?;
                    return Ok(Some(false));
                }
                return Ok(Some(true));
            }
            Node::Posterior(value) => {
                if cursor == 0 {
                    self.nodes[index].trace_cursor = 1;
                    self.edge(Node::Array(value.steps.clone()), true, vm)?;
                    return Ok(Some(false));
                }
                return Ok(Some(true));
            }
            _ => {}
        }
        let value = match &node {
            Node::Storage(v) => match v.fields.try_lock() {
                Ok(fields) => {
                    if cursor >= fields.len() { return Ok(Some(true)); }
                    fields[cursor].clone()
                }
                Err(_) => return Ok(None),
            },
            Node::Array(v) => match v.try_lock() {
                Ok(array) => {
                    if cursor >= array.items.len() { return Ok(Some(true)); }
                    Some(array.items[cursor].clone())
                }
                Err(_) => return Ok(None),
            },
            Node::Map(v) => match v.try_lock() {
                Ok(map) => {
                    if cursor >= map.entries.len() { return Ok(Some(true)); }
                    Some(map.entries[cursor].value.clone())
                }
                Err(_) => return Ok(None),
            },
            Node::Set(v) => match v.try_lock() {
                Ok(set) => {
                    if cursor >= set.items.len() { return Ok(Some(true)); }
                    Some(set.items[cursor].clone())
                }
                Err(_) => return Ok(None),
            },
            Node::Possibility(v) => match v.values.get(cursor) { Some(value) => Some(value.clone()), None => return Ok(Some(true)) },
            Node::Paths(v) => match v.alternatives.get(cursor) { Some(value) => Some(value.result.clone()), None => return Ok(Some(true)) },
            Node::Adt(v) => match v.fields.get(cursor) { Some(value) => Some(value.clone()), None => return Ok(Some(true)) },
            Node::Object(v) => match v.fields.get(cursor) { Some(value) => Some(value.clone()), None => return Ok(Some(true)) },
            Node::Generator(v) => match v.try_lock() {
                Ok(value) => match value.registers.get(cursor) { Some(item) => Some(item.clone()), None => return Ok(Some(true)) },
                Err(_) => return Ok(None),
            },
            Node::Future(v) => match v.try_lock() {
                Ok(value) => match value.registers.get(cursor) { Some(item) => Some(item.clone()), None => return Ok(Some(true)) },
                Err(_) => return Ok(None),
            },
            Node::Dataset(v) => match [&v.source, &v.columns, &v.key, &v.limit, &v.other, &v.aggregate].get(cursor) {
                Some(value) => Some((*value).clone()), None => return Ok(Some(true)),
            },
            Node::Task(v) => match v.state.try_lock() {
                Ok(state) => if cursor == 0 && state.joined { Some(state.result.clone()) } else { return Ok(Some(true)); },
                Err(_) => return Ok(None),
            },
            Node::Claim(v) => if cursor == 0 { Some(v.value.clone()) } else { return Ok(Some(true)); },
            Node::Derivation(v) => {
                let child = if cursor < v.inputs.len() { Some(v.inputs[cursor].clone()) }
                    else if cursor == v.inputs.len() { v.ad_a_deriv.clone() }
                    else if cursor == v.inputs.len() + 1 { v.ad_b_deriv.clone() }
                    else { return Ok(Some(true)); };
                if let Some(child) = child { self.edge(Node::Derivation(child), true, vm)?; }
                self.nodes[index].trace_cursor += 1;
                return Ok(Some(false));
            }
            Node::StateDist(v) => {
                if cursor >= 2 { return Ok(Some(true)); }
                if let Some(child) = distribution_child(v, cursor) { self.edge(Node::StateDist(child), true, vm)?; }
                self.nodes[index].trace_cursor += 1;
                return Ok(Some(false));
            }
            Node::Reactive(v) => match v.try_lock() {
                Ok(reactive) => {
                    let count = 5 + reactive.history.len();
                    if cursor >= count { return Ok(Some(true)); }
                    match cursor {
                        0..=1 => {
                            let Some(input) = &reactive.inputs[cursor] else { self.nodes[index].trace_cursor += 1; return Ok(Some(false)); };
                            self.edge(Node::Reactive(input.clone()), true, vm)?;
                            self.nodes[index].trace_cursor += 1;
                            return Ok(Some(false));
                        }
                        2..=3 => reactive.constants[cursor - 2].clone(),
                        4 => reactive.current.clone(),
                        i => reactive.history[i - 5].value.clone(),
                    }
                }
                Err(_) => return Ok(None),
            },
            Node::Effect(v) if cursor == 0 => Some(v.payload.clone()),
            Node::Effect(v) => match v.state.try_lock() {
                Ok(state) => match state.receipts.get(cursor - 1) { Some(receipt) => Some(receipt.result.clone()), None => return Ok(Some(true)) },
                Err(_) => return Ok(None),
            },
            _ => {
                if cursor != 0 { return Ok(Some(true)); }
                if !self.trace(&node, vm)? { return Ok(None); }
                self.nodes[index].trace_cursor = 1;
                return Ok(Some(true));
            }
        };
        self.nodes[index].trace_cursor += 1;
        if let Some(value) = value { self.value(&value, vm)?; }
        Ok(Some(false))
    }
}

impl Vm<'_> {
    pub(super) fn managed_payload<T: crate::heap::ManagedPayload>(&mut self, payload: T) -> Result<Arc<T>, LanaError> {
        let allocation = self.heap.reserve(payload.heap_bytes()?)?;
        let payload = Arc::new(payload);
        self.track_cycle_with_allocation(T::weak(Arc::downgrade(&payload)), Some(allocation))?;
        Ok(payload)
    }

    pub fn dataset_value(&mut self, payload: Dataset) -> Result<Value, LanaError> {
        Ok(Value::dataset(self.managed_payload(payload)?))
    }

    pub fn possibility_value(&mut self, payload: Possibility) -> Result<Value, LanaError> {
        Ok(Value::possibility(self.managed_payload(payload)?))
    }

    pub fn joint_value(&mut self, payload: JointState) -> Result<Value, LanaError> {
        Ok(Value::joint(self.managed_payload(payload)?))
    }

    fn take_unpublished_payload<T>(&self, payload: Arc<T>) -> Result<T, LanaError> {
        for index in 0..self.heap.cycle_count() {
            let Some(slot) = self.heap.cycle_slot_at(index) else { continue; };
            if slot.node.lock().unwrap().as_ref().is_some_and(|node| node.identity() == Arc::as_ptr(&payload) as usize) {
                if Arc::strong_count(&payload) != 1 + usize::from(slot.owns_node()) { return Err(LanaError::Task); }
                if !slot.release_owner() { return Err(LanaError::UnsupportedOperation); }
                let payload = Arc::try_unwrap(payload).map_err(|_| LanaError::Task)?;
                slot.payload_allocation.lock().unwrap().take();
                return Ok(payload);
            }
        }
        Arc::try_unwrap(payload).map_err(|_| LanaError::Task)
    }

    pub fn into_joint_payload(&self, payload: Arc<JointState>) -> Result<JointState, LanaError> {
        self.take_unpublished_payload(payload)
    }

    pub fn into_possibility_payload(&self, payload: Arc<Possibility>) -> Result<Possibility, LanaError> {
        self.take_unpublished_payload(payload)
    }

    pub(super) fn track_cycle(&mut self, node: crate::heap::CycleWeak) -> Result<(), LanaError> {
        self.track_cycle_with_allocation(node, None)
    }

    fn track_cycle_with_allocation(&mut self, node: crate::heap::CycleWeak, allocation: Option<crate::heap::Reservation>) -> Result<(), LanaError> {
        let allocation = match allocation {
            Some(allocation) => Some(allocation),
            None => node.payload_bytes()?.map(|bytes| self.heap.reserve(bytes)).transpose()?,
        };
        let values = match &node {
            crate::heap::CycleWeak::Generator(value) => value.upgrade().ok_or(LanaError::Task)?.lock().map_err(|_| LanaError::Task)?.registers.capacity(),
            crate::heap::CycleWeak::Future(value) => value.upgrade().ok_or(LanaError::Task)?.lock().map_err(|_| LanaError::Task)?.registers.capacity(),
            crate::heap::CycleWeak::Reactive(value) => value.upgrade().ok_or(LanaError::Task)?.lock().map_err(|_| LanaError::Task)?.history.capacity().checked_add(5).ok_or(LanaError::Oom)?,
            crate::heap::CycleWeak::Effect(value) => value.upgrade().ok_or(LanaError::Task)?.state.lock().map_err(|_| LanaError::Task)?.receipts.capacity().checked_add(1).ok_or(LanaError::Oom)?,
            crate::heap::CycleWeak::Joint(value) => {
                let value = value.upgrade().ok_or(LanaError::Task)?;
                self.charge_bounded_work(value.rows.len() as u64)?;
                value.rows.iter().try_fold(value.values.len(), |count, row|
                    count.checked_add(row.values.len()).ok_or(LanaError::Oom))?
            }
            crate::heap::CycleWeak::Kernel(value) => {
                let value = value.upgrade().ok_or(LanaError::Task)?;
                self.charge_bounded_work(value.input_domains.len() as u64)?;
                value.input_domains.iter().try_fold(value.output_domain.len(), |count, domain|
                    count.checked_add(domain.len()).ok_or(LanaError::Oom))?
            }
            crate::heap::CycleWeak::Network(value) => value.upgrade().ok_or(LanaError::Task)?.nodes.len().checked_add(1).ok_or(LanaError::Oom)?,
            crate::heap::CycleWeak::Possibility(value) => value.upgrade().ok_or(LanaError::Task)?.values.len(),
            crate::heap::CycleWeak::Paths(value) => value.upgrade().ok_or(LanaError::Task)?.alternatives.len(),
            crate::heap::CycleWeak::Adt(value) => value.upgrade().ok_or(LanaError::Task)?.fields.len(),
            crate::heap::CycleWeak::Object(value) => value.upgrade().ok_or(LanaError::Task)?.fields.len(),
            crate::heap::CycleWeak::Training(_) => 2,
            crate::heap::CycleWeak::Posterior(_) => 1,
            crate::heap::CycleWeak::Dataset(_) => 6,
            crate::heap::CycleWeak::Claim(_) => 1,
            crate::heap::CycleWeak::Task(_) => 1,
            crate::heap::CycleWeak::StateDist(_) => 2,
            crate::heap::CycleWeak::Derivation(value) => value.upgrade().ok_or(LanaError::Task)?.inputs.len().checked_add(2).ok_or(LanaError::Oom)?,
            _ => 0,
        };
        let edges = values.checked_mul(5).ok_or(LanaError::Oom)?;
        let slot = self.heap.cycle_slot_with_edges(edges)?;
        *slot.payload_allocation.lock().unwrap() = allocation;
        slot.bind(node);
        self.owned_cycles.push(slot)
    }

    pub(super) fn reserve_cycle_edges(&self, identity: usize, values: usize, bytes: usize) -> Result<(), LanaError> {
        let edges = values.checked_mul(5).ok_or(LanaError::Oom)?;
        for index in 0..self.heap.cycle_count() {
            if let Some(slot) = self.heap.cycle_slot_at(index) {
                if slot.node.lock().unwrap().as_ref().is_some_and(|node| node.identity() == identity) {
                    return slot.reserve_edges(edges, bytes);
                }
            }
        }
        Err(LanaError::Task)
    }

    pub(super) fn collect_classes(&mut self) -> Result<(), LanaError> {
        self.collect_generation(true)
    }

    fn collect_generation(&mut self, major: bool) -> Result<(), LanaError> {
        if !self.constructions.is_empty() { return Ok(()); }
        let heap = self.heap.clone();
        let mut classes = std::mem::take(&mut self.class_objects);
        let owner = self.host_roots.clone();
        let mut roots = owner.roots.lock().unwrap();
        let mut routine = owner.workspace.lock().unwrap();
        let Ok(mut workspace) = owner.retention_workspace.try_lock() else {
            self.class_objects = classes;
            return Ok(());
        };
        workspace.clear();
        let mut context = Context { heap: &heap, charge: &mut |steps| self.charge_bounded_work(steps) };
        let result = match workspace.graph.seed_from(&routine.graph, &mut context) {
            Ok(()) => collect(&mut classes, &mut roots.values, &mut context,
                &mut workspace, major, &mut || routine.clear()),
            Err(error) => { workspace.clear(); Err(error) }
        };
        owner.active_collection.store(routine.progress.as_ref().map_or(0, |progress| if progress.major { 2 } else { 1 }), Ordering::Release);
        self.class_objects = classes;
        if result? {
            self.class_allocations_since_gc = 0;
            self.owned_cycles.retain(|slot| slot.node.lock().unwrap().as_ref().is_some_and(crate::heap::CycleWeak::alive));
            if self.class_objects.is_empty() {
                self.class_objects = Vec::new();
                self.class_objects_reservation.lock().unwrap().resize(0)?;
            }
            if owner.admitted_nodes.load(Ordering::Acquire) == 0 && owner.admitted_edges.load(Ordering::Acquire) == 0 {
                *routine = Workspace::new(&heap);
                *workspace = Workspace::new(&heap);
            }
        }
        Ok(())
    }

    pub(super) fn collect_safepoint_slice(&mut self, major: bool) -> Result<bool, LanaError> {
        self.collect_slice(major, 128)
    }

    fn collect_slice(&mut self, major: bool, slice: usize) -> Result<bool, LanaError> {
        assert!(slice > 0);
        if !self.constructions.is_empty() { return Ok(false); }
        let heap = self.heap.clone();
        let _routine = heap.routine_collection();
        let owner = self.host_roots.clone();
        let mut classes = std::mem::take(&mut self.class_objects);
        let Ok(mut roots) = owner.roots.try_lock() else { self.class_objects = classes; return Ok(false); };
        let Ok(mut workspace) = owner.workspace.try_lock() else { self.class_objects = classes; return Ok(false); };
        if workspace.progress.is_none() {
            debug_assert!(workspace.graph.nodes.is_empty() && workspace.graph.edges.is_empty() && workspace.pending.is_empty());
            if let Some(index) = workspace.spare_index.take() { workspace.graph.index = index; }
            workspace.progress = Some(GcProgress::new(major, heap.mutation_epoch(), classes.len(), heap.cycle_count(), roots.values.len()));
            owner.active_collection.store(if major { 2 } else { 1 }, Ordering::Release);
        }
        let progress = workspace.progress.as_mut().unwrap();
        if !progress.aborted && progress.mutation_epoch != heap.mutation_epoch() {
            if matches!(progress.phase, GcPhase::Classes | GcPhase::Cycles) {
                // Registration has read no graph edges yet. Trace the current
                // contents and include newly registered external roots.
                progress.mutation_epoch = heap.mutation_epoch();
                progress.root_limit = roots.values.len();
            } else {
                progress.aborted = true;
                progress.release_node = 0;
                progress.phase = GcPhase::ReleaseNodes;
            }
        }
        let class_limit = workspace.progress.as_ref().unwrap().class_limit;
        let result = (|| {
            let limit = self.instruction_limit;
            let count = &mut self.instruction_count;
            let cancelled = &self.cancelled;
            let owned_cycles = &mut self.owned_cycles;
            let mut context = Context { heap: &heap, charge: &mut |steps| {
                if cancelled.load(Ordering::Relaxed) { return Err(LanaError::Cancelled); }
                if steps > limit.saturating_sub(*count) { return Err(LanaError::Limit); }
                *count += steps;
                Ok(())
            } };
            let Workspace { graph, pending, progress: state, .. } = &mut *workspace;
            let progress = state.as_mut().unwrap();
            graph.incremental = true;
            let mut work = 0usize;
            let finish = |count: usize| { owner.last_slice_work.store(count, Ordering::Release); };
            while work < slice {
                if graph.pending_add.is_some() {
                    if progress.aborted { graph.pending_add = None; }
                    else { graph.add_one(&mut context)?; }
                    work += 1;
                    continue;
                }
                if let Some(node) = graph.pending_node.take() {
                    if !progress.aborted { graph.add(node, false, &mut context)?; }
                    work += 1;
                    continue;
                }
                if graph.pending_value.is_some() {
                    if progress.aborted { graph.pending_value = None; }
                    else { graph.trace_value_component(&mut context)?; }
                    work += 1;
                    continue;
                }
                match progress.phase {
                    GcPhase::Classes => {
                        if progress.class_cursor >= progress.class_limit { progress.phase = GcPhase::Cycles; continue; }
                        let object = classes[progress.class_cursor].clone();
                        graph.queue_add(Node::Storage(object.clone()), false, None, true, false)?;
                        if let Some(reference) = object.reference.upgrade() { graph.pending_node = Some(Node::Class(reference)); }
                        progress.class_cursor += 1;
                        work += 1;
                    }
                    GcPhase::Cycles => {
                        if progress.cycle_cursor >= progress.cycle_limit { progress.phase = GcPhase::Roots; continue; }
                        let index = progress.cycle_cursor;
                        progress.cycle_cursor += 1;
                        work += 1;
                        if let Some(slot) = heap.cycle_slot_at(index) {
                            let Ok(weak) = slot.node.try_lock() else { progress.cycle_cursor -= 1; finish(work); return Ok(false); };
                            let node = weak.as_ref().and_then(crate::heap::CycleWeak::strong_node);
                            drop(weak);
                            if let Some(node) = node {
                                graph.add_cycle(node, false, Some(slot), &mut context)?;
                            }
                        }
                    }
                    GcPhase::Roots => {
                        if progress.root_cursor >= progress.root_limit {
                            progress.root_edge_end = graph.edges.len();
                            pending.reserve(graph.nodes.len())?;
                            progress.root_cursor = 0;
                            progress.phase = GcPhase::RetiredRoots;
                            continue;
                        }
                        if roots.values[progress.root_cursor].live() {
                            graph.value(&roots.values[progress.root_cursor].value, &mut context)?;
                        }
                        progress.root_cursor += 1;
                        work += 1;
                    }
                    GcPhase::RetiredRoots => {
                        if progress.root_cursor >= progress.root_limit { progress.phase = GcPhase::Trace; continue; }
                        if !roots.values[progress.root_cursor].live() {
                            graph.value(&roots.values[progress.root_cursor].value, &mut context)?;
                        }
                        progress.root_cursor += 1;
                        work += 1;
                    }
                    GcPhase::Trace => {
                        if progress.trace_cursor >= graph.nodes.len() { progress.phase = GcPhase::Classify; continue; }
                        let index = progress.trace_cursor;
                        let old_clean_cycle = graph.nodes[index].cycle.as_ref().is_some_and(|slot|
                            slot.generation() == crate::heap::Generation::Old && !slot.remembered());
                        let old_clean_class = match &graph.nodes[index].node {
                            Node::Storage(storage) => storage.generation.load(Ordering::Acquire) == crate::heap::Generation::Old as u8
                                && !storage.remembered.load(Ordering::Acquire),
                            _ => false,
                        };
                        if !graph.nodes[index].trace_started {
                            graph.nodes[index].edges.start = graph.edges.len();
                            graph.nodes[index].trace_started = true;
                            if let Some(slot) = &graph.nodes[index].cycle { slot.set_tracing(true); }
                        }
                        if !progress.major && (old_clean_cycle || old_clean_class) {
                            graph.nodes[index].edges.end = graph.edges.len();
                            graph.nodes[index].trace_cursor = usize::MAX;
                            graph.nodes[index].trace_started = true;
                            progress.trace_cursor += 1;
                            work += 1;
                            continue;
                        }
                        match graph.trace_one(index, &mut context)? {
                            None => { finish(work); return Ok(false); }
                            Some(true) => {
                                graph.nodes[index].edges.end = graph.edges.len();
                                progress.trace_cursor += 1;
                            }
                            Some(false) => {},
                        }
                        work += 1;
                    }
                    GcPhase::Classify => {
                        if progress.root_seed_cursor < progress.root_edge_end {
                            let index = graph.edges[progress.root_seed_cursor];
                            progress.root_seed_cursor += 1;
                            if !graph.nodes[index].live { graph.nodes[index].live = true; pending.push(index)?; }
                            work += 1;
                            continue;
                        }
                        if progress.classify_cursor >= graph.nodes.len() { progress.phase = GcPhase::Mark; continue; }
                        let index = progress.classify_cursor;
                        progress.classify_cursor += 1;
                        let entry = &mut graph.nodes[index];
                        if (!progress.major && (entry.arena || entry.cycle.as_ref().is_some_and(|slot|
                            matches!(slot.generation(), crate::heap::Generation::Old | crate::heap::Generation::StableShared))))
                            || entry.node.references() > entry.incoming + 1 + usize::from(entry.arena)
                                + usize::from(entry.cycle.as_ref().is_some_and(|slot| slot.owns_node())) {
                            entry.live = true;
                            pending.push(index)?;
                        }
                        work += 1;
                    }
                    GcPhase::Mark => {
                        if progress.mark_current.is_none() {
                            if let Some(index) = pending.pop() {
                                progress.mark_edge = graph.nodes[index].edges.start;
                                progress.mark_current = Some(index);
                                work += 1;
                                continue;
                            }
                            progress.phase = GcPhase::Promote;
                            continue;
                        }
                        let index = progress.mark_current.unwrap();
                        if progress.mark_edge >= graph.nodes[index].edges.end {
                            progress.mark_current = None;
                            continue;
                        }
                        let child = graph.edges[progress.mark_edge];
                        progress.mark_edge += 1;
                        if !graph.nodes[child].live { graph.nodes[child].live = true; pending.push(child)?; }
                        work += 1;
                    }
                    GcPhase::Promote => {
                        if progress.promote_cursor >= graph.nodes.len() { progress.phase = GcPhase::Nursery; continue; }
                        let entry = &graph.nodes[progress.promote_cursor];
                        if entry.live {
                            if entry.node.has_write_barrier() && entry.cycle.as_ref().is_some_and(|slot|
                                slot.generation() == crate::heap::Generation::Young) {
                                entry.cycle.as_ref().unwrap().promote(crate::heap::Generation::Old);
                            }
                            if let Node::Storage(storage) = &entry.node {
                                storage.generation.fetch_max(crate::heap::Generation::Old as u8, Ordering::AcqRel);
                            }
                        }
                        progress.promote_cursor += 1;
                        work += 1;
                    }
                    GcPhase::Nursery => {
                        if progress.nursery_cursor >= graph.nodes.len() { progress.phase = GcPhase::Remember; continue; }
                        let entry = &graph.nodes[progress.nursery_cursor];
                        progress.nursery_nonempty |= entry.cycle.is_none()
                            && !matches!(entry.node, Node::Class(_) | Node::Storage(_));
                        progress.nursery_cursor += 1;
                        work += 1;
                    }
                    GcPhase::Remember => {
                        if progress.remember_cursor >= graph.nodes.len() {
                            if progress.mutation_epoch != heap.mutation_epoch() {
                                progress.aborted = true;
                                progress.release_node = 0;
                                progress.phase = GcPhase::ReleaseNodes;
                                continue;
                            }
                            (context.charge)((graph.nodes.len() + graph.edges.len()) as u64)?;
                            progress.phase = GcPhase::Sweep;
                            continue;
                        }
                        let entry = &graph.nodes[progress.remember_cursor];
                        if !progress.remember_started {
                            progress.remember_edge = entry.edges.start;
                            progress.remembered = progress.nursery_nonempty;
                            progress.remember_started = true;
                            work += 1;
                            continue;
                        }
                        if progress.remember_edge < entry.edges.end {
                            let child = &graph.nodes[graph.edges[progress.remember_edge]];
                            progress.remembered |= child.node.is_young()
                                || child.cycle.as_ref().is_some_and(|slot| slot.generation() == crate::heap::Generation::Young);
                            progress.remember_edge += 1;
                            work += 1;
                            continue;
                        }
                        if entry.live {
                            if let Node::Storage(storage) = &entry.node {
                                if storage.generation.load(Ordering::Acquire) == crate::heap::Generation::Old as u8 {
                                    storage.remembered.store(progress.remembered, Ordering::Release);
                                }
                            }
                            if let Some(slot) = &entry.cycle {
                                if slot.generation() == crate::heap::Generation::Old {
                                    slot.set_remembered(progress.remembered);
                                }
                            }
                        }
                        progress.remember_cursor += 1;
                        progress.remember_started = false;
                        work += 1;
                    }
                    GcPhase::Sweep => {
                        if progress.sweep_cursor >= graph.nodes.len() {
                            progress.phase = GcPhase::RetireRoots;
                            continue;
                        }
                        let entry = &graph.nodes[progress.sweep_cursor];
                        let collectable = progress.major || entry.node.is_young()
                            || entry.cycle.as_ref().is_some_and(|slot| slot.generation() == crate::heap::Generation::Young);
                        if entry.live || !collectable {
                            progress.sweep_cursor += 1;
                            work += 1;
                            continue;
                        }
                        match clear_node_chunk(&entry.node, 1) {
                            Some(true) => {
                                // Cleared mutable owners no longer keep their
                                // former children alive during pin retirement.
                                if matches!(entry.node, Node::Storage(_) | Node::Array(_) | Node::Map(_) | Node::Set(_)
                                    | Node::Generator(_) | Node::Future(_) | Node::Reactive(_) | Node::Effect(_) | Node::Task(_)) {
                                    graph.nodes[progress.sweep_cursor].edges = 0..0;
                                }
                                progress.sweep_cursor += 1;
                            },
                            Some(false) => {},
                            None => { finish(work); return Ok(false); },
                        }
                        // Only the collector's own removals happened while this
                        // slice held the workspace. A later mutator change must
                        // invalidate sweeping just as it invalidates marking.
                        progress.mutation_epoch = heap.mutation_epoch();
                        work += 1;
                    }
                    GcPhase::RetireRoots => {
                        if progress.prune_cursor >= roots.values.len() {
                            progress.prune_cursor = 0;
                            progress.phase = if progress.major { GcPhase::CompactClasses } else { GcPhase::ReleaseCounts };
                            continue;
                        }
                        if !roots.values[progress.prune_cursor].live() { roots.values.swap_remove(progress.prune_cursor); }
                        else { progress.prune_cursor += 1; }
                        work += 1;
                    }
                    GcPhase::CompactClasses => {
                        if progress.compact_cursor >= graph.nodes.len() {
                            progress.phase = GcPhase::TrimClasses;
                            continue;
                        }
                        let entry = &graph.nodes[progress.compact_cursor];
                        if entry.arena {
                            if entry.live {
                                classes.swap(progress.class_read, progress.class_write);
                                progress.class_write += 1;
                            }
                            progress.class_read += 1;
                        }
                        progress.compact_cursor += 1;
                        work += 1;
                    }
                    GcPhase::TrimClasses => {
                        if classes.len() <= progress.class_write {
                            progress.phase = GcPhase::ReleaseCounts;
                            continue;
                        }
                        classes.pop();
                        work += 1;
                    }
                    GcPhase::ReleaseCounts => {
                        if progress.release_node < graph.nodes.len() {
                            graph.nodes[progress.release_node].incoming = 0;
                            progress.release_node += 1;
                            work += 1;
                        } else {
                            progress.release_node = 0;
                            progress.release_edge = graph.nodes.first().map_or(0, |entry| entry.edges.start);
                            progress.phase = GcPhase::ReleaseCountEdges;
                        }
                    }
                    GcPhase::ReleaseCountEdges => {
                        if progress.release_node >= graph.nodes.len() {
                            progress.release_node = 0;
                            progress.phase = GcPhase::ReleaseSeeds;
                            continue;
                        }
                        if progress.release_edge < graph.nodes[progress.release_node].edges.end {
                            let child = graph.edges[progress.release_edge];
                            graph.nodes[child].incoming += 1;
                            progress.release_edge += 1;
                        } else {
                            progress.release_node += 1;
                            progress.release_edge = graph.nodes.get(progress.release_node).map_or(0, |entry| entry.edges.start);
                        }
                        work += 1;
                    }
                    GcPhase::ReleaseSeeds => {
                        if progress.release_node < graph.nodes.len() {
                            if graph.nodes[progress.release_node].incoming == 0 { pending.push(progress.release_node)?; }
                            progress.release_node += 1;
                            work += 1;
                        } else {
                            progress.release_node = 0;
                            progress.phase = GcPhase::ReleaseParents;
                        }
                    }
                    GcPhase::ReleaseParents => {
                        if let Some(retiring) = &mut progress.retiring {
                            if retiring.step() { progress.retiring = None; progress.phase = GcPhase::ReleaseChildren; }
                            work += 1;
                            continue;
                        }
                        let Some(index) = pending.pop() else { progress.release_node = 0; progress.phase = GcPhase::ReleaseNodes; continue; };
                        progress.release_node = index;
                        progress.release_edge = graph.nodes[index].edges.start;
                        let entry = &mut graph.nodes[index];
                        if let Some(slot) = &entry.cycle { slot.set_tracing(false); }
                        if !entry.live {
                            if let Some(slot) = &entry.cycle {
                                if !slot.release_owner() { pending.push(index)?; finish(work); return Ok(false); }
                            }
                        }
                        progress.retiring = RetiringNode::from_node(std::mem::replace(&mut entry.node, Node::Vacant));
                        if progress.retiring.is_none() { progress.phase = GcPhase::ReleaseChildren; }
                        work += 1;
                    }
                    GcPhase::ReleaseChildren => {
                        if progress.release_edge < graph.nodes[progress.release_node].edges.end {
                            let child = graph.edges[progress.release_edge];
                            graph.nodes[child].incoming -= 1;
                            if graph.nodes[child].incoming == 0 { pending.push(child)?; }
                            progress.release_edge += 1;
                            work += 1;
                        } else { progress.phase = GcPhase::ReleaseParents; }
                    }
                    GcPhase::ReleaseNodes => {
                        if let Some(retiring) = &mut progress.retiring {
                            if retiring.step() { progress.retiring = None; }
                            work += 1;
                            continue;
                        }
                        if progress.release_node < graph.nodes.len() {
                            let entry = &mut graph.nodes[progress.release_node];
                            if let Some(slot) = &entry.cycle { slot.set_tracing(false); }
                            if !progress.aborted && !entry.live {
                                if let Some(slot) = &entry.cycle {
                                    if !slot.release_owner() { finish(work); return Ok(false); }
                                }
                            }
                            progress.retiring = RetiringNode::from_node(std::mem::replace(&mut entry.node, Node::Vacant));
                            progress.release_node += 1;
                            work += 1;
                        } else { progress.phase = GcPhase::TrimNodes; }
                    }
                    GcPhase::TrimNodes => {
                        if graph.nodes.pop().is_some() { work += 1; }
                        else { progress.phase = GcPhase::ReleaseEdges; }
                    }
                    GcPhase::ReleaseEdges => {
                        if graph.edges.pop().is_some() { work += 1; }
                        else { progress.phase = GcPhase::ReleaseIndex; }
                    }
                    GcPhase::ReleaseIndex => {
                        if progress.release_index < graph.index.len() {
                            graph.index[progress.release_index] = None;
                            progress.release_index += 1;
                            work += 1;
                        } else { progress.phase = GcPhase::ReleasePending; }
                    }
                    GcPhase::ReleasePending => {
                        if pending.pop().is_some() { work += 1; }
                        else if progress.aborted {
                            workspace.progress = None;
                            owner.active_collection.store(0, Ordering::Release);
                            finish(work);
                            return Ok(false);
                        } else { progress.phase = GcPhase::PruneCycles; }
                    }
                    GcPhase::PruneCycles => {
                        if progress.prune_cursor >= heap.cycle_count() {
                            progress.prune_cursor = 0;
                            progress.phase = GcPhase::PruneOwned;
                            continue;
                        }
                        if !heap.prune_cycle_at(progress.prune_cursor) { progress.prune_cursor += 1; }
                        work += 1;
                    }
                    GcPhase::PruneOwned => {
                        if progress.prune_cursor >= owned_cycles.len() {
                            progress.prune_cursor = 0;
                            progress.phase = GcPhase::Finish;
                            continue;
                        }
                        let Ok(node) = owned_cycles[progress.prune_cursor].node.try_lock() else { finish(work); return Ok(false); };
                        let alive = node.as_ref().is_some_and(crate::heap::CycleWeak::alive);
                        drop(node);
                        if alive {
                            progress.prune_cursor += 1;
                        } else { owned_cycles.swap_remove(progress.prune_cursor); }
                        work += 1;
                    }
                    GcPhase::Finish => {
                        workspace.progress = None;
                        owner.active_collection.store(0, Ordering::Release);
                        heap.finish_cycle_slice(major);
                        finish(work);
                        return Ok(true);
                    }
                }
            }
            finish(work);
            Ok(false)
        })();
        workspace.graph.incremental = false;
        let completed = match result {
            Ok(completed) => completed,
            Err(error) => {
                if let Some(progress) = &mut workspace.progress {
                    progress.aborted = true;
                    progress.release_node = 0;
                    progress.phase = GcPhase::ReleaseNodes;
                }
                self.class_objects = classes;
                return Err(error);
            }
        };
        self.class_objects = classes;
        if completed {
            owner.active_collection.store(0, Ordering::Release);
            self.class_allocations_since_gc = self.class_objects.len().saturating_sub(class_limit);
            if self.class_objects.is_empty() {
                self.class_objects = Vec::new();
                self.class_objects_reservation.lock().unwrap().resize(0)?;
            }
        }
        Ok(completed)
    }
}

fn collect(classes: &mut Vec<Arc<ClassObject>>, roots: &mut Buffer<RootRecord>, vm: &mut Context<'_>, workspace: &mut Workspace, major: bool, before_mark: &mut dyn FnMut()) -> Result<bool, LanaError> {
    let result = collect_graph(classes, roots, vm, workspace, major, before_mark);
    if result == Ok(true) { roots.retain(RootRecord::live); }
    let empty = result == Ok(true) && !workspace.graph.nodes.iter().any(|entry| entry.live);
    if result == Ok(true) {
        // Release parents while child pins still exist. Registry order need not
        // match graph order, and dropping a deep immutable chain must not recurse.
        let Workspace { graph, pending, .. } = workspace;
        pending.clear();
        for entry in graph.nodes.iter_mut() { entry.incoming = 0; }
        for index in 0..graph.nodes.len() {
            for edge in graph.nodes[index].edges.clone() {
                let child = graph.edges[edge];
                graph.nodes[child].incoming += 1;
            }
        }
        for index in 0..graph.nodes.len() {
            if graph.nodes[index].incoming == 0 { pending.push(index)?; }
        }
        while let Some(index) = pending.pop() {
            drop(std::mem::replace(&mut graph.nodes[index].node, Node::Vacant));
            for edge in graph.nodes[index].edges.clone() {
                let child = graph.edges[edge];
                graph.nodes[child].incoming -= 1;
                if graph.nodes[child].incoming == 0 { pending.push(child)?; }
            }
        }
    }
    workspace.clear();
    if empty { *workspace = Workspace::new(vm.heap); }
    if result == Ok(true) { vm.heap.finish_cycles(major); }
    result
}

fn cycle_node(slot: &Arc<crate::heap::CycleSlot>) -> Option<Node> {
    slot.node.try_lock().ok()?.as_ref()?.strong_node()
}

fn distribution_child(value: &StateDist, cursor: usize) -> Option<Arc<StateDist>> {
    match &value.kind {
        StateDistKind::Append { left, right, .. } => match if cursor == 0 { left } else { right } {
            DistOperand::Node(child) => Some(child.clone()), DistOperand::Inline(_) => None,
        },
        StateDistKind::Transform { child, .. } | StateDistKind::Attenuate { child, .. } if cursor == 0 => Some(child.clone()),
        _ => None,
    }
}

fn clear_node_chunk(node: &Node, max: usize) -> Option<bool> {
    macro_rules! lock {
        ($value:expr) => { match $value.try_lock() { Ok(value) => value, Err(_) => return None } };
    }
    fn keep_tail<T>(items: &mut Vec<T>, max: usize) -> bool {
        items.truncate(items.len().saturating_sub(max));
        items.is_empty()
    }
    Some(match node {
        Node::Storage(v) => { let mut fields = lock!(v.fields); keep_tail(&mut fields, max) },
        Node::Array(v) => { let mut value = lock!(v); let len = value.items.len(); value.items.truncate(len.saturating_sub(max)); value.items.is_empty() },
        Node::Map(v) => { let mut value = lock!(v); let len = value.entries.len(); value.entries.truncate(len.saturating_sub(max)); value.entries.is_empty() },
        Node::Set(v) => { let mut value = lock!(v); let len = value.items.len(); value.items.truncate(len.saturating_sub(max)); value.items.is_empty() },
        Node::Generator(v) => keep_tail(&mut lock!(v).registers, max),
        Node::Future(v) => keep_tail(&mut lock!(v).registers, max),
        Node::Task(v) => {
            let root = {
                let mut state = lock!(v.state);
                if !state.joined && state.result_root.is_some() && max != usize::MAX { return None; }
                state.result = Value::null();
                state.result_root.take()
            };
            drop(root);
            true
        },
        Node::Reactive(v) => {
            let mut value = lock!(v);
            if let Some(input) = value.inputs.iter_mut().find(|input| input.is_some()) { *input = None; false }
            else if let Some(input) = value.constants.iter_mut().find(|input| input.is_some()) { *input = None; false }
            else if value.current.take().is_some() { false }
            else { keep_tail(&mut value.history, max) }
        }
        Node::Effect(v) => keep_tail(&mut lock!(v.state).receipts, max),
        _ => true,
    })
}

fn collect_graph(classes: &mut Vec<Arc<ClassObject>>, roots: &[RootRecord], vm: &mut Context<'_>, workspace: &mut Workspace, major: bool, before_mark: &mut dyn FnMut()) -> Result<bool, LanaError> {
    let epoch = vm.heap.mutation_epoch();
    let Workspace { graph, pending, .. } = workspace;
    for index in 0..classes.len() {
        let object = classes[index].clone();
        let entry = graph.add(Node::Storage(object.clone()), false, vm)?;
        graph.nodes[entry].arena = true;
        if let Some(reference) = object.reference.upgrade() { graph.add(Node::Class(reference), false, vm)?; }
    }
    for index in (0..vm.heap.cycle_count()).rev() {
        (vm.charge)(1)?;
        let Some(slot) = vm.heap.cycle_slot_at(index) else { continue; };
        let Ok(weak) = slot.node.try_lock() else { return Ok(false); };
        let node = weak.as_ref().and_then(crate::heap::CycleWeak::strong_node);
        drop(weak);
        if let Some(node) = node {
            graph.add_cycle(node, false, Some(slot), vm)?;
        }
    }
    for root in roots.iter().filter(|root| root.live()) { graph.value(&root.value, vm)?; }
    let root_edge_end = graph.edges.len();
    for root in roots.iter().filter(|root| !root.live()) { graph.value(&root.value, vm)?; }
    for entry in &graph.nodes {
        if let Some(slot) = &entry.cycle { slot.set_tracing(true); }
    }
    let mut index = 0;
        while index < graph.nodes.len() {
            let start = graph.edges.len();
        let old_clean_cycle = graph.nodes[index].cycle.as_ref().is_some_and(|slot|
            slot.generation() == crate::heap::Generation::Old && !slot.remembered());
        let old_clean_class = match &graph.nodes[index].node {
            Node::Storage(storage) => storage.generation.load(Ordering::Acquire) == crate::heap::Generation::Old as u8
                && !storage.remembered.load(Ordering::Acquire),
            _ => false,
        };
        if !major && (old_clean_cycle || old_clean_class) {
            graph.nodes[index].edges = start..start;
            index += 1;
            continue;
        }
        let node = graph.nodes[index].node.clone();
        if !graph.trace(&node, vm)? { return Ok(false); }
        graph.nodes[index].edges = start..graph.edges.len();
        index += 1;
    }
    pending.reserve(graph.nodes.len())?;
    if vm.heap.mutation_epoch() != epoch { return Ok(false); }
    before_mark();
    if vm.heap.mutation_epoch() != epoch { return Ok(false); }
    for edge in 0..root_edge_end {
        let index = graph.edges[edge];
        if !graph.nodes[index].live { graph.nodes[index].live = true; pending.push(index)?; }
    }
    for (index, entry) in graph.nodes.iter_mut().enumerate() {
        // Ignore exactly one collector reference and the task's arena owner.
        // All other untraced owners are roots, including Rust locals, host
        // aliases, suspended frames, and task or shared Information state.
        if !major && (entry.arena || entry.cycle.as_ref().is_some_and(|slot|
                matches!(slot.generation(), crate::heap::Generation::Old | crate::heap::Generation::StableShared)))
            || entry.node.references() > entry.incoming + 1 + usize::from(entry.arena)
                + usize::from(entry.cycle.as_ref().is_some_and(|slot| slot.owns_node())) {
            entry.live = true;
            pending.push(index)?;
        }
    }
    while let Some(index) = pending.pop() {
        for edge in graph.nodes[index].edges.clone() {
            (vm.charge)(1)?;
            let child = graph.edges[edge];
            if !graph.nodes[child].live {
                graph.nodes[child].live = true;
                pending.push(child)?;
            }
        }
    }
    // Admit all remaining budgeted work before changing generation metadata
    // or clearing an unreachable graph node.
    (vm.charge)((graph.nodes.len() + graph.edges.len()) as u64)?;
    if vm.heap.mutation_epoch() != epoch { return Ok(false); }
    for entry in &graph.nodes {
        if entry.live && entry.node.has_write_barrier() && entry.cycle.as_ref().is_some_and(|slot|
            slot.generation() == crate::heap::Generation::Young) {
            entry.cycle.as_ref().unwrap().promote(crate::heap::Generation::Old);
        }
        if entry.live {
            if let Node::Storage(storage) = &entry.node {
                storage.generation.fetch_max(crate::heap::Generation::Old as u8, Ordering::AcqRel);
            }
        }
    }
    // Foreign immutable payloads have no local age. Preserve conservative
    // tracing for that boundary; registered owners remember actual young edges.
    let nursery_nonempty = graph.nodes.iter().any(|entry| entry.cycle.is_none()
        && !matches!(entry.node, Node::Class(_) | Node::Storage(_)));
    for index in 0..graph.nodes.len() {
        let entry = &graph.nodes[index];
        if let Node::Storage(storage) = &entry.node {
            if entry.live && storage.generation.load(Ordering::Acquire) == crate::heap::Generation::Old as u8 {
                let remembered = nursery_nonempty || entry.edges.clone().any(|edge| {
                    graph.nodes[graph.edges[edge]].node.is_young()
                        || graph.nodes[graph.edges[edge]].cycle.as_ref()
                            .is_some_and(|child| child.generation() == crate::heap::Generation::Young)
                });
                storage.remembered.store(remembered, Ordering::Release);
            }
        }
        let Some(slot) = &entry.cycle else { continue; };
        if !entry.live || slot.generation() != crate::heap::Generation::Old { continue; }
        let remembered = nursery_nonempty || entry.edges.clone().any(|edge| {
            graph.nodes[graph.edges[edge]].cycle.as_ref()
                .is_some_and(|child| child.generation() == crate::heap::Generation::Young)
        });
        slot.set_remembered(remembered);
    }
    // Finish every fallible operation before mutation. Keeping the graph's
    // references until all cycles are broken also avoids recursive disposal
    // of class chains. Only task-owned class storage is removed from the arena.
    for entry in &graph.nodes {
        let collectable = major || entry.node.is_young()
            || entry.cycle.as_ref().is_some_and(|slot| slot.generation() == crate::heap::Generation::Young);
        if entry.live || !collectable { continue; }
        loop {
            match clear_node_chunk(&entry.node, usize::MAX) {
                Some(true) => break,
                Some(false) => {},
                None => return Ok(false),
            }
        }
        if let Some(slot) = &entry.cycle {
            if !slot.release_owner() { return Ok(false); }
        }
    }
    // Arena nodes were inserted in arena order, interleaved only with their
    // reference nodes. This sweep needs no lookup or unbudgeted allocation.
    if major {
        let mut arena_nodes = graph.nodes.iter().filter(|entry| entry.arena);
        classes.retain(|_| arena_nodes.next().unwrap().live);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lana_bytecode::assembler;

    #[test]
    fn full_collection_defers_mutation_after_marking_before_sweep() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let live = vm.array_value(vec![Value::null()]).unwrap();
        let child = vm.array_value(vec![Value::number(9.0)]).unwrap();
        let ValueKind::Array(child_array) = &child.kind else { unreachable!() };
        let child_weak = Arc::downgrade(child_array);
        let dead = vm.array_value(vec![child]).unwrap();
        let child = child_weak;
        let ValueKind::Array(dead_array) = &dead.kind else { unreachable!() };
        let dead_array = Arc::downgrade(dead_array);
        drop(dead);
        let root = vm.retain_value(&live).unwrap();
        let ValueKind::Array(live_array) = &live.kind else { unreachable!() };
        let live_array = live_array.clone();
        drop(live);
        let heap = vm.heap();
        let mut workspace = Workspace::new(&heap);
        let roots = vm.host_roots.roots.lock().unwrap();
        let mut moved = false;
        let marked = std::cell::Cell::new(false);
        let result = collect_graph(&mut vm.class_objects, &roots.values,
            &mut Context { heap: &heap, charge: &mut |steps| {
                if steps > 1 && marked.get() && !moved {
                    let value = dead_array.upgrade().unwrap().lock().unwrap().items.pop().unwrap();
                    live_array.lock().unwrap().items[0] = value;
                    moved = true;
                }
                Ok(())
            } }, &mut workspace, true, &mut || marked.set(true));
        drop(roots);
        assert!(moved);
        assert_eq!(result, Ok(false));
        assert_eq!(child.upgrade().unwrap().lock().unwrap().items.len(), 1);
        workspace.clear();
        vm.collect_classes().unwrap();
        assert_eq!(root.child(0).unwrap().unwrap().child(0).unwrap().unwrap().as_number(), 9.0);
    }

    #[test]
    fn ending_thread_restores_queue_ownership_for_retained_heap() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let value = vm.array_value(Vec::new()).unwrap();
        let ValueKind::Array(array) = &value.kind else { unreachable!() };
        let weak = Arc::downgrade(array);
        array.lock().unwrap().items.push(value.clone()).unwrap();
        let root = vm.retain_value(&value).unwrap();
        let child = root.child(0).unwrap().unwrap();
        drop(value);
        let owner = vm.host_roots.clone();
        let heap = vm.heap();
        drop(vm);
        let roots = owner.roots.lock().unwrap();
        std::thread::spawn(move || drop(child)).join().unwrap();
        assert!(!owner.queued.load(Ordering::Acquire));
        drop(roots);
        drop(root);
        assert!(weak.upgrade().is_none());
        drop(owner);
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn ending_thread_flushes_deferred_retired_owners_without_recursion() {
        let worker = std::thread::spawn(|| {
            let chunk = assembler::assemble("HALT\n").unwrap();
            let mut root: Option<RootedValue> = None;
            let mut owners = Vec::new();
            for id in 0..20_000 {
                let mut vm = Vm::new(&chunk);
                let task = Arc::new(Task::new(id, 0));
                {
                    let mut state = task.state.lock().unwrap();
                    state.completed = true;
                    state.result = root.as_ref().map_or_else(Value::null, |root| root.value.clone());
                    state.result_root = root.take();
                }
                vm.track_cycle(crate::heap::CycleWeak::Task(Arc::downgrade(&task))).unwrap();
                vm.result = Value::task(task);
                root = Some(vm.result().unwrap());
                owners.push(Arc::downgrade(&vm.host_roots));
                drop(vm);
            }
            let _defer = RootOwner::defer_retired();
            drop(root);
            owners
        });
        let owners = worker.join().unwrap();
        assert!(owners.iter().all(|owner| owner.upgrade().is_none()));
    }

    #[test]
    fn full_graph_duplicate_lookup_uses_admitted_capacity_at_half_load() {
        let heap = Heap::new(1024 * 1024);
        let mut workspace = Workspace::new(&heap);
        workspace.admit(&heap, 8, 0).unwrap();
        let mut context = Context { heap: &heap, charge: &mut |_| Ok(()) };
        for _ in 0..8 {
            workspace.graph.add(Node::Adt(Arc::new(Adt { variant: 0, fields: vec![] })), false, &mut context).unwrap();
        }
        assert_eq!(workspace.graph.index.len(), 16);
        let bytes = heap.live_bytes();
        heap.set_limit(bytes).unwrap();
        let node = workspace.graph.nodes[0].node.clone();
        assert_eq!(workspace.graph.add(node, true, &mut context).unwrap(), 0);
        assert_eq!(heap.live_bytes(), bytes);
        assert_eq!(workspace.graph.index.len(), 16);
    }

    #[test]
    fn root_release_does_not_wait_for_the_registry_lock() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        for retired in [false, true] {
            let mut vm = Vm::new(&chunk);
            let value = vm.array_value(Vec::new()).unwrap();
            let ValueKind::Array(array) = &value.kind else { unreachable!() };
            let weak = Arc::downgrade(array);
            array.lock().unwrap().items.push(value.clone()).unwrap();
            let root = vm.retain_value(&value).unwrap();
            drop(value);
            let owner = vm.host_roots.clone();
            let mut vm = Some(vm);
            if retired { drop(vm.take()); }
            let roots = owner.roots.lock().unwrap();
            let (sender, receiver) = std::sync::mpsc::channel();
            let worker = std::thread::spawn(move || { drop(root); sender.send(()).unwrap(); });
            let released = receiver.recv_timeout(std::time::Duration::from_secs(1));
            if released.is_ok() { assert!(roots.values.iter().all(|root| !root.live())); }
            drop(roots);
            worker.join().unwrap();
            assert!(released.is_ok());
            if let Some(vm) = &mut vm { vm.collect_classes().unwrap(); }
            else { assert!(owner.collect().unwrap()); }
            assert!(weak.upgrade().is_none());
        }
    }

    #[test]
    fn nested_task_heap_owners_release_without_recursive_collection() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut root: Option<RootedValue> = None;
        let mut owners = Vec::new();
        for id in 0..20_000 {
            let mut vm = Vm::new(&chunk);
            let task = Arc::new(Task::new(id, 0));
            {
                let mut state = task.state.lock().unwrap();
                state.completed = true;
                state.result = root.as_ref().map_or_else(Value::null, |root| root.value.clone());
                state.result_root = root.take();
            }
            vm.track_cycle(crate::heap::CycleWeak::Task(Arc::downgrade(&task))).unwrap();
            vm.result = Value::task(task);
            root = Some(vm.result().unwrap());
            owners.push(Arc::downgrade(&vm.host_roots));
            drop(vm);
        }
        drop(root);
        assert!(owners.iter().all(|owner| owner.upgrade().is_none()));
    }

    #[test]
    fn routine_collection_reclaims_joined_task_cycles_within_slice_bound() {
        let chunk = assembler::assemble(
            ".function main 0 4\nFORK worker R1 0 R0\nJOIN R0 R1\nRETURN R0\n.function worker 0 2\nARRAY_NEW R0 R1 0\nRETURN R0\n"
        ).unwrap();
        let mut vm = Vm::new(&chunk);
        vm.configured_worker_count = 0;
        assert_eq!(vm.run(), LanaError::Ok);
        let ValueKind::Task(task) = &vm.result.kind else { unreachable!() };
        let task_weak = Arc::downgrade(task);
        let state = task.state.lock().unwrap();
        let ValueKind::Array(array) = &state.result.kind else { unreachable!() };
        let array_weak = Arc::downgrade(array);
        array.lock().unwrap().items.push(Value::task(task.clone())).unwrap();
        drop(state);
        vm.heap.mutate_cycle(array_weak.as_ptr() as usize);
        vm.frames[0].registers.fill(Value::null());
        vm.result = Value::null();
        let mut complete = false;
        for _ in 0..1_000 {
            complete = vm.collect_slice(true, 1).unwrap();
            assert!(vm.host_roots.last_slice_work() <= 1);
            if complete { break; }
        }
        assert!(complete);
        assert!(task_weak.upgrade().is_none());
        assert!(array_weak.upgrade().is_none());
    }

    #[test]
    fn shutdown_and_last_host_root_collect_standalone_cycles() {
        let chunk = assembler::assemble(".function main 0 1\nHALT\n").unwrap();
        for retain in [false, true] {
            for _ in 0..32 {
                let mut vm = Vm::new(&chunk);
                let array = vm.array_value(vec![Value::null()]).unwrap();
                let ValueKind::Array(items) = &array.kind else { unreachable!() };
                let weak = Arc::downgrade(items);
                items.lock().unwrap().items[0] = array.clone();
                vm.result = array;
                let root = retain.then(|| vm.result().unwrap());
                if let Some(root) = &root {
                    assert_eq!(vm.host_roots.roots.lock().unwrap().values.len(), 1);
                    let alias = root.clone();
                    drop(root.clone());
                    assert_eq!(vm.host_roots.roots.lock().unwrap().values.len(), 1);
                    drop(alias);
                    assert_eq!(vm.host_roots.roots.lock().unwrap().values.len(), 1);
                }
                let heap = vm.heap();
                drop(vm);
                assert_eq!(weak.upgrade().is_some(), retain);
                if let Some(root) = &root { assert_eq!(root.print(), "[<cycle>]"); }
                drop(root);
                assert!(weak.upgrade().is_none());
                assert_eq!(heap.live_bytes(), 0);
            }
        }
    }

    #[test]
    fn explicit_host_roots_validate_heap_locks_and_budget_before_retention() {
        let chunk = assembler::assemble(".function main 0 1\nHALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let other = Vm::new(&chunk);
        let foreign = other.array_value(vec![]).unwrap();
        assert!(matches!(vm.retain_value(&foreign), Err(LanaError::Task)));
        let array = vm.array_value(vec![Value::null()]).unwrap();
        let ValueKind::Array(items) = &array.kind else { unreachable!() };
        let weak = Arc::downgrade(items);
        let mut locked = items.lock().unwrap();
        locked.items[0] = array.clone();
        assert!(matches!(vm.retain_value(&array), Err(LanaError::UnsupportedOperation)));
        drop(locked);
        vm.set_instruction_limit(0);
        assert!(matches!(vm.retain_value(&array), Err(LanaError::Limit)));
        vm.set_instruction_limit(50_000_000);
        let root = vm.retain_value(&array).unwrap();
        let heap = vm.heap();
        drop(array);
        drop(vm);
        assert_eq!(root.print(), "[<cycle>]");
        drop(root);
        assert!(weak.upgrade().is_none());
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn result_root_registration_obeys_heap_limit() {
        let chunk = assembler::assemble(".function main 0 1\nHALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let value = vm.array_value(vec![Value::null()]).unwrap();
        vm.result = value;
        let heap = vm.heap();
        let before = heap.live_bytes();
        heap.set_limit(before).unwrap();
        assert!(matches!(vm.result(), Err(LanaError::Oom)));
        assert_eq!(heap.live_bytes(), before);
        drop(vm);
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn admitted_workspace_reclaims_cycles_without_shutdown_headroom() {
        let chunk = assembler::assemble(".function main 0 1\nHALT\n").unwrap();
        for width in [1, 128, 20_000] {
            let mut vm = Vm::new(&chunk);
            let array = vm.array_value(vec![Value::null(); width]).unwrap();
            let ValueKind::Array(items) = &array.kind else { unreachable!() };
            let weak = Arc::downgrade(items);
            for index in 0..width {
                items.lock().unwrap().items[index] = vm.array_value(vec![array.clone()]).unwrap();
            }
            let root = vm.retain_value(&array).unwrap();
            let alias = root.clone();
            drop(array);
            let heap = vm.heap();
            heap.set_limit(heap.live_bytes()).unwrap();
            let allocations = heap.allocations();
            drop(vm);
            drop(root);
            let ValueKind::Array(items) = &alias.value.kind else { unreachable!() };
            assert_eq!(items.lock().unwrap().items.len(), width);
            drop(alias);
            assert!(weak.upgrade().is_none());
            assert_eq!(heap.allocations(), allocations);
            assert_eq!(heap.live_bytes(), 0);
        }
    }

    #[test]
    fn shutdown_scratch_exhaustion_preserves_a_retained_graph() {
        let chunk = assembler::assemble(".function main 0 1\nHALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let array = vm.array_value(vec![Value::null()]).unwrap();
        let ValueKind::Array(items) = &array.kind else { unreachable!() };
        let weak = Arc::downgrade(items);
        items.lock().unwrap().items[0] = array.clone();
        vm.result = array;
        let root = vm.result().unwrap();
        let heap = vm.heap();
        heap.set_limit(heap.live_bytes()).unwrap();
        drop(vm);
        assert_eq!(root.print(), "[<cycle>]");
        // Root admission already reserved the shutdown workspace. No fresh
        // headroom or caller-driven collection is needed for the final release.
        let allocations = heap.allocations();
        drop(root);
        assert_eq!(heap.allocations(), allocations);
        assert!(weak.upgrade().is_none());
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn shutdown_releases_unjoined_children_without_workers() {
        let chunk = assembler::assemble(
            ".function main 0 8\nLOAD_CONST R0 7\nARRAY_NEW R1 R0 1\nFORK worker R1 1 R2\nRETURN R0\n.function worker 1 2\nRETURN R0\n"
        ).unwrap();
        let mut vm = Vm::new(&chunk);
        vm.configured_worker_count = 0; // The wasm scheduler runs without OS workers.
        assert_eq!(vm.run(), LanaError::Ok);
        let child_heap = {
            let state = vm.scheduler.as_ref().unwrap().state.lock().unwrap();
            assert_eq!(state.queue.len(), 1);
            state.queue[0].child.heap()
        };
        assert!(child_heap.live_bytes() > 0);
        drop(vm);
        child_heap.collect_strings();
        assert_eq!(child_heap.live_bytes(), 0);
    }

    #[test]
    fn container_collection_preserves_host_aliases_and_breaks_standalone_cycles() {
        let chunk = assembler::assemble(".function main 0 1\nHALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let array = vm.array_value(vec![Value::null()]).unwrap();
        let map = Value::map(Arc::new(Mutex::new(Map::new(&vm.heap, 1).unwrap())));
        let ValueKind::Array(items) = &array.kind else { unreachable!() };
        let alias = items.clone();
        let weak = Arc::downgrade(items);
        items.lock().unwrap().items[0] = map.clone();
        let ValueKind::Map(entries) = &map.kind else { unreachable!() };
        entries.lock().unwrap().set(Arc::from("cycle"), array.clone(), true).unwrap();
        drop(array);
        drop(map);
        vm.collect_classes().unwrap();
        assert_eq!(alias.lock().unwrap().items.len(), 1);
        drop(alias);
        vm.collect_classes().unwrap();
        assert!(weak.upgrade().is_none());
        vm.heap.collect_strings();
        let heap = vm.heap.clone();
        drop(vm);
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn remembered_old_array_keeps_then_releases_young_child() {
        let chunk = assembler::assemble(".function main 0 1\nHALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let old = vm.array_value(vec![Value::null()]).unwrap();
        let ValueKind::Array(old_array) = &old.kind else { unreachable!() };
        vm.collect_generation(false).unwrap();
        assert_eq!(old_array.lock().unwrap().cycle.generation(), crate::heap::Generation::Old);

        let young = vm.array_value(vec![Value::null()]).unwrap();
        let ValueKind::Array(young_array) = &young.kind else { unreachable!() };
        let weak = Arc::downgrade(young_array);
        old_array.lock().unwrap().items.push(young.clone()).unwrap();
        drop(young);
        vm.collect_generation(false).unwrap();
        assert!(weak.upgrade().is_some(), "remembered edge must retain young child");

        old_array.lock().unwrap().items.clear();
        vm.collect_generation(false).unwrap();
        assert!(weak.upgrade().is_some(), "promoted child remains owned until major collection");
        vm.collect_generation(true).unwrap();
        assert!(weak.upgrade().is_none(), "removed edge must allow promoted child reclamation");
    }

    #[test]
    fn suspended_owner_promotes_and_remembers_only_its_writes() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let future = Arc::new(Mutex::new(Future {
            function: 0, ip: 0, registers: vec![Value::null()],
            exhausted: false, ready: false, queued: false,
        }));
        vm.track_cycle(crate::heap::CycleWeak::Future(Arc::downgrade(&future))).unwrap();
        let unrelated = vm.array_value(vec![]).unwrap();
        let ValueKind::Array(unrelated_array) = &unrelated.kind else { unreachable!() };
        let unrelated_slot = unrelated_array.lock().unwrap().cycle.clone();
        vm.collect_generation(true).unwrap();
        let slot = vm.owned_cycles.iter().find(|slot| slot.node.lock().unwrap().as_ref()
            .is_some_and(|node| node.identity() == Arc::as_ptr(&future) as usize)).unwrap().clone();
        assert_eq!(slot.generation(), crate::heap::Generation::Old);
        assert!(!slot.remembered());
        assert!(!unrelated_slot.remembered());
        let child = vm.array_value(vec![]).unwrap();
        let ValueKind::Array(child_array) = &child.kind else { unreachable!() };
        let weak = Arc::downgrade(child_array);
        vm.heap.mutate_cycle(Arc::as_ptr(&future) as usize);
        future.lock().unwrap().registers[0] = child;
        assert!(slot.remembered());
        assert!(!unrelated_slot.remembered());
        vm.collect_generation(false).unwrap();
        assert!(weak.upgrade().is_some());
        assert!(!slot.remembered());
        vm.heap.mutate_cycle(Arc::as_ptr(&future) as usize);
        future.lock().unwrap().registers[0] = Value::null();
        vm.collect_generation(true).unwrap();
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn last_scalar_string_root_releases_its_heap_charge() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        vm.result = vm.string_value("retained").unwrap();
        let root = vm.result().unwrap();
        let heap = vm.heap();
        drop(vm);
        assert_eq!(root.as_string().as_ref(), "retained");
        drop(root);
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn array_buffer_admits_one_header_when_published() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let items = vm.allocate_array_items(3).unwrap();
        assert_eq!(vm.host_roots.admitted_nodes.load(Ordering::Acquire), 0);
        assert_eq!(vm.host_roots.admitted_edges.load(Ordering::Acquire), 0);
        let array = Array::from_buffer(items).unwrap();
        assert_eq!(vm.host_roots.admitted_nodes.load(Ordering::Acquire), 1);
        assert_eq!(vm.host_roots.admitted_edges.load(Ordering::Acquire), 15);
        drop(array);
        assert_eq!(vm.host_roots.admitted_nodes.load(Ordering::Acquire), 0);
        assert_eq!(vm.host_roots.admitted_edges.load(Ordering::Acquire), 0);
    }

    #[test]
    fn full_fallback_reclaims_cycles_pinned_by_an_interrupted_slice() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let value = vm.array_value(vec![Value::null()]).unwrap();
        let ValueKind::Array(array) = &value.kind else { unreachable!() };
        array.lock().unwrap().items[0] = value.clone();
        let weak = Arc::downgrade(array);
        drop(value);
        for _ in 0..10 {
            vm.collect_slice(true, 1).unwrap();
            if !vm.host_roots.workspace.lock().unwrap().graph.nodes.is_empty() { break; }
        }
        assert!(!vm.host_roots.workspace.lock().unwrap().graph.nodes.is_empty());
        assert!(weak.upgrade().is_some());
        vm.collect_classes().unwrap();
        assert!(weak.upgrade().is_none());
        assert!(vm.host_roots.workspace.lock().unwrap().progress.is_none());
        assert!(vm.host_roots.collection_major().is_none());
    }

    #[test]
    fn collision_probes_resume_one_at_a_time_without_allocating() {
        let chunk = assembler::assemble(".function main 0 1\nHALT\n").unwrap();
        let vm = Vm::new(&chunk);
        let value = vm.array_value(vec![]).unwrap();
        let ValueKind::Array(array) = &value.kind else { unreachable!() };
        let node = Node::Array(array.clone());
        let identity = node.identity();
        vm.host_roots.admit(300, 0).unwrap();
        let allocations = vm.heap.allocations();
        {
            let mut workspace = vm.host_roots.workspace.lock().unwrap();
            let graph = &mut workspace.graph;
            let first = identity.rotate_right(4).wrapping_mul(0x9e3779b9) & (graph.index.len() - 1);
            for offset in 0..300 {
                let slot = (first + offset) & (graph.index.len() - 1);
                graph.index[slot] = Some((identity.wrapping_add(offset + 1), usize::MAX));
            }
            graph.queue_add(node, false, None, false, false).unwrap();
            for _ in 0..301 {
                let mut charged = 0;
                graph.add_one(&mut Context { heap: &vm.heap, charge: &mut |steps| { charged += steps; Ok(()) } }).unwrap();
                assert_eq!(charged, 1);
                assert!(graph.nodes.is_empty());
                assert!(graph.pending_add.is_some());
            }
            graph.add_one(&mut Context { heap: &vm.heap, charge: &mut |_| Ok(()) }).unwrap();
            assert_eq!(graph.nodes.len(), 1);
            assert!(graph.pending_add.is_none());
            assert_eq!(vm.heap.allocations(), allocations);
            workspace.clear();
        }
        vm.host_roots.release_admission(300, 0);
    }

    #[test]
    fn one_unit_slice_adds_at_most_one_metadata_edge() {
        let chunk = assembler::assemble(".function main 0 1\nHALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let payload = vm.array_value(vec![Value::number(7.0)]).unwrap();
        let mut value = vm.planned_effect("copy", &payload).unwrap();
        value.kind = payload.kind.clone();
        value.claim = Some(vm.managed_payload(Claim { value: payload.clone(), proposition: Arc::from("p"),
            exactness: DerivationExactness::Exact, tolerance: 0.0, source_valid: true }).unwrap());
        let root = vm.retain_value(&value).unwrap();
        for _ in 0..2_000 {
            let before = {
                let workspace = vm.host_roots.workspace.lock().unwrap();
                (workspace.graph.nodes.len(), workspace.graph.edges.len())
            };
            let complete = vm.collect_slice(true, 1).unwrap();
            let workspace = vm.host_roots.workspace.lock().unwrap();
            assert!(workspace.graph.nodes.len() <= before.0 + 1);
            assert!(workspace.graph.edges.len() <= before.1 + 1);
            assert!(vm.host_roots.last_slice_work() <= 1);
            if complete { assert_eq!(root.child(0).unwrap().unwrap().as_number(), 7.0); return; }
        }
        panic!("collector did not finish");
    }

    #[test]
    fn one_unit_slices_preserve_mutations_and_new_roots_through_sweep() {
        let chunk = assembler::assemble(".function main 0 1\nHALT\n").unwrap();
        for phase in [GcPhase::Cycles, GcPhase::Trace, GcPhase::Classify, GcPhase::Mark,
            GcPhase::Promote, GcPhase::Nursery, GcPhase::Remember, GcPhase::Sweep] {
            let mut vm = Vm::new(&chunk);
            vm.result = vm.array_value(vec![Value::number(7.0)]).unwrap();
            let root = vm.result().unwrap();
            let mut reached = false;
            for _ in 0..500 {
                vm.collect_slice(true, 1).unwrap();
                assert!(vm.host_roots.last_slice_work() <= 1);
                if vm.host_roots.workspace.lock().unwrap().progress.as_ref().is_some_and(|p| p.phase == phase) {
                    reached = true;
                    break;
                }
            }
            assert!(reached, "did not reach {phase:?}");
            let child = vm.array_value(vec![Value::number(9.0)]).unwrap();
            let ValueKind::Array(child_array) = &child.kind else { unreachable!() };
            let weak = Arc::downgrade(child_array);
            let ValueKind::Array(array) = &root.value.kind else { unreachable!() };
            array.lock().unwrap().push(child).unwrap();
            let second_root = vm.result().unwrap();
            drop(second_root);
            let mut finished = false;
            for _ in 0..1000 {
                if vm.collect_slice(true, 1).unwrap() { finished = true; break; }
                assert!(vm.host_roots.last_slice_work() <= 1);
            }
            assert!(finished, "did not finish after mutation in {phase:?}");
            assert_eq!(root.print(), "[7, [9]]");
            assert!(weak.upgrade().is_some());
            drop(root);
            drop(vm);
            assert!(weak.upgrade().is_none());
        }
    }

    #[test]
    fn retained_child_survives_parent_and_vm() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let child = vm.array_value(vec![Value::number(9.0)]).unwrap();
        vm.result = vm.array_value(vec![child]).unwrap();
        let parent = vm.result().unwrap();
        let child = parent.child(0).unwrap().unwrap();
        let heap = vm.heap();
        drop(vm);
        drop(parent);
        assert_eq!(child.print(), "[9]");
        drop(child);
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn deep_immutable_graph_retires_during_one_unit_collection() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let mut value = Value::number(1.0);
        for _ in 0..20_000 { value = Value::adt(vm.managed_payload(crate::value::Adt { variant: 0, fields: vec![value] }).unwrap()); }
        vm.result = value;
        let root = vm.result().unwrap();
        vm.result = Value::null();
        drop(root);
        let mut complete = false;
        for _ in 0..1_000_000 {
            if vm.collect_slice(true, 1).unwrap() { complete = true; break; }
            assert!(vm.host_roots.last_slice_work() <= 1);
        }
        assert!(complete);
        assert!(vm.host_roots.roots.lock().unwrap().values.is_empty());
    }

    #[test]
    fn unretained_execution_roots_release_deep_managed_graphs_without_headroom() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let mut value = Value::number(1.0);
        for _ in 0..20_000 {
            value = Value::adt(vm.managed_payload(crate::value::Adt {
                variant: 0, fields: vec![value],
            }).unwrap());
        }
        let ValueKind::Adt(payload) = &value.kind else { unreachable!() };
        let weak = Arc::downgrade(payload);
        vm.result = value;
        vm.set_memory_limit(vm.allocated_bytes()).unwrap();
        let heap = vm.heap();
        drop(vm);
        assert!(weak.upgrade().is_none());
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn mutator_release_and_interrupted_retirement_preserve_managed_ownership() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let kept = vm.array_value(vec![Value::null()]).unwrap();
        let mut value = Value::number(1.0);
        for _ in 0..20_000 {
            value = Value::adt(vm.managed_payload(Adt { variant: 0, fields: vec![value] }).unwrap());
        }
        let ValueKind::Adt(parent) = &value.kind else { unreachable!() };
        let weak = Arc::downgrade(parent);
        let container = vm.array_value(vec![value]).unwrap();
        let ValueKind::Array(array) = &container.kind else { unreachable!() };
        array.lock().unwrap().items.clear();
        assert!(weak.upgrade().is_some(), "collector retains the released graph until retirement");
        drop(container);
        let mut interrupted = false;
        let mut complete = false;
        for _ in 0..2_000_000 {
            complete = vm.collect_slice(true, 1).unwrap();
            assert!(vm.host_roots.last_slice_work() <= 1);
            let retiring = vm.host_roots.workspace.lock().unwrap().progress.as_ref()
                .is_some_and(|progress| progress.retiring.is_some());
            if !interrupted && retiring {
                let ValueKind::Array(array) = &kept.kind else { unreachable!() };
                array.lock().unwrap().items[0] = Value::number(9.0);
                interrupted = true;
            }
            if complete { break; }
        }
        assert!(interrupted);
        assert!(complete);
        assert!(weak.upgrade().is_none());
        assert_eq!(kept.print(), "[9]");
        drop(kept);
        let heap = vm.heap();
        vm.set_memory_limit(vm.allocated_bytes()).unwrap();
        drop(vm);
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn wide_immutable_payload_retires_one_field_per_slice() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        vm.result = Value::adt(vm.managed_payload(Adt {
            variant: 0, fields: vec![Value::number(1.0); 20_000],
        }).unwrap());
        let root = vm.result().unwrap();
        vm.result = Value::null();
        drop(root);
        let mut retired_fields = 0;
        let mut complete = false;
        for _ in 0..1_000_000 {
            let before = vm.host_roots.workspace.lock().unwrap().progress.as_ref()
                .and_then(|progress| match &progress.retiring {
                    Some(RetiringNode::Adt(value)) => Some(value.fields.len()), _ => None,
                });
            complete = vm.collect_slice(true, 1).unwrap();
            assert!(vm.host_roots.last_slice_work() <= 1);
            let after = vm.host_roots.workspace.lock().unwrap().progress.as_ref()
                .and_then(|progress| match &progress.retiring {
                    Some(RetiringNode::Adt(value)) => Some(value.fields.len()), _ => None,
                });
            if let (Some(before), Some(after)) = (before, after) {
                assert!(before.saturating_sub(after) <= 1);
                retired_fields += before.saturating_sub(after);
            }
            if complete { break; }
        }
        assert!(complete);
        assert_eq!(retired_fields, 20_000);
    }

    #[test]
    fn immutable_payload_charges_follow_collected_owners() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let baseline = vm.allocated_bytes();
        let mut released_bytes = None;
        for _ in 0..32 {
            let payload = vm.managed_payload(Adt { variant: 0, fields: vec![Value::null(); 1024] }).unwrap();
            let slot = vm.owned_cycles.last().unwrap();
            assert!(slot.payload_allocation.lock().unwrap().is_some());
            assert!(vm.allocated_bytes() >= baseline + 1024 * std::mem::size_of::<Value>());
            drop(payload);
            vm.collect_generation(true).unwrap();
            assert!(vm.owned_cycles.is_empty());
            let released = *released_bytes.get_or_insert_with(|| vm.allocated_bytes());
            assert_eq!(vm.allocated_bytes(), released);
        }
        vm.set_memory_limit(vm.allocated_bytes()).unwrap();
        assert!(matches!(vm.managed_payload(Adt { variant: 0, fields: vec![Value::null(); 1024] }), Err(LanaError::Oom)));
        assert!(vm.owned_cycles.is_empty());
    }

    #[test]
    fn sweep_lock_is_nonblocking_and_retries_after_unlock() {
        let heap = Heap::new(1024 * 1024);
        let value = Arc::new(Mutex::new(Array::from_items(&heap, vec![Value::number(1.0)]).unwrap()));
        let node = Node::Array(value.clone());
        let guard = value.lock().unwrap();
        assert_eq!(clear_node_chunk(&node, 1), None);
        assert_eq!(guard.items.len(), 1);
        drop(guard);
        assert_eq!(clear_node_chunk(&node, 1), Some(true));
    }

    #[test]
    fn retained_guard_owner_survives_deferred_vm_teardown() {
        let chunk = assembler::assemble("HALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let value = vm.array_value(vec![Value::number(7.0)]).unwrap();
        let ValueKind::Array(array) = &value.kind else { unreachable!() };
        let slot = array.lock().unwrap().cycle.clone();
        let locked_slot = slot.node.lock().unwrap();
        assert!(!vm.collect_slice(true, 1).unwrap());
        assert!(vm.host_roots.last_slice_work() <= 1);
        drop(locked_slot);
        vm.result = value.clone();
        let root = vm.result().unwrap();
        let heap = vm.heap();
        let guard = array.lock().unwrap();
        drop(vm);
        assert_eq!(guard.items[0].as_number(), 7.0);
        drop(guard);
        assert_eq!(root.print(), "[7]");
        drop(value);
        drop(slot);
        drop(root);
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn routine_collection_uses_bounded_slices_and_restarts_after_mutation() {
        let chunk = assembler::assemble(".function main 0 1\nHALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let array = vm.array_value(vec![Value::null(); 20_000]).unwrap();
        let ValueKind::Array(items) = &array.kind else { unreachable!() };
        for index in 0..20_000 { items.lock().unwrap().items[index] = vm.array_value(vec![]).unwrap(); }
        vm.result = array;
        let root = vm.result().unwrap();
        vm.collect_generation(true).unwrap();
        let mut slices = 0;
        let mut slice_times = Vec::new();
        let started = std::time::Instant::now();
        assert!(!vm.collect_safepoint_slice(true).unwrap());
        slice_times.push(started.elapsed());
        assert!(vm.host_roots.last_slice_work() <= 128);
        slices += 1;

        let child = vm.array_value(vec![]).unwrap();
        let ValueKind::Array(items) = &root.value.kind else { unreachable!() };
        items.lock().unwrap().push(child.clone()).unwrap();
        let ValueKind::Array(child_items) = &child.kind else { unreachable!() };
        let weak_child = Arc::downgrade(child_items);
        drop(child);

        loop {
            let started = std::time::Instant::now();
            let complete = vm.collect_safepoint_slice(true).unwrap();
            slice_times.push(started.elapsed());
            assert!(vm.host_roots.last_slice_work() <= 128);
            if complete { break; }
            slices += 1;
            assert!(slices < 6_000, "collector did not finish its sliced pass: {:?}", vm.host_roots.workspace.lock().unwrap().progress.as_ref().map(|p|
                (p.phase, p.trace_cursor, p.cycle_cursor, p.classify_cursor, p.root_seed_cursor, p.mark_current, p.mark_edge, p.sweep_cursor)));
        }
        assert!(slices > 1_000, "20,000 graph nodes and edges should span bounded work slices");
        if !cfg!(debug_assertions) {
            slice_times.sort_unstable();
            assert!(slice_times[slice_times.len() * 99 / 100] <= std::time::Duration::from_millis(10),
                "wide graph safepoint p99 exceeded 10 ms: {:?}", slice_times[slice_times.len() * 99 / 100]);
        }
        assert!(weak_child.upgrade().is_some(), "rooted mutation target must survive");
        drop(root);
        drop(vm);
        assert!(weak_child.upgrade().is_none());

        let mut vm = Vm::new(&chunk);
        let mut chain = Value::null();
        for _ in 0..20_000 { chain = vm.array_value(vec![chain]).unwrap(); }
        vm.result = chain;
        let root = vm.result().unwrap();
        vm.collect_generation(true).unwrap();
        let mut times = Vec::new();
        loop {
            let started = std::time::Instant::now();
            let complete = vm.collect_safepoint_slice(true).unwrap();
            times.push(started.elapsed());
            assert!(vm.host_roots.last_slice_work() <= 128);
            if complete { break; }
            assert!(times.len() < 6_000, "deep collector pass did not finish");
        }
        if !cfg!(debug_assertions) {
            times.sort_unstable();
            assert!(times[times.len() * 99 / 100] <= std::time::Duration::from_millis(10),
                "deep graph safepoint p99 exceeded 10 ms: {:?}", times[times.len() * 99 / 100]);
        }
        drop(root);
        drop(vm);
    }

    #[test]
    fn container_collection_discovers_information_and_receipt_cycles_without_classes() {
        let chunk = assembler::assemble(".function main 0 1\nHALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let root = vm.information_root(&Value::number(1.0)).unwrap();
        let reactive = Arc::downgrade(root.reactive.as_ref().unwrap());
        root.reactive.as_ref().unwrap().lock().unwrap().history.push(ReactiveVersion {
            revision: 0, value: Some(root.clone()),
        });
        let plan = vm.planned_effect("test", &Value::number(2.0)).unwrap();
        let effect = Arc::downgrade(plan.planned_effect.as_ref().unwrap());
        plan.planned_effect.as_ref().unwrap().state.lock().unwrap().receipts.push(EffectReceipt {
            revision: 0, result: plan.clone(),
        });
        vm.collect_classes().unwrap();
        assert_eq!(root.reactive.as_ref().unwrap().lock().unwrap().history.len(), 1);
        assert_eq!(plan.planned_effect.as_ref().unwrap().state.lock().unwrap().receipts.len(), 1);
        drop(root);
        drop(plan);
        vm.collect_classes().unwrap();
        assert!(reactive.upgrade().is_none());
        assert!(effect.upgrade().is_none());
        assert!(vm.owned_cycles.is_empty());
    }

    #[test]
    fn container_collection_traces_a_twenty_thousand_node_cycle_with_linear_work() {
        let chunk = assembler::assemble(".function main 0 1\nHALT\n").unwrap();
        let mut vm = Vm::new(&chunk);
        let first = vm.array_value(vec![Value::null()]).unwrap();
        let ValueKind::Array(first_array) = &first.kind else { unreachable!() };
        let weak = Arc::downgrade(first_array);
        let mut tail = first.clone();
        for _ in 1..20_000 {
            let next = vm.array_value(vec![Value::null()]).unwrap();
            let ValueKind::Array(array) = &tail.kind else { unreachable!() };
            array.lock().unwrap().items[0] = next.clone();
            tail = next;
        }
        let ValueKind::Array(array) = &tail.kind else { unreachable!() };
        array.lock().unwrap().items[0] = first.clone();
        drop(first);
        drop(tail);
        vm.set_instruction_limit(500_000);
        vm.collect_classes().unwrap();
        assert!(weak.upgrade().is_none());
        let heap = vm.heap.clone();
        drop(vm);
        assert_eq!(heap.live_bytes(), 0);
    }
}
