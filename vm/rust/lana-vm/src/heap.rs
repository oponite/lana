//! Checked, lifetime-owned reservations for VM allocations.
use std::mem::size_of;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use lana_bytecode::LanaError;

#[derive(Clone, Debug)]
pub(crate) enum CycleWeak {
    Array(Weak<Mutex<crate::value::Array>>),
    Map(Weak<Mutex<crate::value::Map>>),
    Set(Weak<Mutex<crate::value::Set>>),
    Reactive(Weak<Mutex<crate::value::Reactive>>),
    Generator(Weak<Mutex<crate::value::Generator>>),
    Future(Weak<Mutex<crate::value::Future>>),
    Effect(Weak<crate::value::PlannedEffect>),
    Joint(Weak<crate::value::JointState>),
    Kernel(Weak<crate::value::FiniteKernel>),
    Network(Weak<crate::value::FiniteNetwork>),
    Possibility(Weak<crate::value::Possibility>),
    Paths(Weak<crate::value::PathSet>),
    Adt(Weak<crate::value::Adt>),
    Object(Weak<crate::value::ObjectValue>),
    Training(Weak<crate::value::TrainingResult>),
    Posterior(Weak<crate::value::Posterior>),
    Dataset(Weak<crate::value::Dataset>),
    Claim(Weak<crate::value::Claim>),
    StateDist(Weak<crate::value::StateDist>),
    Derivation(Weak<crate::derivation::Derivation>),
    Task(Weak<crate::value::Task>),
}

impl CycleWeak {
    pub(crate) fn payload_bytes(&self) -> Result<Option<usize>, LanaError> {
        macro_rules! locked {
            ($value:expr, $ty:ty, $field:ident) => {{
                let value = $value.upgrade().ok_or(LanaError::Task)?;
                let value = value.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
                Some(add_bytes(&[size_of::<$ty>() + 2 * size_of::<usize>(), vector_bytes(&value.$field)?])?)
            }};
        }
        Ok(match self {
            Self::Task(_) => Some(size_of::<crate::value::Task>() + size_of::<crate::value::TaskState>()
                + size_of::<std::sync::Condvar>() + size_of::<AtomicBool>() + 8 * size_of::<usize>()),
            Self::Generator(value) => locked!(value, crate::value::Generator, registers),
            Self::Future(value) => locked!(value, crate::value::Future, registers),
            Self::Reactive(value) => locked!(value, crate::value::Reactive, history),
            Self::Effect(value) => {
                let value = value.upgrade().ok_or(LanaError::Task)?;
                let state = value.state.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
                Some(add_bytes(&[size_of::<crate::value::PlannedEffect>() + 2 * size_of::<usize>(), vector_bytes(&state.receipts)?])?)
            }
            _ => None,
        })
    }

    pub(crate) fn strong_node(&self) -> Option<crate::vm::class_gc::Node> {
        use crate::vm::class_gc::Node;
        match self {
            Self::Array(value) => value.upgrade().map(Node::Array),
            Self::Map(value) => value.upgrade().map(Node::Map),
            Self::Set(value) => value.upgrade().map(Node::Set),
            Self::Reactive(value) => value.upgrade().map(Node::Reactive),
            Self::Generator(value) => value.upgrade().map(Node::Generator),
            Self::Future(value) => value.upgrade().map(Node::Future),
            Self::Effect(value) => value.upgrade().map(Node::Effect),
            Self::Joint(value) => value.upgrade().map(Node::Joint),
            Self::Kernel(value) => value.upgrade().map(Node::Kernel),
            Self::Network(value) => value.upgrade().map(Node::Network),
            Self::Possibility(value) => value.upgrade().map(Node::Possibility),
            Self::Paths(value) => value.upgrade().map(Node::Paths),
            Self::Adt(value) => value.upgrade().map(Node::Adt),
            Self::Object(value) => value.upgrade().map(Node::Object),
            Self::Training(value) => value.upgrade().map(Node::Training),
            Self::Posterior(value) => value.upgrade().map(Node::Posterior),
            Self::Dataset(value) => value.upgrade().map(Node::Dataset),
            Self::Claim(value) => value.upgrade().map(Node::Claim),
            Self::StateDist(value) => value.upgrade().map(Node::StateDist),
            Self::Derivation(value) => value.upgrade().map(Node::Derivation),
            Self::Task(value) => value.upgrade().map(Node::Task),
        }
    }

    pub(crate) fn identity(&self) -> usize {
        match self {
            Self::Array(v) => v.as_ptr() as usize,
            Self::Map(v) => v.as_ptr() as usize,
            Self::Set(v) => v.as_ptr() as usize,
            Self::Reactive(v) => v.as_ptr() as usize,
            Self::Generator(v) => v.as_ptr() as usize,
            Self::Future(v) => v.as_ptr() as usize,
            Self::Effect(v) => v.as_ptr() as usize,
            Self::Joint(v) => v.as_ptr() as usize,
            Self::Kernel(v) => v.as_ptr() as usize,
            Self::Network(v) => v.as_ptr() as usize,
            Self::Possibility(v) => v.as_ptr() as usize,
            Self::Paths(v) => v.as_ptr() as usize,
            Self::Adt(v) => v.as_ptr() as usize,
            Self::Object(v) => v.as_ptr() as usize,
            Self::Training(v) => v.as_ptr() as usize,
            Self::Posterior(v) => v.as_ptr() as usize,
            Self::Dataset(v) => v.as_ptr() as usize,
            Self::Claim(v) => v.as_ptr() as usize,
            Self::StateDist(v) => v.as_ptr() as usize,
            Self::Derivation(v) => v.as_ptr() as usize,
            Self::Task(v) => v.as_ptr() as usize,
        }
    }

    pub(crate) fn alive(&self) -> bool {
        match self {
            Self::Array(v) => v.strong_count() != 0,
            Self::Map(v) => v.strong_count() != 0,
            Self::Set(v) => v.strong_count() != 0,
            Self::Reactive(v) => v.strong_count() != 0,
            Self::Generator(v) => v.strong_count() != 0,
            Self::Future(v) => v.strong_count() != 0,
            Self::Effect(v) => v.strong_count() != 0,
            Self::Joint(v) => v.strong_count() != 0,
            Self::Kernel(v) => v.strong_count() != 0,
            Self::Network(v) => v.strong_count() != 0,
            Self::Possibility(v) => v.strong_count() != 0,
            Self::Paths(v) => v.strong_count() != 0,
            Self::Adt(v) => v.strong_count() != 0,
            Self::Object(v) => v.strong_count() != 0,
            Self::Training(v) => v.strong_count() != 0,
            Self::Posterior(v) => v.strong_count() != 0,
            Self::Dataset(v) => v.strong_count() != 0,
            Self::Claim(v) => v.strong_count() != 0,
            Self::StateDist(v) => v.strong_count() != 0,
            Self::Derivation(v) => v.strong_count() != 0,
            Self::Task(v) => v.strong_count() != 0,
        }
    }
}

pub(crate) trait ManagedPayload: Sized {
    fn weak(value: Weak<Self>) -> CycleWeak;
    fn heap_bytes(&self) -> Result<usize, LanaError>;
}

macro_rules! managed_payloads {
    ($($kind:ident($ty:ty) => $bytes:expr),+ $(,)?) => { $(
        impl ManagedPayload for $ty {
            fn weak(value: Weak<Self>) -> CycleWeak { CycleWeak::$kind(value) }
            fn heap_bytes(&self) -> Result<usize, LanaError> {
                let extra: Result<usize, LanaError> = ($bytes)(self);
                (size_of::<Self>() + 2 * size_of::<usize>()).checked_add(extra?).ok_or(LanaError::Oom)
            }
        }
    )+ };
}

fn vector_bytes<T>(value: &Vec<T>) -> Result<usize, LanaError> {
    value.capacity().checked_mul(size_of::<T>()).ok_or(LanaError::Oom)
}

fn add_bytes(parts: &[usize]) -> Result<usize, LanaError> {
    parts.iter().try_fold(0usize, |total, part| total.checked_add(*part).ok_or(LanaError::Oom))
}

managed_payloads! {
    Joint(crate::value::JointState) => |value: &crate::value::JointState| {
        value.rows.iter().try_fold(add_bytes(&[vector_bytes(&value.names)?, vector_bytes(&value.domains)?,
            vector_bytes(&value.values)?, vector_bytes(&value.rows)?])?,
            |total, row| total.checked_add(vector_bytes(&row.values)?).ok_or(LanaError::Oom))
    },
    Kernel(crate::value::FiniteKernel) => |value: &crate::value::FiniteKernel| {
        let bytes = add_bytes(&[vector_bytes(&value.input_domains)?, vector_bytes(&value.output_domain)?, vector_bytes(&value.rows)?])?;
        let bytes = value.input_domains.iter().try_fold(bytes, |total, row|
            total.checked_add(vector_bytes(row)?).ok_or(LanaError::Oom))?;
        value.rows.iter().try_fold(bytes, |total, row|
            total.checked_add(vector_bytes(row)?).ok_or(LanaError::Oom))
    },
    Network(crate::value::FiniteNetwork) => |value: &crate::value::FiniteNetwork| {
        value.nodes.iter().try_fold(add_bytes(&[vector_bytes(&value.nodes)?, vector_bytes(&value.order)?])?,
            |total, node| total.checked_add(vector_bytes(&node.parents)?).ok_or(LanaError::Oom))
    },
    Possibility(crate::value::Possibility) => |value: &crate::value::Possibility| {
        add_bytes(&[vector_bytes(&value.values)?, value.weights.as_ref().map(vector_bytes).transpose()?.unwrap_or(0)])
    },
    Paths(crate::value::PathSet) => |value: &crate::value::PathSet| vector_bytes(&value.alternatives),
    Adt(crate::value::Adt) => |value: &crate::value::Adt| vector_bytes(&value.fields),
    Object(crate::value::ObjectValue) => |value: &crate::value::ObjectValue| vector_bytes(&value.fields),
    Training(crate::value::TrainingResult) => |_: &crate::value::TrainingResult| Ok(0),
    Posterior(crate::value::Posterior) => |_: &crate::value::Posterior| Ok(0),
    Dataset(crate::value::Dataset) => |_: &crate::value::Dataset| Ok(0),
    Claim(crate::value::Claim) => |_: &crate::value::Claim| Ok(0),
    StateDist(crate::value::StateDist) => |_: &crate::value::StateDist| Ok(0),
    Derivation(crate::derivation::Derivation) => |value: &crate::derivation::Derivation| {
        add_bytes(&[vector_bytes(&value.inputs)?, size_of::<Mutex<Option<crate::value::Tensor>>>() + 2 * size_of::<usize>()])
    },
}

#[derive(Debug)]
pub(crate) struct CycleSlot {
    pub node: Mutex<Option<CycleWeak>>,
    generation: std::sync::atomic::AtomicU8,
    remembered: std::sync::atomic::AtomicBool,
    tracing: std::sync::atomic::AtomicBool,
    _allocation: Mutex<Reservation>,
    pub(crate) payload_allocation: Mutex<Option<Reservation>>,
    owner: Mutex<Option<crate::vm::class_gc::Node>>,
    owns_node: AtomicBool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Generation {
    Young = 0,
    Old = 1,
    StableShared = 2,
}

impl CycleSlot {
    pub(crate) fn bind(&self, node: CycleWeak) {
        let managed = self._allocation.lock().unwrap().heap().0.collector.lock().unwrap().strong_count() != 0;
        let owner = if managed { node.strong_node() } else { None };
        self.owns_node.store(owner.is_some(), Ordering::Release);
        *self.owner.lock().unwrap() = owner;
        *self.node.lock().unwrap() = Some(node);
    }

    pub(crate) fn owns_node(&self) -> bool { self.owns_node.load(Ordering::Acquire) }

    pub(crate) fn release_owner(&self) -> bool {
        let Ok(mut owner) = self.owner.try_lock() else { return false; };
        let value = owner.take();
        self.owns_node.store(false, Ordering::Release);
        drop(owner);
        drop(value);
        true
    }

    pub(crate) fn generation(&self) -> Generation {
        match self.generation.load(Ordering::Acquire) {
            1 => Generation::Old,
            2 => Generation::StableShared,
            _ => Generation::Young,
        }
    }

    pub(crate) fn promote(&self, generation: Generation) {
        self.generation.fetch_max(generation as u8, Ordering::AcqRel);
    }

    pub(crate) fn remembered(&self) -> bool {
        self.remembered.load(Ordering::Acquire)
    }

    pub(crate) fn set_remembered(&self, value: bool) {
        self.remembered.store(value, Ordering::Release);
    }

    pub(crate) fn set_tracing(&self, value: bool) { self.tracing.store(value, Ordering::Release); }

    pub(crate) fn reserve_edges(&self, edges: usize, bytes: usize) -> Result<(), LanaError> {
        let mut allocation = self._allocation.lock().unwrap();
        let old_edges = allocation.gc_edges;
        allocation.admit_gc(1, edges.max(old_edges))?;
        let size = allocation.bytes.checked_add(bytes).ok_or(LanaError::Oom);
        if let Err(error) = size.and_then(|size| allocation.resize(size)) {
            allocation.admit_gc(1, old_edges)?;
            return Err(error);
        }
        Ok(())
    }


    pub(crate) fn mutated(&self) {
        if self.tracing.load(Ordering::Acquire) || self.generation() != Generation::Young {
            self._allocation.lock().unwrap().heap().bump_mutation_epoch();
        }
        if self.generation() == Generation::Old {
            self.remembered.store(true, Ordering::Release);
        }
    }
}

impl Drop for CycleSlot {
    fn drop(&mut self) {
        let allocation = self._allocation.get_mut().unwrap();
        let mut state = allocation.heap.0.lock();
        state.live_cycles -= 1;
        if state.live_cycles == 0 && (state.cycles.is_empty()
            || !allocation.heap.0.routine_collection.load(Ordering::Acquire)) {
            state.live -= state.cycles.capacity() * size_of::<Weak<CycleSlot>>();
            state.cycles = Vec::new();
            state.cycle_bytes_at_collection = 0;
        }
    }
}

#[derive(Debug)]
struct State {
    limit: usize,
    live: usize,
    peak: usize,
    allocations: u64,
    strings: HashMap<usize, (Weak<str>, usize)>,
    cycles: Vec<Weak<CycleSlot>>,
    cycles_since_collection: usize,
    live_cycles: usize,
    cycle_bytes_at_collection: usize,
    cycle_bytes_at_major: usize,
    collection_deferred: bool,
}
impl State {
    fn prune_cycles(&mut self) {
        self.cycles.retain(|slot| slot.strong_count() != 0);
        if self.cycles.is_empty() {
            self.live -= self.cycles.capacity() * size_of::<Weak<CycleSlot>>();
            self.cycles = Vec::new();
        }
    }

    fn collect_strings(&mut self) {
        let live = &mut self.live;
        self.strings.retain(|_, (string, bytes)| {
            if string.strong_count() != 0 { return true; }
            *live -= *bytes;
            false
        });
    }

    fn charge(&mut self, bytes: usize) -> Result<(), LanaError> {
        if bytes > self.limit.saturating_sub(self.live) { self.collect_strings(); }
        if bytes > self.limit.saturating_sub(self.live) { return Err(LanaError::Oom); }
        self.live += bytes;
        self.peak = self.peak.max(self.live);
        if bytes != 0 { self.allocations = self.allocations.saturating_add(1); }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Heap(Arc<HeapState>);

#[derive(Debug)]
struct HeapState {
    state: Mutex<State>,
    live: AtomicUsize,
    mutation_epoch: AtomicU64,
    routine_collection: AtomicBool,
    collector: Mutex<Weak<crate::vm::class_gc::RootOwner>>,
}

// Publish accounting before releasing the allocator lock. Safepoints only
// observe usage; allocation admission still checks State under this lock.
struct HeapGuard<'a> {
    state: MutexGuard<'a, State>,
    published_live: &'a AtomicUsize,
}

impl HeapState {
    fn lock(&self) -> HeapGuard<'_> {
        HeapGuard { state: self.state.lock().unwrap(), published_live: &self.live }
    }
}

impl Deref for HeapGuard<'_> {
    type Target = State;
    fn deref(&self) -> &State { &self.state }
}

impl DerefMut for HeapGuard<'_> {
    fn deref_mut(&mut self) -> &mut State { &mut self.state }
}

impl Drop for HeapGuard<'_> {
    fn drop(&mut self) { self.published_live.store(self.state.live, Ordering::Release); }
}

impl Default for Heap {
    fn default() -> Self { Self::new(256 * 1024 * 1024) }
}

pub(crate) struct RoutineCollection<'a>(&'a Heap);
impl Drop for RoutineCollection<'_> {
    fn drop(&mut self) { self.0.0.routine_collection.store(false, Ordering::Release); }
}

impl Heap {
    pub(crate) fn routine_collection(&self) -> RoutineCollection<'_> {
        self.0.routine_collection.store(true, Ordering::Release);
        RoutineCollection(self)
    }
    pub fn new(limit: usize) -> Self {
        Self(Arc::new(HeapState { state: Mutex::new(State { limit, live: 0, peak: 0, allocations: 0, strings: HashMap::new(), cycles: Vec::new(), cycles_since_collection: 0, live_cycles: 0, cycle_bytes_at_collection: 0, cycle_bytes_at_major: 0, collection_deferred: false }), live: AtomicUsize::new(0), mutation_epoch: AtomicU64::new(0), routine_collection: AtomicBool::new(false), collector: Mutex::new(Weak::new()) }))
    }

    pub(crate) fn set_collector(&self, owner: &Arc<crate::vm::class_gc::RootOwner>) {
        *self.0.collector.lock().unwrap() = Arc::downgrade(owner);
    }

    pub(crate) fn clear_collector(&self, owner: *const crate::vm::class_gc::RootOwner) {
        let mut collector = self.0.collector.lock().unwrap();
        if collector.as_ptr() == owner { *collector = Weak::new(); }
    }

    fn admit_gc(&self, nodes: usize, edges: usize) -> Result<(), LanaError> {
        let owner = self.0.collector.lock().unwrap().upgrade();
        if let Some(owner) = owner { owner.admit(nodes, edges)?; }
        Ok(())
    }

    fn release_gc(&self, nodes: usize, edges: usize) {
        let owner = self.0.collector.lock().unwrap().upgrade();
        if let Some(owner) = owner { owner.release_admission(nodes, edges); }
    }

    pub(crate) fn cycle_slot(&self) -> Result<Arc<CycleSlot>, LanaError> {
        self.cycle_slot_with_edges(0)
    }

    pub(crate) fn cycle_slot_with_edges(&self, edges: usize) -> Result<Arc<CycleSlot>, LanaError> {
        let mut allocation = self.reserve(size_of::<CycleSlot>() + 2 * size_of::<usize>())?;
        allocation.admit_gc(1, edges)?;
        let mut state = self.0.lock();
        if state.cycles.len() == state.cycles.capacity() {
            state.prune_cycles();
            if state.cycles.len() == state.cycles.capacity() {
                let capacity = state.cycles.capacity().saturating_mul(2).max(16);
                let additional = capacity - state.cycles.len();
                let bytes = additional.checked_mul(size_of::<Weak<CycleSlot>>()).ok_or(LanaError::Oom)?;
                state.charge(bytes)?;
                if state.cycles.try_reserve_exact(additional).is_err() {
                    state.live -= bytes;
                    return Err(LanaError::Oom);
                }
            }
        }
        let slot = Arc::new(CycleSlot { node: Mutex::new(None),
            generation: std::sync::atomic::AtomicU8::new(Generation::Young as u8),
            remembered: std::sync::atomic::AtomicBool::new(false),
            tracing: std::sync::atomic::AtomicBool::new(false), _allocation: Mutex::new(allocation),
            payload_allocation: Mutex::new(None), owner: Mutex::new(None), owns_node: AtomicBool::new(false) });
        state.live_cycles += 1;
        state.cycles.push(Arc::downgrade(&slot));
        state.cycles_since_collection = state.cycles_since_collection.saturating_add(1);
        Ok(slot)
    }

    pub(crate) fn cycle_count(&self) -> usize { self.0.lock().cycles.len() }

    pub(crate) fn mutate_cycle(&self, identity: usize) {
        self.bump_mutation_epoch();
        // ponytail: suspended frames use the existing weak registry; add a
        // direct slot link if measured mutation lookup cost becomes material.
        for index in 0..self.cycle_count() {
            let Some(slot) = self.cycle_slot_at(index) else { continue; };
            let matches = slot.node.lock().unwrap().as_ref().is_some_and(|node| node.identity() == identity);
            if matches { slot.mutated(); break; }
        }
    }

    pub(crate) fn cycle_node(&self, index: usize) -> Option<CycleWeak> {
        let slot = self.0.lock().cycles.get(index)?.upgrade()?;
        let node = slot.node.lock().unwrap().clone();
        node
    }

    pub(crate) fn cycle_slot_at(&self, index: usize) -> Option<Arc<CycleSlot>> {
        self.0.lock().cycles.get(index)?.upgrade()
    }

    pub(crate) fn cycles_due(&self, pressure: bool) -> bool {
        if !pressure { return false; }
        let mut state = self.0.lock();
        state.cycle_bytes_at_collection = state.cycle_bytes_at_collection.min(state.live);
        // A full trace should buy headroom, not repeat for every allocation
        // while a large live graph keeps the heap above the pressure threshold.
        pressure && (state.cycles_since_collection > 0 || state.collection_deferred)
            && state.live >= state.cycle_bytes_at_collection.saturating_add(state.limit / 8)
    }

    pub(crate) fn major_collection_due(&self, pressure: bool) -> bool {
        if !pressure { return false; }
        let state = self.0.lock();
        state.live >= state.cycle_bytes_at_major.saturating_add(state.limit / 2)
    }

    // One weak-registry visit per routine slice work unit.
    pub(crate) fn prune_cycle_at(&self, index: usize) -> bool {
        let Some(weak) = self.0.lock().cycles.get(index).cloned() else { return false; };
        let slot = weak.upgrade();
        let dead = match &slot {
            None => true,
            Some(slot) => match slot.node.try_lock() {
                Ok(node) => !node.as_ref().is_some_and(CycleWeak::alive),
                Err(_) => false,
            },
        };
        let mut state = self.0.lock();
        let remove = dead && state.cycles.get(index).is_some_and(|current| current.ptr_eq(&weak));
        if remove { state.cycles.swap_remove(index); }
        // A concurrently retired slot may now have its last strong owner here.
        // Its destructor takes the heap lock, so release that lock first.
        drop(state);
        remove
    }

    pub(crate) fn finish_cycle_slice(&self, major: bool) {
        let mut state = self.0.lock();
        if state.cycles.is_empty() {
            state.live -= state.cycles.capacity() * size_of::<Weak<CycleSlot>>();
            state.cycles = Vec::new();
        }
        state.cycles_since_collection = 0;
        state.cycle_bytes_at_collection = state.live;
        state.collection_deferred = false;
        if major { state.cycle_bytes_at_major = state.live; }
    }

    pub(crate) fn finish_cycles(&self, major: bool) {
        let mut state = self.0.lock();
        state.prune_cycles();
        state.cycles_since_collection = 0;
        state.cycle_bytes_at_collection = state.live;
        state.collection_deferred = false;
        if major { state.cycle_bytes_at_major = state.live; }
    }

    pub(crate) fn defer_cycles(&self) {
        let mut state = self.0.lock();
        state.cycles_since_collection = 0;
        state.cycle_bytes_at_collection = state.live;
        state.collection_deferred = true;
    }

    pub fn live_bytes(&self) -> usize { self.0.live.load(Ordering::Acquire) }
    pub(crate) fn same_heap(&self, other: &Self) -> bool { Arc::ptr_eq(&self.0, &other.0) }
    pub(crate) fn mutation_epoch(&self) -> u64 { self.0.mutation_epoch.load(Ordering::Acquire) }
    pub(crate) fn bump_mutation_epoch(&self) { self.0.mutation_epoch.fetch_add(1, Ordering::AcqRel); }
    pub fn peak_bytes(&self) -> usize { self.0.lock().peak }
    pub fn allocations(&self) -> u64 { self.0.lock().allocations }

    /// Strings may be exposed as ordinary Arc<str> aliases. Keep a weak entry
    /// and its charge until every alias is gone; no Value wrapper can lose it.
    pub fn string(&self, text: &str) -> Result<Arc<str>, LanaError> {
        let bytes = text.len().checked_add(1).ok_or(LanaError::Oom)?;
        let mut reservation = self.reserve(bytes)?;
        let mut state = self.0.lock();
        if state.allocations % 256 == 0 { state.collect_strings(); }
        state.strings.try_reserve(1).map_err(|_| LanaError::Oom)?;
        let string: Arc<str> = Arc::from(text);
        state.strings.insert(Arc::as_ptr(&string) as *const () as usize, (Arc::downgrade(&string), bytes));
        reservation.bytes = 0; // The weak entry now owns this charge.
        Ok(string)
    }

    pub fn lossy_string(&self, mut bytes: &[u8]) -> Result<Arc<str>, LanaError> {
        if let Ok(text) = std::str::from_utf8(bytes) { return self.string(text); }
        let mut decoded = Buffer::new(self, 0, 0)?;
        loop {
            match std::str::from_utf8(bytes) {
                Ok(text) => { decoded.extend_from_slice(text.as_bytes())?; break; }
                Err(error) => {
                    decoded.extend_from_slice(&bytes[..error.valid_up_to()])?;
                    decoded.extend_from_slice("\u{fffd}".as_bytes())?;
                    match error.error_len() {
                        Some(length) => bytes = &bytes[error.valid_up_to() + length..],
                        None => break,
                    }
                }
            }
        }
        self.string(std::str::from_utf8(&decoded).unwrap())
    }

    pub fn collect_strings(&self) { self.0.lock().collect_strings(); }

    pub fn set_limit(&self, limit: usize) -> Result<(), LanaError> {
        let mut state = self.0.lock();
        state.collect_strings();
        if state.live > limit { return Err(LanaError::Oom); }
        state.limit = limit;
        Ok(())
    }

    pub fn reserve(&self, bytes: usize) -> Result<Reservation, LanaError> {
        let mut reservation = Reservation { heap: self.clone(), bytes: 0, gc_nodes: 0, gc_edges: 0 };
        reservation.resize(bytes)?;
        Ok(reservation)
    }
}

/// Not Clone: exactly one owner releases each allocation. Shared buffers put
/// the buffer and its reservation together behind a single Arc.
#[derive(Debug)]
pub struct Reservation {
    heap: Heap,
    bytes: usize,
    gc_nodes: usize,
    gc_edges: usize,
}

impl Reservation {
    pub(crate) fn heap(&self) -> &Heap { &self.heap }

    pub(crate) fn admit_gc(&mut self, nodes: usize, edges: usize) -> Result<(), LanaError> {
        let added_nodes = nodes.saturating_sub(self.gc_nodes);
        let added_edges = edges.saturating_sub(self.gc_edges);
        if added_nodes != 0 || added_edges != 0 { self.heap.admit_gc(added_nodes, added_edges)?; }
        self.heap.release_gc(self.gc_nodes.saturating_sub(nodes), self.gc_edges.saturating_sub(edges));
        self.gc_nodes = nodes;
        self.gc_edges = edges;
        Ok(())
    }

    pub(crate) fn resize(&mut self, bytes: usize) -> Result<(), LanaError> {
        let mut state = self.heap.0.lock();
        if bytes > self.bytes {
            let growth = bytes - self.bytes;
            state.charge(growth)?;
        } else {
            state.live -= self.bytes - bytes;
        }
        self.bytes = bytes;
        Ok(())
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.heap.0.lock().live -= self.bytes;
        if self.gc_nodes != 0 || self.gc_edges != 0 { self.heap.release_gc(self.gc_nodes, self.gc_edges); }
    }
}

/// A capacity-accounted buffer. Slice access cannot grow or replace its Vec.
/// The reservation includes optional owner bytes (for example, an Array).
#[derive(Debug)]
pub struct Buffer<T> {
    values: Vec<T>,
    reservation: Reservation,
    owner_bytes: usize,
    barrier: Option<Arc<CycleSlot>>,
}

impl<T> Buffer<T> {
    pub fn filled(heap: &Heap, len: usize, value: T) -> Result<Self, LanaError> where T: Clone {
        let mut buffer = Self::new(heap, len, 0)?;
        buffer.values.resize(len, value);
        Ok(buffer)
    }

    pub fn from_slice(heap: &Heap, values: &[T]) -> Result<Self, LanaError> where T: Clone {
        let mut buffer = Self::new(heap, values.len(), 0)?;
        buffer.values.extend_from_slice(values);
        Ok(buffer)
    }

    /// Admit an already-owned host buffer to this heap without copying it.
    pub fn from_vec(heap: &Heap, values: Vec<T>, owner_bytes: usize) -> Result<Self, LanaError> {
        let reservation = heap.reserve(Self::bytes(values.capacity(), owner_bytes)?)?;
        Ok(Self { values, reservation, owner_bytes, barrier: None })
    }

    pub fn new(heap: &Heap, capacity: usize, owner_bytes: usize) -> Result<Self, LanaError> {
        let bytes = Self::bytes(capacity, owner_bytes)?;
        let reservation = heap.reserve(bytes)?;
        let mut values = Vec::new();
        values.try_reserve_exact(capacity).map_err(|_| LanaError::Oom)?;
        Ok(Self { values, reservation, owner_bytes, barrier: None })
    }

    fn bytes(capacity: usize, owner_bytes: usize) -> Result<usize, LanaError> {
        capacity.checked_mul(size_of::<T>()).and_then(|n| n.checked_add(owner_bytes))
            .ok_or(LanaError::Oom)
    }

    pub fn capacity(&self) -> usize { self.values.capacity() }

    pub(crate) fn set_barrier(&mut self, barrier: &Arc<CycleSlot>) -> Result<(), LanaError> {
        self.reservation.admit_gc(0, self.capacity().checked_mul(5).ok_or(LanaError::Oom)?)?;
        self.barrier = Some(barrier.clone());
        Ok(())
    }

    pub(crate) fn retain(&mut self, keep: impl FnMut(&T) -> bool) {
        let before = self.values.len();
        self.values.retain(keep);
        if self.values.len() != before { if let Some(barrier) = &self.barrier { barrier.mutated(); } }
    }

    pub(crate) fn heap(&self) -> &Heap { &self.reservation.heap }

    pub fn reserve(&mut self, additional: usize) -> Result<(), LanaError> {
        let required = self.values.len().checked_add(additional).ok_or(LanaError::Oom)?;
        if required <= self.capacity() { return Ok(()); }
        let capacity = required.max(self.capacity().saturating_mul(2));
        match self.reserve_capacity(capacity) {
            Err(LanaError::Oom) if capacity > required => self.reserve_capacity(required),
            result => result,
        }
    }

    fn reserve_capacity(&mut self, capacity: usize) -> Result<(), LanaError> {
        let old_bytes = self.reservation.bytes;
        let old_edges = self.reservation.gc_edges;
        if self.barrier.is_some() { self.reservation.admit_gc(0, capacity.checked_mul(5).ok_or(LanaError::Oom)?)?; }
        if let Err(error) = self.reservation.resize(Self::bytes(capacity, self.owner_bytes)?) {
            self.reservation.admit_gc(0, old_edges)?;
            return Err(error);
        }
        if self.values.try_reserve_exact(capacity - self.values.len()).is_err() {
            self.reservation.resize(old_bytes)?;
            self.reservation.admit_gc(0, old_edges)?;
            return Err(LanaError::Oom);
        }
        Ok(())
    }

    pub fn push(&mut self, value: T) -> Result<(), LanaError> {
        self.reserve(1)?;
        self.values.push(value);
        if let Some(barrier) = &self.barrier { barrier.mutated(); }
        Ok(())
    }

    pub fn extend(&mut self, values: impl IntoIterator<Item = T>) -> Result<(), LanaError> {
        for value in values { self.push(value)?; }
        Ok(())
    }

    pub fn extend_from_slice(&mut self, values: &[T]) -> Result<(), LanaError> where T: Clone {
        self.reserve(values.len())?;
        self.values.extend_from_slice(values);
        if !values.is_empty() { if let Some(barrier) = &self.barrier { barrier.mutated(); } }
        Ok(())
    }

    pub(crate) fn swap_remove(&mut self, index: usize) -> T {
        let value = self.values.swap_remove(index);
        if let Some(barrier) = &self.barrier { barrier.mutated(); }
        value
    }

    pub fn pop(&mut self) -> Option<T> {
        let value = self.values.pop();
        if value.is_some() { if let Some(barrier) = &self.barrier { barrier.mutated(); } }
        value
    }
    pub fn clear(&mut self) {
        if !self.values.is_empty() { if let Some(barrier) = &self.barrier { barrier.mutated(); } }
        self.values.clear();
    }
    pub fn truncate(&mut self, len: usize) {
        if len < self.values.len() { if let Some(barrier) = &self.barrier { barrier.mutated(); } }
        self.values.truncate(len);
    }

    pub fn resize(&mut self, len: usize, value: T) -> Result<(), LanaError> where T: Clone {
        let changed = len != self.values.len();
        self.reserve(len.saturating_sub(self.values.len()))?;
        self.values.resize(len, value);
        if changed { if let Some(barrier) = &self.barrier { barrier.mutated(); } }
        Ok(())
    }
}

impl<T> Deref for Buffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] { &self.values }
}

impl<T: PartialEq> PartialEq for Buffer<T> {
    fn eq(&self, other: &Self) -> bool { self.values == other.values }
}

impl<T> DerefMut for Buffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        if let Some(barrier) = &self.barrier { barrier.mutated(); }
        &mut self.values
    }
}

impl<'a, T> IntoIterator for &'a Buffer<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter { self.values.iter() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safepoint_usage_does_not_wait_for_the_allocator_lock() {
        let heap = Heap::new(64);
        let reservation = heap.reserve(32).unwrap();
        std::thread::scope(|scope| {
            let guard = heap.0.lock();
            let (send, receive) = std::sync::mpsc::channel();
            let heap = &heap;
            scope.spawn(move || send.send(heap.live_bytes()).unwrap());
            let observed = receive.recv_timeout(std::time::Duration::from_secs(1));
            drop(guard);
            assert_eq!(observed.unwrap(), 32);
        });
        drop(reservation);
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn string_aliases_keep_the_charge_and_pressure_reclaims_dead_strings() {
        let heap = Heap::new(4);
        let string = heap.string("abc").unwrap();
        let alias = string.clone();
        let weak = Arc::downgrade(&alias);
        drop(string);
        heap.collect_strings();
        assert_eq!(heap.live_bytes(), 4);
        assert_eq!(heap.string("x"), Err(LanaError::Oom));
        drop(alias);
        assert!(weak.upgrade().is_none());
        for _ in 0..1000 {
            assert_eq!(&*heap.string("abc").unwrap(), "abc");
        }
        heap.collect_strings();
        assert_eq!(heap.live_bytes(), 0);
        assert_eq!(heap.peak_bytes(), 4);
    }

    #[test]
    fn lossy_decoding_matches_the_standard_library_and_is_bounded() {
        let heap = Heap::new(4096);
        for first in 0..=255u8 {
            for second in [0, 0x7f, 0x80, 0xbf, 0xe2, 0xff] {
                let bytes = [first, second];
                assert_eq!(&*heap.lossy_string(&bytes).unwrap(), &*String::from_utf8_lossy(&bytes));
            }
        }
        heap.collect_strings();
        assert_eq!(heap.live_bytes(), 0);
        let heap = Heap::new(2);
        assert_eq!(heap.lossy_string(&[0xff]), Err(LanaError::Oom));
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn reservation_rolls_back_failure_and_releases_on_last_owner() {
        let heap = Heap::new(64);
        let reservation = Arc::new(heap.reserve(64).unwrap());
        let alias = reservation.clone();
        assert_eq!(heap.reserve(1).unwrap_err(), LanaError::Oom);
        assert_eq!(heap.reserve(usize::MAX).unwrap_err(), LanaError::Oom);
        drop(reservation);
        assert_eq!(heap.live_bytes(), 64);
        drop(alias);
        assert_eq!(heap.live_bytes(), 0);
        assert_eq!(heap.peak_bytes(), 64);
    }

    #[test]
    fn growth_retries_exact_capacity_when_doubling_exceeds_limit() {
        let heap = Heap::new(3 * size_of::<u64>());
        let mut buffer = Buffer::<u64>::new(&heap, 2, 0).unwrap();
        buffer.push(7).unwrap();
        buffer.push(9).unwrap();
        buffer.push(11).unwrap();
        assert_eq!(&*buffer, &[7, 9, 11]);
        assert_eq!(heap.live_bytes(), 3 * size_of::<u64>());
        assert_eq!(buffer.push(13).unwrap_err(), LanaError::Oom);
        assert_eq!(&*buffer, &[7, 9, 11]);
        drop(buffer);
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn growth_is_checked_before_mutation_and_capacity_stays_charged() {
        let heap = Heap::new(16);
        let mut buffer = Buffer::<u64>::new(&heap, 1, 8).unwrap();
        buffer.push(7).unwrap();
        assert_eq!(buffer.push(9).unwrap_err(), LanaError::Oom);
        assert_eq!(&*buffer, &[7]);
        buffer.clear();
        assert_eq!(heap.live_bytes(), 16);
        drop(buffer);
        assert_eq!(heap.live_bytes(), 0);
        assert_eq!(Buffer::<u64>::new(&heap, usize::MAX, 0).unwrap_err(), LanaError::Oom);
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn weak_pruning_defers_locked_slot_metadata() {
        let heap = Heap::new(16 * 1024);
        let slot = heap.cycle_slot().unwrap();
        let guard = slot.node.lock().unwrap();
        let other_heap = heap.clone();
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || sender.send(other_heap.prune_cycle_at(0)).unwrap());
        let result = receiver.recv_timeout(std::time::Duration::from_secs(1));
        drop(guard);
        worker.join().unwrap();
        assert_eq!(result.unwrap(), false);
        assert!(heap.prune_cycle_at(0));
        drop(slot);
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn reservations_cannot_overcommit_across_threads() {
        let heap = Heap::new(64);
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let successes = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let heap = &heap;
                let barrier = &barrier;
                let successes = &successes;
                scope.spawn(move || {
                    let reservation = heap.reserve(64);
                    if reservation.is_ok() { successes.fetch_add(1, std::sync::atomic::Ordering::SeqCst); }
                    barrier.wait();
                    drop(reservation);
                });
            }
        });
        assert_eq!(successes.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(heap.live_bytes(), 0);
    }
}
