//! Non-moving cycle collection for value graphs. Leaf buffers retain ordinary Arc ownership.
use std::cell::{Cell, RefCell, UnsafeCell};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};

use lana_bytecode::LanaError;
use crate::heap::{Heap, Reservation};

// ponytail: serialize graph access during collection; partition ownership domains if
// measured contention warrants it. A process-wide domain also covers shared task graphs.
static WORLD: Mutex<()> = Mutex::new(());
// The arena owns one reference per slot. Reclaim all unreachable payloads before
// releasing slots, so dropping a deep acyclic graph never recurses through nodes.
static NODES: Mutex<Vec<Arc<dyn Erased>>> = Mutex::new(Vec::new());
thread_local! {
    static ACCESS: RefCell<std::rc::Weak<MutexGuard<'static, ()>>> = RefCell::new(std::rc::Weak::new());
    static DEFER: Cell<usize> = const { Cell::new(0) };
    static PENDING: Cell<bool> = const { Cell::new(false) };
}

struct Access(Option<Rc<MutexGuard<'static, ()>>>);
fn access() -> Access {
    ACCESS.with(|current| {
        if let Some(held) = current.borrow().upgrade() { return Access(Some(held)); }
        let held = Rc::new(WORLD.lock().unwrap_or_else(|error| error.into_inner()));
        *current.borrow_mut() = Rc::downgrade(&held);
        Access(Some(held))
    })
}
impl Drop for Access {
    fn drop(&mut self) {
        drop(self.0.take());
        maybe_collect();
    }
}

/// Defer automatic collection across a VM operation, not graph synchronization.
pub struct Scope;
impl Scope {
    pub fn new() -> Self { DEFER.with(|depth| depth.set(depth.get() + 1)); Self }
}
impl Drop for Scope {
    fn drop(&mut self) {
        DEFER.with(|depth| depth.set(depth.get() - 1));
        maybe_collect();
    }
}
fn maybe_collect() {
    if PENDING.get() && DEFER.get() == 0 { collect(); }
}

/// Every directly owned managed reference must be visited exactly once. Visitors
/// borrow edges; they must not clone handles, allocate, or change the graph.
///
/// # Safety
/// Duplicating an edge can misclassify a live reference as internal. Implementations
/// are crate-private and must enumerate storage, not recursively traverse objects.
pub(crate) unsafe trait Trace: Send + Sync + 'static {
    fn trace(&self, visit: &mut dyn FnMut(usize));
}
trait Erased: Send + Sync {
    fn id(&self) -> usize;
    fn trace(&self, visit: &mut dyn FnMut(usize));
    unsafe fn reclaim(&self);
}
struct Slot<T: Trace> {
    value: UnsafeCell<Option<T>>,
    _charge: Reservation,
}
// Slot mutation occurs only while WORLD excludes all handle changes and graph
// mutations, and only after proving there is no external reference to the slot.
unsafe impl<T: Trace> Send for Slot<T> {}
unsafe impl<T: Trace> Sync for Slot<T> {}
impl<T: Trace> Erased for Slot<T> {
    fn id(&self) -> usize { self as *const Self as usize }
    fn trace(&self, visit: &mut dyn FnMut(usize)) {
        unsafe { if let Some(value) = &*self.value.get() { value.trace(visit); } }
    }
    unsafe fn reclaim(&self) { drop((*self.value.get()).take()); }
}

/// A strong, cycle-aware reference. Its backing Arc and weak references cannot
/// escape: root acquisition must participate in collector synchronization.
#[allow(private_bounds)] // Only the VM's audited graph types implement Trace.
pub struct Gc<T: Trace>(Option<Arc<Slot<T>>>);
#[allow(private_bounds)]
impl<T: Trace> Gc<T> {
    pub fn new(heap: &Heap, value: T) -> Result<Self, LanaError> {
        let charge = heap.reserve(Self::allocation_bytes())?;
        let _access = access();
        let slot = Arc::new(Slot { value: UnsafeCell::new(Some(value)), _charge: charge });
        let erased: Arc<dyn Erased> = slot.clone();
        let mut nodes = NODES.lock().unwrap();
        nodes.try_reserve(1).map_err(|_| LanaError::Oom)?;
        nodes.push(erased);
        Ok(Self(Some(slot)))
    }
    pub(crate) fn allocation_bytes() -> usize {
        // Include Arc counters, the registry's minimum four-entry allocation,
        // and one collection row/worklist entry per node.
        std::mem::size_of::<Slot<T>>() + 2 * std::mem::size_of::<usize>()
            + 4 * std::mem::size_of::<Arc<dyn Erased>>()
            + std::mem::size_of::<Row>() + std::mem::size_of::<usize>()
    }
    fn slot(&self) -> &Arc<Slot<T>> { self.0.as_ref().expect("live managed reference") }
    pub fn ptr_eq(left: &Self, right: &Self) -> bool { Arc::ptr_eq(left.slot(), right.slot()) }
    pub fn as_ptr(value: &Self) -> *const T { &**value as *const T }
    pub fn downgrade(value: &Self) -> WeakGc<T> { WeakGc(Arc::downgrade(value.slot())) }
    pub(crate) fn edge(&self, visit: &mut dyn FnMut(usize)) { visit(Arc::as_ptr(self.slot()) as usize); }
}

#[allow(private_bounds)]
pub struct WeakGc<T: Trace>(Weak<Slot<T>>);
#[allow(private_bounds)]
impl<T: Trace> WeakGc<T> {
    pub fn upgrade(&self) -> Option<Gc<T>> {
        let _access = access();
        self.0.upgrade().map(|node| Gc(Some(node)))
    }
}
impl<T: Trace> Clone for Gc<T> {
    fn clone(&self) -> Self {
        let _access = access();
        Self(Some(self.slot().clone()))
    }
}
impl<T: Trace> Deref for Gc<T> {
    type Target = T;
    fn deref(&self) -> &T {
        // This handle is either an external root or borrowed through a rooted
        // owner. Collection cannot reclaim its slot for the lifetime of the borrow.
        unsafe { (&*self.slot().value.get()).as_ref().expect("rooted managed value") }
    }
}
impl<T: Trace + fmt::Debug> fmt::Debug for Gc<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result { (**self).fmt(formatter) }
}
impl<T: Trace> Drop for Gc<T> {
    fn drop(&mut self) {
        let _access = access();
        if let Some(slot) = self.0.take() {
            PENDING.set(true);
            drop(slot);
        }
    }
}

/// Interior graph mutation holds the same domain lock as cloning and collection.
pub struct GraphCell<T>(Mutex<T>);
impl<T> GraphCell<T> {
    pub fn new(value: T) -> Self { Self(Mutex::new(value)) }
    pub fn lock(&self) -> Result<GraphGuard<'_, T>, LanaError> {
        let access = access();
        let guard = self.0.lock().map_err(|_| LanaError::Corruption)?;
        Ok(GraphGuard { guard, _access: access, cell: self })
    }
    pub fn try_lock(&self) -> Result<GraphGuard<'_, T>, LanaError> {
        let access = access();
        let guard = self.0.try_lock().map_err(|_| LanaError::InvalidState)?;
        Ok(GraphGuard { guard, _access: access, cell: self })
    }
}
impl<T: fmt::Debug> fmt::Debug for GraphCell<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result { self.0.fmt(formatter) }
}
pub struct GraphGuard<'a, T> {
    // Drop the object lock before Access can start a collection.
    guard: MutexGuard<'a, T>,
    _access: Access,
    cell: &'a GraphCell<T>,
}
impl<T> Deref for GraphGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T { &self.guard }
}
impl<T> DerefMut for GraphGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T { &mut self.guard }
}
unsafe impl<T: Trace> Trace for GraphCell<T> {
    fn trace(&self, visit: &mut dyn FnMut(usize)) {
        self.0.lock().unwrap_or_else(|error| error.into_inner()).trace(visit);
    }
}

/// Waiting must release the collection domain as well as the object mutex.
#[derive(Debug, Default)]
pub struct GraphCondvar(Condvar);
impl GraphCondvar {
    pub fn new() -> Self { Self(Condvar::new()) }
    pub fn notify_all(&self) { self.0.notify_all(); }
    pub fn wait<'a, T>(&self, held: GraphGuard<'a, T>) -> Result<GraphGuard<'a, T>, LanaError> {
        self.wait_inner(held, None).map(|(guard, _)| guard)
    }
    pub fn wait_timeout<'a, T>(&self, held: GraphGuard<'a, T>, timeout: std::time::Duration)
        -> Result<(GraphGuard<'a, T>, std::sync::WaitTimeoutResult), LanaError> {
        self.wait_inner(held, Some(timeout)).map(|(guard, timeout)| (guard, timeout.unwrap()))
    }
    fn wait_inner<'a, T>(&self, held: GraphGuard<'a, T>, timeout: Option<std::time::Duration>)
        -> Result<(GraphGuard<'a, T>, Option<std::sync::WaitTimeoutResult>), LanaError> {
        let _scope = Scope::new();
        let GraphGuard { guard, _access: access, cell } = held;
        // Holding another graph mutex across a wait would block its producer.
        assert_eq!(Rc::strong_count(access.0.as_ref().unwrap()), 1, "nested graph wait");
        drop(access);
        let (guard, timed_out) = match timeout {
            Some(timeout) => self.0.wait_timeout(guard, timeout)
                .map(|(guard, result)| (guard, Some(result))).map_err(|_| LanaError::Corruption)?,
            None => (self.0.wait(guard).map_err(|_| LanaError::Corruption)?, None),
        };
        // Restore domain-before-object lock order. Callers recheck the predicate.
        drop(guard);
        Ok((cell.lock()?, timed_out))
    }
}

struct Row {
    node: Arc<dyn Erased>,
    incoming: usize,
    marked: bool,
}

/// Reclaim unreachable cycles at a safe boundary. No partial reclamation occurs
/// if scratch allocation fails. Calling while a graph guard is held defers work.
pub fn collect() -> bool {
    if ACCESS.with(|held| held.borrow().upgrade().is_some()) { return false; }
    PENDING.set(false);
    let _scope = Scope::new();
    let _access = access();
    let mut rows = Vec::new();
    {
        let nodes = NODES.lock().unwrap();
        if rows.try_reserve_exact(nodes.len()).is_err() { return false; }
        for node in nodes.iter() {
            rows.push(Row { node: node.clone(), incoming: 0, marked: false });
        }
    }
    let mut work = Vec::new();
    if work.try_reserve_exact(rows.len()).is_err() { return false; }
    rows.sort_unstable_by_key(|row| row.node.id());
    // ponytail: sorted IDs avoid a second hash table; O(E log N), replace only
    // if graph-traversal measurements justify a separately budgeted index.
    for index in 0..rows.len() {
        let node = rows[index].node.clone();
        node.trace(&mut |id| {
            if let Ok(target) = rows.binary_search_by_key(&id, |row| row.node.id()) {
                rows[target].incoming += 1;
            }
        });
    }
    for (index, row) in rows.iter_mut().enumerate() {
        // Exclude this snapshot's reference and the arena's reference.
        if Arc::strong_count(&row.node) > row.incoming + 2 {
            row.marked = true;
            work.push(index);
        }
    }
    while let Some(index) = work.pop() {
        let node = rows[index].node.clone();
        node.trace(&mut |id| {
            if let Ok(target) = rows.binary_search_by_key(&id, |row| row.node.id()) {
                if !rows[target].marked {
                    rows[target].marked = true;
                    work.push(target);
                }
            }
        });
    }
    for row in &rows {
        if !row.marked {
            // All roots and reachable objects are marked, all pointer changes
            // are excluded, and rows retain every slot until edge destruction ends.
            unsafe { row.node.reclaim(); }
        }
    }
    {
        let mut nodes = NODES.lock().unwrap();
        nodes.retain(|node| {
            rows.binary_search_by_key(&node.id(), |row| row.node.id()).is_ok_and(|index| rows[index].marked)
        });
        // Do not retain a historical peak after its owning heap charges drop.
        nodes.shrink_to_fit();
    }
    drop(rows);
    PENDING.set(false);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Debug)]
    struct Links(Vec<Gc<GraphCell<Links>>>);
    unsafe impl Trace for Links {
        fn trace(&self, visit: &mut dyn FnMut(usize)) {
            for edge in &self.0 { edge.edge(visit); }
        }
    }

    #[test]
    fn cycles_release_and_aliases_remain_roots() {
        let heap = Heap::default();
        let a = Gc::new(&heap, GraphCell::new(Links(Vec::new()))).unwrap();
        let b = Gc::new(&heap, GraphCell::new(Links(vec![a.clone()]))).unwrap();
        a.lock().unwrap().0.push(b.clone());
        let alias = b.clone();
        drop(a);
        drop(b);
        collect();
        assert_eq!(alias.lock().unwrap().0.len(), 1);
        assert!(heap.live_bytes() > 0);
        drop(alias);
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn deferred_cycles_and_allocation_failure() {
        let heap = Heap::default();
        let scope = Scope::new();
        let a = Gc::new(&heap, GraphCell::new(Links(Vec::new()))).unwrap();
        a.lock().unwrap().0.push(a.clone());
        drop(a);
        assert!(heap.live_bytes() > 0);
        drop(scope);
        assert_eq!(heap.live_bytes(), 0);
        let heap = Heap::new(0);
        assert!(matches!(Gc::new(&heap, GraphCell::new(Links(Vec::new()))), Err(LanaError::Oom)));
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn deep_graph_drop_is_iterative() {
        let heap = Heap::default();
        let scope = Scope::new();
        let mut root = Gc::new(&heap, GraphCell::new(Links(Vec::new()))).unwrap();
        for _ in 0..10000 {
            root = Gc::new(&heap, GraphCell::new(Links(vec![root]))).unwrap();
        }
        drop(root);
        drop(scope);
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn condition_wait_releases_graph_access() {
        let state = GraphCell::new(false);
        let ready = GraphCondvar::new();
        let (started, receiving) = std::sync::mpsc::channel();
        std::thread::scope(|threads| {
            let worker = threads.spawn(|| {
                let mut guard = state.lock().unwrap();
                started.send(()).unwrap();
                while !*guard {
                    let (next, timeout) = ready.wait_timeout(guard, std::time::Duration::from_secs(5)).unwrap();
                    guard = next;
                    assert!(!timeout.timed_out(), "producer could not acquire graph access");
                }
            });
            receiving.recv().unwrap();
            *state.lock().unwrap() = true;
            ready.notify_all();
            worker.join().unwrap();
        });
    }

    #[test]
    fn concurrent_aliases_and_mutation() {
        let heap = Heap::default();
        let a = Gc::new(&heap, GraphCell::new(Links(Vec::new()))).unwrap();
        a.lock().unwrap().0.push(a.clone());
        std::thread::scope(|threads| {
            for _ in 0..4 {
                let a = a.clone();
                threads.spawn(move || {
                    for _ in 0..100 {
                        let alias = a.clone();
                        assert_eq!(alias.lock().unwrap().0.len(), 1);
                        collect();
                    }
                });
            }
        });
        drop(a);
        assert_eq!(heap.live_bytes(), 0);
    }
}
