//! Information-aware memory runtime (proof of concept).
//!
//! An application-level store for *information objects*: ordered fields of
//! Lana [`State`] values. It exposes five operations:
//!
//! * [`InfoMemory::allocate`] — validate and store an object uncompacted.
//! * [`InfoMemory::compact`] — re-encode it in the smallest *exact*
//!   representation the runtime can find for it.
//! * [`InfoMemory::recall`] — reconstruct the original object bit-for-bit.
//! * [`InfoMemory::release`] — drop a handle and reclaim memory once no
//!   other object depends on it.
//! * [`InfoMemory::profile`] — report the runtime's own accounting and
//!   per-operation timing.
//!
//! Compaction is lossless and uses information the runtime has about the
//! *whole store*, not just one object:
//!
//! * **Shared** — an object whose content digest (Lana's SHA-256) and content
//!   match an existing object becomes a reference to that object.
//! * **Runs** — consecutive identical states are stored once with a count.
//! * **Delta** — an object of the same length as a resident raw object that
//!   differs in few positions is stored as a sparse patch over that base.
//! * **Raw** — kept when nothing above is smaller (high-entropy data).
//!
//! Equality is bitwise on the three `f64` fields, so reconstruction is exact.
//! This module makes no claim that any representation is smaller in
//! general; it picks whichever candidate is smallest for the data it sees.

use std::collections::{HashMap, VecDeque};
use std::mem::size_of;
use std::time::Instant;

use lana_bytecode::LanaError;
use lana_vm::state::{state_valid, State};

use crate::sha256::{Sha256, SHA256_DIGEST_SIZE};

/// Opaque handle returned by [`InfoMemory::allocate`].
pub type Handle = u32;
type BlockId = u32;
type Digest = [u8; SHA256_DIGEST_SIZE];

/// How many resident raw objects of the same length are tried as delta bases.
const BASE_CANDIDATES: usize = 8;
/// Evenly spaced positions probed before a full delta scan against a base.
const BASE_PROBES: usize = 32;
const STATE_BYTES: usize = size_of::<State>();
const INDEX_BYTES: usize = size_of::<u32>();
const PATCH_BYTES: usize = STATE_BYTES + INDEX_BYTES;

#[inline]
fn same(a: &State, b: &State) -> bool {
    a.p.to_bits() == b.p.to_bits()
        && a.d_re.to_bits() == b.d_re.to_bits()
        && a.d_im.to_bits() == b.d_im.to_bits()
}

/// SHA-256 over the big-endian field bits, the same field encoding as the
/// `LST1` state codec without its per-record header.
fn digest(states: &[State]) -> Digest {
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 24];
    for chunk in states.chunks(64) {
        let mut used = 0;
        for state in chunk {
            buffer[used..used + 8].copy_from_slice(&state.p.to_bits().to_be_bytes());
            buffer[used + 8..used + 16].copy_from_slice(&state.d_re.to_bits().to_be_bytes());
            buffer[used + 16..used + 24].copy_from_slice(&state.d_im.to_bits().to_be_bytes());
            used += 24;
        }
        hasher.update(&buffer[..used]);
    }
    let mut out = [0u8; SHA256_DIGEST_SIZE];
    hasher.finalize(&mut out);
    out
}

enum Repr {
    Raw(Box<[State]>),
    Runs {
        lengths: Box<[u32]>,
        states: Box<[State]>,
    },
    Delta {
        base: BlockId,
        indexes: Box<[u32]>,
        states: Box<[State]>,
    },
}

impl Repr {
    fn payload_bytes(&self) -> usize {
        match self {
            Repr::Raw(states) => states.len() * STATE_BYTES,
            Repr::Runs { lengths, states }
            | Repr::Delta {
                indexes: lengths,
                states,
                ..
            } => lengths.len() * INDEX_BYTES + states.len() * STATE_BYTES,
        }
    }
}

struct Block {
    repr: Repr,
    len: usize,
    digest: Option<Digest>,
    refs: u32,
    compacted: bool,
}

/// Which representation an object ended up in after [`InfoMemory::compact`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compaction {
    Raw,
    Runs,
    Delta,
    Shared,
}

/// Cumulative count and wall time for one operation.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpStats {
    pub count: u64,
    pub nanos: u128,
}

impl OpStats {
    fn record(&mut self, start: Instant) {
        self.count += 1;
        self.nanos += start.elapsed().as_nanos();
    }
}

/// Runtime accounting returned by [`InfoMemory::profile`].
///
/// `payload_bytes` and `metadata_bytes` are the runtime's own estimate from
/// allocation capacities; benchmarks should check them against a measured
/// allocator total rather than trust them.
#[derive(Clone, Debug, Default)]
pub struct MemoryProfile {
    pub live_objects: usize,
    pub live_blocks: usize,
    pub raw_blocks: usize,
    pub runs_blocks: usize,
    pub delta_blocks: usize,
    pub shared_handles: usize,
    /// Bytes the live objects occupy as plain `[State]`.
    pub logical_bytes: usize,
    pub payload_bytes: usize,
    pub metadata_bytes: usize,
    pub allocate: OpStats,
    pub compact: OpStats,
    pub recall: OpStats,
    pub release: OpStats,
}

/// The information-aware memory runtime.
#[derive(Default)]
pub struct InfoMemory {
    handles: Vec<Option<BlockId>>,
    free_handles: Vec<Handle>,
    blocks: Vec<Option<Block>>,
    free_blocks: Vec<BlockId>,
    by_digest: HashMap<Digest, BlockId>,
    bases: HashMap<usize, VecDeque<BlockId>>,
    allocate_stats: OpStats,
    compact_stats: OpStats,
    recall_stats: OpStats,
    release_stats: OpStats,
}

impl InfoMemory {
    pub fn new() -> Self {
        Self::default()
    }

    fn block(&self, id: BlockId) -> &Block {
        self.blocks[id as usize].as_ref().expect("live block")
    }

    fn block_of(&self, handle: Handle) -> Result<BlockId, LanaError> {
        self.handles
            .get(handle as usize)
            .copied()
            .flatten()
            .ok_or(LanaError::NotFound)
    }

    fn insert_block(&mut self, block: Block) -> BlockId {
        match self.free_blocks.pop() {
            Some(id) => {
                self.blocks[id as usize] = Some(block);
                id
            }
            None => {
                self.blocks.push(Some(block));
                (self.blocks.len() - 1) as BlockId
            }
        }
    }

    /// ALLOCATE: store an information object. Every state must be valid.
    pub fn allocate(&mut self, states: Vec<State>) -> Result<Handle, LanaError> {
        let start = Instant::now();
        if states.len() > u32::MAX as usize {
            return Err(LanaError::Limit);
        }
        if !states.iter().all(state_valid) {
            return Err(LanaError::InvalidState);
        }
        let len = states.len();
        let block = self.insert_block(Block {
            repr: Repr::Raw(states.into_boxed_slice()),
            len,
            digest: None,
            refs: 1,
            compacted: false,
        });
        let handle = match self.free_handles.pop() {
            Some(handle) => {
                self.handles[handle as usize] = Some(block);
                handle
            }
            None => {
                self.handles.push(Some(block));
                (self.handles.len() - 1) as Handle
            }
        };
        self.allocate_stats.record(start);
        Ok(handle)
    }

    /// Whether block `id` holds exactly `data`, without decoding it.
    fn equals(&self, id: BlockId, data: &[State]) -> bool {
        let block = self.block(id);
        if block.len != data.len() {
            return false;
        }
        match &block.repr {
            Repr::Raw(states) => states.iter().zip(data).all(|(a, b)| same(a, b)),
            Repr::Runs { lengths, states } => {
                let mut at = 0;
                for (&n, state) in lengths.iter().zip(states.iter()) {
                    let n = n as usize;
                    if !data[at..at + n].iter().all(|x| same(x, state)) {
                        return false;
                    }
                    at += n;
                }
                true
            }
            Repr::Delta {
                base,
                indexes,
                states,
            } => {
                let Repr::Raw(base) = &self.block(*base).repr else {
                    unreachable!("delta bases are raw");
                };
                let mut patch = 0;
                for (i, x) in data.iter().enumerate() {
                    let expected = if patch < indexes.len() && indexes[patch] as usize == i {
                        patch += 1;
                        &states[patch - 1]
                    } else {
                        &base[i]
                    };
                    if !same(x, expected) {
                        return false;
                    }
                }
                true
            }
        }
    }

    /// Number of differing positions between `data` and raw block `base`, or
    /// `None` once that would cost at least `limit` bytes as a patch.
    fn delta_count(&self, base: BlockId, data: &[State], limit: usize) -> Option<usize> {
        let Repr::Raw(base) = &self.block(base).repr else {
            return None;
        };
        let n = data.len();
        let probes = BASE_PROBES.min(n);
        let differing = (0..probes)
            .map(|k| k * n / probes)
            .filter(|&i| !same(&data[i], &base[i]))
            .count();
        // Skip the full scan when the sampled patch cost already reaches the
        // best representation found so far (same cost model as below).
        if probes > 0 && differing * n * PATCH_BYTES >= limit * probes {
            return None;
        }
        let mut count = 0;
        for (a, b) in data.iter().zip(base.iter()) {
            if !same(a, b) {
                count += 1;
                if count * PATCH_BYTES >= limit {
                    return None;
                }
            }
        }
        Some(count)
    }

    /// COMPACT: re-encode the object in its smallest exact representation.
    pub fn compact(&mut self, handle: Handle) -> Result<Compaction, LanaError> {
        let start = Instant::now();
        let id = self.block_of(handle)?;
        let result = self.compact_block(handle, id);
        self.compact_stats.record(start);
        Ok(result)
    }

    fn compact_block(&mut self, handle: Handle, id: BlockId) -> Compaction {
        let block = self.blocks[id as usize].as_mut().expect("live block");
        if block.compacted {
            return match block.repr {
                Repr::Raw(_) if block.refs > 1 => Compaction::Shared,
                Repr::Raw(_) => Compaction::Raw,
                Repr::Runs { .. } => Compaction::Runs,
                Repr::Delta { .. } => Compaction::Delta,
            };
        }
        block.compacted = true;
        let Repr::Raw(data) = std::mem::replace(&mut block.repr, Repr::Raw(Box::new([]))) else {
            unreachable!("uncompacted blocks are raw");
        };
        let len = data.len();
        let key = digest(&data);

        // Shared: identical content already resident.
        if let Some(&other) = self.by_digest.get(&key) {
            if other != id && self.equals(other, &data) {
                self.blocks[id as usize] = None;
                self.free_blocks.push(id);
                self.blocks[other as usize]
                    .as_mut()
                    .expect("live block")
                    .refs += 1;
                self.handles[handle as usize] = Some(other);
                return Compaction::Shared;
            }
        }

        // Candidate costs in payload bytes.
        let raw_cost = len * STATE_BYTES;
        let runs = 1 + data.windows(2).filter(|w| !same(&w[0], &w[1])).count();
        let runs_cost = if len == 0 { 0 } else { runs * PATCH_BYTES };
        let mut best_cost = raw_cost.min(runs_cost);
        let mut best_base = None;
        if let Some(pool) = self.bases.get(&len) {
            for &base in pool {
                if let Some(count) = self.delta_count(base, &data, best_cost) {
                    let cost = count * PATCH_BYTES;
                    if cost < best_cost {
                        best_cost = cost;
                        best_base = Some(base);
                    }
                }
            }
        }

        let (repr, kind) = if let Some(base) = best_base {
            let Repr::Raw(base_states) = &self.block(base).repr else {
                unreachable!("delta bases are raw");
            };
            let mut indexes = Vec::new();
            let mut states = Vec::new();
            for (i, (a, b)) in data.iter().zip(base_states.iter()).enumerate() {
                if !same(a, b) {
                    indexes.push(i as u32);
                    states.push(*a);
                }
            }
            self.blocks[base as usize]
                .as_mut()
                .expect("live block")
                .refs += 1;
            let repr = Repr::Delta {
                base,
                indexes: indexes.into_boxed_slice(),
                states: states.into_boxed_slice(),
            };
            (repr, Compaction::Delta)
        } else if runs_cost < raw_cost {
            let mut lengths = Vec::with_capacity(runs);
            let mut states: Vec<State> = Vec::with_capacity(runs);
            for state in data.iter() {
                match states.last() {
                    Some(last) if same(last, state) => *lengths.last_mut().unwrap() += 1,
                    _ => {
                        states.push(*state);
                        lengths.push(1u32);
                    }
                }
            }
            let repr = Repr::Runs {
                lengths: lengths.into_boxed_slice(),
                states: states.into_boxed_slice(),
            };
            (repr, Compaction::Runs)
        } else {
            let pool = self.bases.entry(len).or_default();
            if pool.len() == BASE_CANDIDATES {
                pool.pop_front();
            }
            pool.push_back(id);
            (Repr::Raw(data), Compaction::Raw)
        };
        let block = self.blocks[id as usize].as_mut().expect("live block");
        block.repr = repr;
        block.digest = Some(key);
        self.by_digest.entry(key).or_insert(id);
        kind
    }

    /// RECALL: reconstruct the original object exactly.
    pub fn recall(&mut self, handle: Handle) -> Result<Vec<State>, LanaError> {
        let start = Instant::now();
        let id = self.block_of(handle)?;
        let block = self.block(id);
        let mut out = Vec::with_capacity(block.len);
        match &block.repr {
            Repr::Raw(states) => out.extend_from_slice(states),
            Repr::Runs { lengths, states } => {
                for (&n, state) in lengths.iter().zip(states.iter()) {
                    out.extend(std::iter::repeat_n(*state, n as usize));
                }
            }
            Repr::Delta {
                base,
                indexes,
                states,
            } => {
                let Repr::Raw(base) = &self.block(*base).repr else {
                    unreachable!("delta bases are raw");
                };
                out.extend_from_slice(base);
                for (&i, state) in indexes.iter().zip(states.iter()) {
                    out[i as usize] = *state;
                }
            }
        }
        self.recall_stats.record(start);
        Ok(out)
    }

    /// RELEASE: drop a handle; storage is reclaimed when nothing refers to it.
    pub fn release(&mut self, handle: Handle) -> Result<(), LanaError> {
        let start = Instant::now();
        let id = self.block_of(handle)?;
        self.handles[handle as usize] = None;
        self.free_handles.push(handle);
        self.unref(id);
        self.release_stats.record(start);
        Ok(())
    }

    fn unref(&mut self, id: BlockId) {
        let block = self.blocks[id as usize].as_mut().expect("live block");
        block.refs -= 1;
        if block.refs > 0 {
            return;
        }
        let block = self.blocks[id as usize].take().expect("live block");
        self.free_blocks.push(id);
        if let Some(key) = block.digest {
            if self.by_digest.get(&key) == Some(&id) {
                self.by_digest.remove(&key);
            }
        }
        if let Some(pool) = self.bases.get_mut(&block.len) {
            pool.retain(|&base| base != id);
            if pool.is_empty() {
                self.bases.remove(&block.len);
            }
        }
        if let Repr::Delta { base, .. } = block.repr {
            self.unref(base);
        }
    }

    /// Release excess capacity in the runtime's own tables.
    pub fn shrink(&mut self) {
        self.handles.shrink_to_fit();
        self.free_handles.shrink_to_fit();
        self.blocks.shrink_to_fit();
        self.free_blocks.shrink_to_fit();
        self.by_digest.shrink_to_fit();
        self.bases.shrink_to_fit();
    }

    /// PROFILE: report memory accounting and operation timings.
    pub fn profile(&self) -> MemoryProfile {
        let mut profile = MemoryProfile {
            allocate: self.allocate_stats,
            compact: self.compact_stats,
            recall: self.recall_stats,
            release: self.release_stats,
            ..MemoryProfile::default()
        };
        for block in self.handles.iter().flatten() {
            profile.live_objects += 1;
            let block = self.block(*block);
            profile.logical_bytes += block.len * STATE_BYTES;
        }
        for block in self.blocks.iter().flatten() {
            profile.live_blocks += 1;
            profile.payload_bytes += block.repr.payload_bytes();
            profile.shared_handles += block.refs.saturating_sub(1) as usize;
            match block.repr {
                Repr::Raw(_) => profile.raw_blocks += 1,
                Repr::Runs { .. } => profile.runs_blocks += 1,
                Repr::Delta { .. } => profile.delta_blocks += 1,
            }
        }
        // Delta dependents hold a reference on their base, not a handle.
        profile.shared_handles -= profile.delta_blocks.min(profile.shared_handles);
        profile.metadata_bytes = self.handles.capacity() * size_of::<Option<BlockId>>()
            + self.free_handles.capacity() * size_of::<Handle>()
            + self.blocks.capacity() * size_of::<Option<Block>>()
            + self.free_blocks.capacity() * size_of::<BlockId>()
            + self.by_digest.capacity() * (size_of::<Digest>() + size_of::<BlockId>() + 1)
            + self.bases.capacity() * (size_of::<usize>() + size_of::<VecDeque<BlockId>>() + 1)
            + self
                .bases
                .values()
                .map(|pool| pool.capacity() * size_of::<BlockId>())
                .sum::<usize>();
        profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lana_vm::state::make_complex;

    fn state(p: f64, re: f64, im: f64) -> State {
        let mut out = State::default();
        assert_eq!(make_complex(p, re, im, &mut out), LanaError::Ok);
        out
    }

    fn field(seed: u64, n: usize) -> Vec<State> {
        (0..n)
            .map(|i| {
                let x = ((seed * 7919 + i as u64 * 104729) % 9973) as f64 / 9973.0;
                state(0.01 + 0.98 * x, 0.5 * x, -0.25 * x)
            })
            .collect()
    }

    fn bits(states: &[State]) -> Vec<[u64; 3]> {
        states
            .iter()
            .map(|s| [s.p.to_bits(), s.d_re.to_bits(), s.d_im.to_bits()])
            .collect()
    }

    #[test]
    fn every_representation_recalls_exactly() {
        let mut memory = InfoMemory::new();
        let base = field(1, 512);
        let mut variant = base.clone();
        variant[17] = state(0.5, 0.1, 0.1);
        variant[400] = state(0.25, 0.0, 0.3);
        let runs = vec![state(0.3, 0.2, 0.0); 512];
        let random = field(9, 512);

        let cases = [
            (base.clone(), Compaction::Raw),
            (variant.clone(), Compaction::Delta),
            (base.clone(), Compaction::Shared),
            (runs.clone(), Compaction::Runs),
            (random.clone(), Compaction::Raw),
            (Vec::new(), Compaction::Raw),
        ];
        let mut handles = Vec::new();
        for (data, expected) in &cases {
            let handle = memory.allocate(data.clone()).unwrap();
            assert_eq!(memory.compact(handle).unwrap(), *expected);
            handles.push(handle);
        }
        for ((data, _), handle) in cases.iter().zip(&handles) {
            assert_eq!(bits(&memory.recall(*handle).unwrap()), bits(data));
        }
        let profile = memory.profile();
        assert_eq!(profile.live_objects, 6);
        assert_eq!(profile.delta_blocks, 1);
        assert_eq!(profile.shared_handles, 1);
        assert!(profile.payload_bytes < profile.logical_bytes);
    }

    #[test]
    fn release_keeps_bases_until_dependents_are_released() {
        let mut memory = InfoMemory::new();
        let base = field(3, 256);
        let mut variant = base.clone();
        variant[5] = state(0.75, 0.0, 0.0);
        let b = memory.allocate(base.clone()).unwrap();
        memory.compact(b).unwrap();
        let v = memory.allocate(variant.clone()).unwrap();
        assert_eq!(memory.compact(v).unwrap(), Compaction::Delta);
        memory.release(b).unwrap();
        assert_eq!(memory.recall(b), Err(LanaError::NotFound));
        assert_eq!(bits(&memory.recall(v).unwrap()), bits(&variant));
        memory.release(v).unwrap();
        let profile = memory.profile();
        assert_eq!(profile.live_blocks, 0);
        assert_eq!(profile.payload_bytes, 0);
        assert!(memory.by_digest.is_empty() && memory.bases.is_empty());
    }

    #[test]
    fn negative_zero_is_not_merged_with_positive_zero() {
        let mut memory = InfoMemory::new();
        let a = vec![
            State {
                p: 0.5,
                d_re: 0.0,
                d_im: 0.0
            };
            4
        ];
        let b = vec![
            State {
                p: 0.5,
                d_re: -0.0,
                d_im: 0.0
            };
            4
        ];
        let ha = memory.allocate(a.clone()).unwrap();
        let hb = memory.allocate(b.clone()).unwrap();
        memory.compact(ha).unwrap();
        assert_ne!(memory.compact(hb).unwrap(), Compaction::Shared);
        assert_eq!(bits(&memory.recall(hb).unwrap()), bits(&b));
    }

    #[test]
    fn invalid_states_and_handles_are_rejected() {
        let mut memory = InfoMemory::new();
        let invalid = State {
            p: 1.5,
            d_re: 0.0,
            d_im: 0.0,
        };
        assert_eq!(memory.allocate(vec![invalid]), Err(LanaError::InvalidState));
        assert_eq!(memory.compact(0), Err(LanaError::NotFound));
        assert_eq!(memory.release(0), Err(LanaError::NotFound));
    }
}
