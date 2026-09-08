//! Checked, lifetime-owned reservations for VM allocations.
use std::mem::size_of;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, Weak};

use lana_bytecode::LanaError;

#[derive(Debug)]
struct State {
    limit: usize,
    live: usize,
    peak: usize,
    allocations: u64,
    strings: HashMap<usize, (Weak<str>, usize)>,
}

impl State {
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
        self.allocations = self.allocations.saturating_add(1);
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Heap(Arc<Mutex<State>>);

impl Default for Heap {
    fn default() -> Self { Self::new(256 * 1024 * 1024) }
}

impl Heap {
    pub fn new(limit: usize) -> Self {
        Self(Arc::new(Mutex::new(State { limit, live: 0, peak: 0, allocations: 0, strings: HashMap::new() })))
    }

    pub fn live_bytes(&self) -> usize { self.0.lock().unwrap().live }
    pub fn peak_bytes(&self) -> usize { self.0.lock().unwrap().peak }
    pub fn allocations(&self) -> u64 { self.0.lock().unwrap().allocations }

    /// Strings may be exposed as ordinary Arc<str> aliases. Keep a weak entry
    /// and its charge until every alias is gone; no Value wrapper can lose it.
    pub fn string(&self, text: &str) -> Result<Arc<str>, LanaError> {
        let bytes = text.len().checked_add(1).ok_or(LanaError::Oom)?;
        let mut reservation = self.reserve(bytes)?;
        let mut state = self.0.lock().unwrap();
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

    pub fn collect_strings(&self) { self.0.lock().unwrap().collect_strings(); }

    pub fn set_limit(&self, limit: usize) -> Result<(), LanaError> {
        let mut state = self.0.lock().unwrap();
        state.collect_strings();
        if state.live > limit { return Err(LanaError::Oom); }
        state.limit = limit;
        Ok(())
    }

    pub fn reserve(&self, bytes: usize) -> Result<Reservation, LanaError> {
        let mut reservation = Reservation { heap: self.clone(), bytes: 0 };
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
}

impl Reservation {
    fn resize(&mut self, bytes: usize) -> Result<(), LanaError> {
        let mut state = self.heap.0.lock().unwrap();
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
        self.heap.0.lock().unwrap().live -= self.bytes;
    }
}

/// A capacity-accounted buffer. Slice access cannot grow or replace its Vec.
/// The reservation includes optional owner bytes (for example, an Array).
#[derive(Debug)]
pub struct Buffer<T> {
    values: Vec<T>,
    reservation: Reservation,
    owner_bytes: usize,
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
        Ok(Self { values, reservation, owner_bytes })
    }

    pub fn new(heap: &Heap, capacity: usize, owner_bytes: usize) -> Result<Self, LanaError> {
        let bytes = Self::bytes(capacity, owner_bytes)?;
        let reservation = heap.reserve(bytes)?;
        let mut values = Vec::new();
        values.try_reserve_exact(capacity).map_err(|_| LanaError::Oom)?;
        Ok(Self { values, reservation, owner_bytes })
    }

    fn bytes(capacity: usize, owner_bytes: usize) -> Result<usize, LanaError> {
        capacity.checked_mul(size_of::<T>()).and_then(|n| n.checked_add(owner_bytes))
            .ok_or(LanaError::Oom)
    }

    pub fn capacity(&self) -> usize { self.values.capacity() }

    pub(crate) fn heap(&self) -> &Heap { &self.reservation.heap }

    pub fn reserve(&mut self, additional: usize) -> Result<(), LanaError> {
        let required = self.values.len().checked_add(additional).ok_or(LanaError::Oom)?;
        if required <= self.capacity() { return Ok(()); }
        let capacity = required.max(self.capacity().saturating_mul(2));
        let old_bytes = self.reservation.bytes;
        self.reservation.resize(Self::bytes(capacity, self.owner_bytes)?)?;
        if self.values.try_reserve_exact(capacity - self.values.len()).is_err() {
            self.reservation.resize(old_bytes)?;
            return Err(LanaError::Oom);
        }
        Ok(())
    }

    pub fn push(&mut self, value: T) -> Result<(), LanaError> {
        self.reserve(1)?;
        self.values.push(value);
        Ok(())
    }

    pub fn extend(&mut self, values: impl IntoIterator<Item = T>) -> Result<(), LanaError> {
        for value in values { self.push(value)?; }
        Ok(())
    }

    pub fn extend_from_slice(&mut self, values: &[T]) -> Result<(), LanaError> where T: Clone {
        self.reserve(values.len())?;
        self.values.extend_from_slice(values);
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> { self.values.pop() }
    pub fn clear(&mut self) { self.values.clear(); }
    pub fn truncate(&mut self, len: usize) { self.values.truncate(len); }

    pub fn resize(&mut self, len: usize, value: T) -> Result<(), LanaError> where T: Clone {
        self.reserve(len.saturating_sub(self.values.len()))?;
        self.values.resize(len, value);
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
    fn deref_mut(&mut self) -> &mut [T] { &mut self.values }
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
