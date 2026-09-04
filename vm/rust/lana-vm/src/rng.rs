//! Deterministic PCG32 RNG, mirroring `lana_vm_random` and `lana_vm_seed` in
//! `vm/c/vm.c`.
//!
//! The stream must be identical to the C11 VM so that sampling and
//! measurement produce the same results under both implementations.

/// PCG32 state, mirroring the `rng_state`/`rng_increment` fields of `LanaVM`.
#[derive(Debug, Clone, Copy)]
pub struct Rng {
    pub state: u64,
    pub increment: u64,
}

impl Rng {
    /// The default increment, matching `lana_vm_seed`.
    const DEFAULT_INCREMENT: u64 = (1442695040888963407u64 << 1) | 1;

    pub fn new() -> Self {
        Self { state: 0, increment: Self::DEFAULT_INCREMENT }
    }

    /// Advance the generator and return the next 32-bit value, matching
    /// `lana_vm_random`.
    pub fn random(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state.wrapping_mul(6364136223846793005).wrapping_add(self.increment);
        let xor_shifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rotation = (old_state >> 59) as u32;
        (xor_shifted >> rotation) | (xor_shifted << ((0u32.wrapping_sub(rotation)) & 31))
    }

    /// Seed the generator, matching `lana_vm_seed`.
    pub fn seed(&mut self, seed: u64) {
        self.state = 0;
        self.increment = Self::DEFAULT_INCREMENT;
        let _ = self.random();
        self.state = self.state.wrapping_add(seed);
        let _ = self.random();
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_stream_is_deterministic() {
        // The exact stream is verified differentially against the C11 VM in
        // the conformance harness; here we only check determinism.
        let mut rng = Rng::new();
        rng.seed(0x4c414e41);
        let first = rng.random();
        let second = rng.random();
        let mut again = Rng::new();
        again.seed(0x4c414e41);
        assert_eq!(first, again.random());
        assert_eq!(second, again.random());
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = Rng::new();
        let mut b = Rng::new();
        a.seed(1);
        b.seed(2);
        assert_ne!(a.random(), b.random());
    }
}
