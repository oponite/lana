//! SHA-256, mirroring `runtime/c/sha256.c` and `runtime/include/sha256.h`.
//!
//! The implementation is a direct port so the digest bytes are identical to
//! the C11 reference (the store and claims modules hash canonical payloads and
//! compare digests byte-for-byte).

/// The digest size in bytes, matching `LANA_SHA256_DIGEST_SIZE`.
pub const SHA256_DIGEST_SIZE: usize = 32;

const ROUND_CONSTANTS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

fn rotate_right(value: u32, count: u32) -> u32 {
    (value >> count) | (value << (32 - count))
}

/// A streaming SHA-256 context, mirroring `LanaSha256`.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    length: u64,
    block: [u8; 64],
    block_length: usize,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// Initialize a context, mirroring `lana_sha256_init`.
    pub fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
                0x1f83d9ab, 0x5be0cd19,
            ],
            length: 0,
            block: [0u8; 64],
            block_length: 0,
        }
    }

    /// Feed bytes, mirroring `lana_sha256_update`.
    pub fn update(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            let available = 64 - self.block_length;
            let copied = data.len().min(available);
            self.block[self.block_length..self.block_length + copied]
                .copy_from_slice(&data[..copied]);
            self.block_length += copied;
            self.length += copied as u64;
            data = &data[copied..];
            if self.block_length == 64 {
                let block = self.block;
                self.transform(&block);
                self.block_length = 0;
            }
        }
    }

    /// Finalize and write the 32-byte digest, mirroring `lana_sha256_final`.
    pub fn finalize(&mut self, out: &mut [u8; SHA256_DIGEST_SIZE]) {
        let bits = self.length * 8;
        self.block[self.block_length] = 0x80;
        self.block_length += 1;
        if self.block_length > 56 {
            for byte in &mut self.block[self.block_length..] {
                *byte = 0;
            }
            let block = self.block;
            self.transform(&block);
            self.block_length = 0;
        }
        for byte in &mut self.block[self.block_length..56] {
            *byte = 0;
        }
        for index in 0..8 {
            self.block[63 - index] = (bits >> (index * 8)) as u8;
        }
        let block = self.block;
        self.transform(&block);
        for index in 0..8 {
            out[index * 4] = (self.state[index] >> 24) as u8;
            out[index * 4 + 1] = (self.state[index] >> 16) as u8;
            out[index * 4 + 2] = (self.state[index] >> 8) as u8;
            out[index * 4 + 3] = self.state[index] as u8;
        }
    }

    fn transform(&mut self, block: &[u8; 64]) {
        let mut words = [0u32; 64];
        for index in 0..16 {
            let offset = index * 4;
            words[index] = ((block[offset] as u32) << 24)
                | ((block[offset + 1] as u32) << 16)
                | ((block[offset + 2] as u32) << 8)
                | (block[offset + 3] as u32);
        }
        for index in 16..64 {
            let s0 = rotate_right(words[index - 15], 7)
                ^ rotate_right(words[index - 15], 18)
                ^ (words[index - 15] >> 3);
            let s1 = rotate_right(words[index - 2], 17)
                ^ rotate_right(words[index - 2], 19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];
        let mut e = self.state[4];
        let mut f = self.state[5];
        let mut g = self.state[6];
        let mut h = self.state[7];
        for index in 0..64 {
            let s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            let choose = (e & f) ^ (!e & g);
            let temporary1 = h
                .wrapping_add(s1)
                .wrapping_add(choose)
                .wrapping_add(ROUND_CONSTANTS[index])
                .wrapping_add(words[index]);
            let s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary1);
            d = c;
            c = b;
            b = a;
            a = temporary1.wrapping_add(temporary2);
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

/// One-shot SHA-256, mirroring `lana_sha256`.
pub fn sha256(data: &[u8]) -> [u8; SHA256_DIGEST_SIZE] {
    let mut context = Sha256::new();
    context.update(data);
    let mut out = [0u8; SHA256_DIGEST_SIZE];
    context.finalize(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_matches_fips_vector() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let digest = sha256(b"");
        let expected = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(digest, expected);
    }

    #[test]
    fn abc_matches_fips_vector() {
        // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let digest = sha256(b"abc");
        let expected = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ];
        assert_eq!(digest, expected);
    }

    #[test]
    fn streaming_matches_one_shot() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let mut context = Sha256::new();
        for chunk in data.chunks(3) {
            context.update(chunk);
        }
        let mut out = [0u8; 32];
        context.finalize(&mut out);
        assert_eq!(out, sha256(data));
    }
}
