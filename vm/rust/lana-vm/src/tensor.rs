//! LIP-004 tensor helpers, mirroring `vm/c/vm.c` lines 3163-3496.
//!
//! The C11 VM allocates tensor buffers through the GC (`lana_vm_alloc`); the
//! Rust VM accounts the same bytes through the caller-supplied `alloc` closure
//! (the VM's `alloc_bytes`), so the 256 MiB memory limit is preserved. Every
//! per-operation temporary — shape/stride scratch and BLAS packing buffers —
//! is accounted through the same closure in both VMs (LIP-004).

use std::sync::{Arc, Mutex};

use lana_bytecode::LanaError;

use crate::backend;
use crate::metal;
use crate::value::{Map, Tensor, TensorDtype, Value, ValueKind};

pub const TENSOR_MAX_RANK: usize = 32;

pub fn tensor_dimension(n: f64) -> Result<usize, LanaError> {
    let limit = ((usize::MAX / 2) as f64 + 1.0) * 2.0;
    if !n.is_finite() || n < 0.0 || n.floor() != n || n >= limit {
        return Err(LanaError::InvalidParameters);
    }
    Ok(n as usize)
}

/// LIP-027: convert a double to a binary16 (f16) bit pattern,
/// round-to-nearest-even. Mirrors `f64_to_f16_bits` in `vm/c/vm.c`.
pub fn f64_to_f16_bits(x: f64) -> u16 {
    if x.is_nan() {
        return 0x7E00;
    }
    if x.is_infinite() {
        return if x < 0.0 { 0xFC00 } else { 0x7C00 };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { 0x8000 } else { 0x0000 };
    }
    let bits = x.to_bits();
    let sign = (bits >> 63) as u32;
    let mut exp = ((bits >> 52) & 0x7FF) as i32 - 1023;
    let sig = (1u64 << 52) | (bits & ((1u64 << 52) - 1));

    // Round the 53-bit significand to 11 bits (1 implicit + 10 explicit).
    let drop = 42u64;
    let round_bit = 1u64 << (drop - 1);
    let mask = (1u64 << drop) - 1;
    let lsb = 1u64 << drop;
    let mut rounded = sig + round_bit;
    if (sig & mask) == round_bit && (sig & lsb) == 0 {
        rounded = sig;
    }
    let mut sig11 = rounded >> drop;
    if sig11 == (1u64 << 11) {
        sig11 = 0;
        exp += 1;
    }
    let mant10 = sig11 & 0x3FF;

    if exp > 15 {
        return ((sign << 15) | 0x7C00) as u16; // overflow to inf
    }
    if exp >= -14 {
        return ((sign << 15) | (((exp + 15) as u32) << 10) | mant10 as u32) as u16;
    }
    // Subnormal: value = sig11 * 2^exp, exp < -14.
    let shift = -14 - exp;
    if shift >= 11 {
        return (sign << 15) as u16; // rounds to zero (even)
    }
    let mut sub_mant = sig11 >> shift;
    let dropped = sig11 & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    if dropped > half || (dropped == half && (sub_mant & 1) != 0) {
        sub_mant += 1;
    }
    if sub_mant == (1u64 << 10) {
        return ((sign << 15) | (1u32 << 10)) as u16; // min normal 2^-14
    }
    ((sign << 15) | sub_mant as u32) as u16
}

/// LIP-027: convert a binary16 (f16) bit pattern to a double. Mirrors
/// `f16_to_double` in `vm/c/vm.c`.
pub fn f16_bits_to_f64(h: u16) -> f64 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let mant = h & 0x3FF;
    let value = if exp == 0 {
        (mant as f64) * 2f64.powi(-24)
    } else if exp == 31 {
        if mant != 0 { f64::NAN } else { f64::INFINITY }
    } else {
        (((1 << 10) | mant) as f64) * 2f64.powi(exp as i32 - 15 - 10)
    };
    if sign != 0 { -value } else { value }
}

/// LIP-027: round a double to binary16 (f16) precision, round-to-nearest-even,
/// stored back as a double. Mirrors `round_f16` in `vm/c/vm.c`.
pub fn round_f16(x: f64) -> f64 {
    f16_bits_to_f64(f64_to_f16_bits(x))
}

/// LIP-027: convert a double to a bfloat16 (bf16) bit pattern,
/// round-to-nearest-even. Mirrors `f64_to_bf16_bits` in `vm/c/vm.c`.
pub fn f64_to_bf16_bits(x: f64) -> u16 {
    if x.is_nan() {
        return 0x7FC0;
    }
    if x.is_infinite() {
        return if x < 0.0 { 0xFF80 } else { 0x7F80 };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { 0x8000 } else { 0x0000 };
    }
    let mut bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7FF) as i32 - 1023;
    if exp > 127 {
        return if x < 0.0 { 0xFF80 } else { 0x7F80 }; // overflow beyond fp32 range
    }
    // Round the 52-bit mantissa to 7 bits (drop 45), round-to-nearest-even.
    let low = bits & 0x1FFFFFFFFFFF;
    let half = 1u64 << 44;
    let lsb = 1u64 << 45;
    if low > half || (low == half && (bits & lsb) != 0) {
        bits += 1u64 << 45;
    }
    bits &= !0x1FFFFFFFFFFF;
    // The rounded f64 has 8 significant mantissa bits, so it converts to f32
    // exactly; bf16 is the top 16 bits of that f32 representation.
    let f = f64::from_bits(bits) as f32;
    (f.to_bits() >> 16) as u16
}

/// LIP-027: convert a bfloat16 (bf16) bit pattern to a double. bf16 is the top
/// 16 bits of an f32, so widen the f32 to f64. Mirrors `bf16_to_double` in
/// `vm/c/vm.c`.
pub fn bf16_bits_to_f64(h: u16) -> f64 {
    let f32_bits = (h as u32) << 16;
    f32::from_bits(f32_bits) as f64
}

/// LIP-027: round a double to bfloat16 (bf16) precision, round-to-nearest-even,
/// stored back as a double. Mirrors `round_bf16` in `vm/c/vm.c`.
pub fn round_bf16(x: f64) -> f64 {
    bf16_bits_to_f64(f64_to_bf16_bits(x))
}

/// LIP-027: element width in bytes for a tensor's dtype.
pub fn tensor_elem_bytes(t: &Tensor) -> usize {
    match t.dtype {
        TensorDtype::F32 => 4,
        TensorDtype::F16 | TensorDtype::Bf16 => 2,
        TensorDtype::Complex => 16,
        TensorDtype::F64 => 8,
    }
}

/// LIP-027: read the real part of element `i` (absolute element index, already
/// including `offset`), converting from the storage dtype to f64. Mirrors
/// `tensor_get_real` in `vm/c/vm.c`.
pub fn tensor_get_real(t: &Tensor, i: usize) -> f64 {
    match t.dtype {
        TensorDtype::F32 => {
            let mut b = [0u8; 4];
            b.copy_from_slice(&t.data[i * 4..i * 4 + 4]);
            f32::from_ne_bytes(b) as f64
        }
        TensorDtype::F16 => {
            let mut b = [0u8; 2];
            b.copy_from_slice(&t.data[i * 2..i * 2 + 2]);
            f16_bits_to_f64(u16::from_ne_bytes(b))
        }
        TensorDtype::Bf16 => {
            let mut b = [0u8; 2];
            b.copy_from_slice(&t.data[i * 2..i * 2 + 2]);
            bf16_bits_to_f64(u16::from_ne_bytes(b))
        }
        TensorDtype::Complex => {
            let mut b = [0u8; 8];
            b.copy_from_slice(&t.data[i * 16..i * 16 + 8]);
            f64::from_ne_bytes(b)
        }
        TensorDtype::F64 => {
            let mut b = [0u8; 8];
            b.copy_from_slice(&t.data[i * 8..i * 8 + 8]);
            f64::from_ne_bytes(b)
        }
    }
}

/// LIP-027: read the imaginary part of element `i` (0.0 for a real dtype).
/// Mirrors `tensor_get_imag` in `vm/c/vm.c`.
pub fn tensor_get_imag(t: &Tensor, i: usize) -> f64 {
    if t.dtype != TensorDtype::Complex {
        return 0.0;
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&t.data[i * 16 + 8..i * 16 + 16]);
    f64::from_ne_bytes(b)
}

/// LIP-027: write the real part of element `i`, converting from f64 to the
/// storage dtype (round-to-nearest-even for f16/bf16). Mirrors
/// `tensor_set_real` in `vm/c/vm.c`. Panics if the buffer is shared (a view);
/// callers write only to freshly-created tensors, matching the existing
/// `Arc::get_mut` usage.
pub fn tensor_set_real(t: &mut Tensor, i: usize, v: f64) {
    let data = Arc::get_mut(&mut t.data).expect("tensor_set_real on a shared buffer");
    match t.dtype {
        TensorDtype::F32 => {
            let f = v as f32;
            data[i * 4..i * 4 + 4].copy_from_slice(&f.to_ne_bytes());
        }
        TensorDtype::F16 => {
            let h = f64_to_f16_bits(v);
            data[i * 2..i * 2 + 2].copy_from_slice(&h.to_ne_bytes());
        }
        TensorDtype::Bf16 => {
            let h = f64_to_bf16_bits(v);
            data[i * 2..i * 2 + 2].copy_from_slice(&h.to_ne_bytes());
        }
        TensorDtype::Complex => {
            data[i * 16..i * 16 + 8].copy_from_slice(&v.to_ne_bytes());
        }
        TensorDtype::F64 => {
            data[i * 8..i * 8 + 8].copy_from_slice(&v.to_ne_bytes());
        }
    }
}

/// LIP-027: write the imaginary part of element `i` (no-op for a real dtype).
/// Mirrors `tensor_set_imag` in `vm/c/vm.c`.
pub fn tensor_set_imag(t: &mut Tensor, i: usize, v: f64) {
    if t.dtype != TensorDtype::Complex {
        return;
    }
    let data = Arc::get_mut(&mut t.data).expect("tensor_set_imag on a shared buffer");
    data[i * 16 + 8..i * 16 + 16].copy_from_slice(&v.to_ne_bytes());
}

/// Allocate a zero-initialized tensor with the given shape and dtype, mirroring
/// `tensor_new_dtype`. Returns `LanaError::Oom` on allocation failure or shape
/// overflow.
pub fn tensor_new_dtype(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    ndim: usize,
    shape: &[usize],
    dtype: TensorDtype,
) -> Result<Tensor, LanaError> {
    if ndim > TENSOR_MAX_RANK || ndim != shape.len() {
        return Err(LanaError::InvalidParameters);
    }
    if alloc(std::mem::size_of::<Tensor>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    let is_complex = dtype == TensorDtype::Complex;
    let mut strides = vec![0usize; ndim];
    if ndim > 0 {
        if alloc(ndim * std::mem::size_of::<usize>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        if alloc(ndim * std::mem::size_of::<usize>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        let mut stride: usize = 1;
        for i in (0..ndim).rev() {
            strides[i] = stride;
            if shape[i] != 0 && stride > usize::MAX / shape[i] {
                return Err(LanaError::Oom);
            }
            stride *= shape[i];
        }
    }
    let mut total: usize = 1;
    for i in 0..ndim {
        if shape[i] == 0 {
            total = 0;
            break;
        }
        if total > usize::MAX / shape[i] {
            return Err(LanaError::Oom);
        }
        total *= shape[i];
    }
    let elem_count = total
        .checked_mul(if is_complex { 2 } else { 1 })
        .ok_or(LanaError::Oom)?;
    // `elem_count` already counts the two components of a complex tensor, so
    // the per-component width here is 8 (one f64) for complex, giving
    // `total * 16` bytes total.
    let width = match dtype {
        TensorDtype::F32 => 4,
        TensorDtype::F16 | TensorDtype::Bf16 => 2,
        TensorDtype::Complex => 8,
        TensorDtype::F64 => 8,
    };
    let mut data = Vec::new();
    if elem_count > 0 {
        let data_bytes = elem_count
            .checked_mul(width)
            .ok_or(LanaError::Oom)?;
        if alloc(data_bytes) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        data = vec![0u8; elem_count * width];
    }
    Ok(Tensor {
        ndim,
        shape: shape.to_vec(),
        strides,
        is_complex,
        dtype,
        data: Arc::new(data),
        offset: 0,
        is_state: false,
    })
}

/// Allocate a zero-initialized tensor with the given shape, defaulting to F64
/// (or COMPLEX when `is_complex`). Kept for the common case; dtype-specific
/// callers use `tensor_new_dtype`.
pub fn tensor_new(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    ndim: usize,
    shape: &[usize],
    is_complex: bool,
) -> Result<Tensor, LanaError> {
    tensor_new_dtype(
        alloc,
        ndim,
        shape,
        if is_complex { TensorDtype::Complex } else { TensorDtype::F64 },
    )
}

/// LIP-027: cast a tensor to another real dtype. Same-dtype is a no-op
/// (returns the same tensor). Cast to/from complex is out of scope (`Type`).
/// The result is a fresh base tensor; a view is never mutated.
pub fn tensor_cast(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    t: &Tensor,
    dtype: TensorDtype,
) -> Result<Tensor, LanaError> {
    if t.dtype == TensorDtype::Complex || dtype == TensorDtype::Complex {
        return Err(LanaError::Type);
    }
    if t.dtype == dtype {
        return Ok(t.clone());
    }
    let mut r = tensor_new_dtype(alloc, t.ndim, &t.shape, dtype)?;
    let total: usize = t.shape.iter().product();
    let mut idx = vec![0usize; t.ndim];
    for lin in 0..total {
        let mut rem = lin;
        for i in (0..t.ndim).rev() {
            idx[i] = if t.shape[i] == 0 { 0 } else { rem % t.shape[i] };
            rem /= t.shape[i];
        }
        let mut src = 0usize;
        for i in 0..t.ndim {
            src += idx[i] * t.strides[i];
        }
        tensor_set_real(&mut r, lin, tensor_get_real(t, t.offset + src));
    }
    Ok(r)
}

/// Extract a shape from a `VAL_ARRAY` of numbers, mirroring
/// `tensor_shape_from_array`. Non-array input raises `Type`; non-number items
/// raise `Type`; negative or non-integer items raise `InvalidParameters`. The
/// shape buffer is accounted through `alloc`.
pub fn tensor_shape_from_array(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    v: &Value,
) -> Result<Vec<usize>, LanaError> {
    let ValueKind::Array(array) = &v.kind else {
        return Err(LanaError::Type);
    };
    let array = array.lock().unwrap();
    if array.items.len() > TENSOR_MAX_RANK {
        return Err(LanaError::InvalidParameters);
    }
    if !array.items.is_empty()
        && alloc(array.items.len() * std::mem::size_of::<usize>()) != LanaError::Ok
    {
        return Err(LanaError::Oom);
    }
    let mut shape = Vec::with_capacity(array.items.len());
    for item in &array.items {
        let ValueKind::Number(n) = item.kind else {
            return Err(LanaError::Type);
        };
        shape.push(tensor_dimension(n)?);
    }
    Ok(shape)
}

/// Compute the row-major broadcast strides of `t` against an output shape of
/// `out_ndim` dims, mirroring `tensor_broadcast_strides`.
fn tensor_broadcast_strides(t: &Tensor, out_ndim: usize, out_shape: &[usize], strides: &mut [usize]) {
    let offset = out_ndim - t.ndim;
    for i in 0..out_ndim {
        if i < offset {
            strides[i] = 0;
        } else {
            let ti = i - offset;
            strides[i] = if t.shape[ti] == 1 && out_shape[i] != 1 {
                0
            } else {
                t.strides[ti]
            };
        }
    }
}

/// Element-wise binary op with NumPy broadcasting, mirroring
/// `tensor_elementwise`. `op`: 0=add 1=sub 2=mul 3=div.
pub fn tensor_elementwise(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    a: &Tensor,
    b: &Tensor,
    op: u32,
) -> Result<Tensor, LanaError> {
    // LIP-027: element-wise requires the same dtype; no silent promotion.
    if a.dtype != b.dtype {
        return Err(LanaError::Type);
    }
    let out_ndim = a.ndim.max(b.ndim);
    if out_ndim > 0
        && alloc(out_ndim * std::mem::size_of::<usize>()) != LanaError::Ok
    {
        return Err(LanaError::Oom);
    }
    let mut out_shape = vec![0usize; out_ndim];
    for i in 0..out_ndim {
        let ai = if i < out_ndim - a.ndim {
            1
        } else {
            a.shape[i - (out_ndim - a.ndim)]
        };
        let bi = if i < out_ndim - b.ndim {
            1
        } else {
            b.shape[i - (out_ndim - b.ndim)]
        };
        if ai == bi {
            out_shape[i] = ai;
        } else if ai == 1 {
            out_shape[i] = bi;
        } else if bi == 1 {
            out_shape[i] = ai;
        } else {
            return Err(LanaError::InvalidParameters);
        }
    }
    let mut r = tensor_new_dtype(alloc, out_ndim, &out_shape, a.dtype)?;
    if out_ndim > 0
        && alloc(out_ndim * std::mem::size_of::<usize>()) != LanaError::Ok
    {
        return Err(LanaError::Oom);
    }
    let mut a_strides = vec![0usize; out_ndim];
    if out_ndim > 0
        && alloc(out_ndim * std::mem::size_of::<usize>()) != LanaError::Ok
    {
        return Err(LanaError::Oom);
    }
    let mut b_strides = vec![0usize; out_ndim];
    tensor_broadcast_strides(a, out_ndim, &out_shape, &mut a_strides);
    tensor_broadcast_strides(b, out_ndim, &out_shape, &mut b_strides);
    let mut total: usize = 1;
    for i in 0..out_ndim {
        total *= out_shape[i];
    }
    if out_ndim > 0
        && alloc(out_ndim * std::mem::size_of::<usize>()) != LanaError::Ok
    {
        return Err(LanaError::Oom);
    }
    let mut idx = vec![0usize; out_ndim];
    let complex = a.is_complex;
    for lin in 0..total {
        let mut rem = lin;
        for i in (0..out_ndim).rev() {
            idx[i] = if out_shape[i] == 0 { 0 } else { rem % out_shape[i] };
            rem /= out_shape[i];
        }
        let mut ai = 0usize;
        let mut bi = 0usize;
        for i in 0..out_ndim {
            ai += idx[i] * a_strides[i];
            bi += idx[i] * b_strides[i];
        }
        if complex {
            let ar = tensor_get_real(&a, a.offset + ai);
            let aim = tensor_get_imag(&a, a.offset + ai);
            let br = tensor_get_real(&b, b.offset + bi);
            let bim = tensor_get_imag(&b, b.offset + bi);
            let (rr, ri) = match op {
                0 => (ar + br, aim + bim),
                1 => (ar - br, aim - bim),
                2 => (ar * br - aim * bim, ar * bim + aim * br),
                _ => {
                    let den = br * br + bim * bim;
                    if den == 0.0 {
                        return Err(LanaError::InvalidParameters);
                    }
                    ((ar * br + aim * bim) / den, (aim * br - ar * bim) / den)
                }
            };
            tensor_set_real(&mut r, lin, rr);
            tensor_set_imag(&mut r, lin, ri);
        } else {
            let av = tensor_get_real(&a, a.offset + ai);
            let bv = tensor_get_real(&b, b.offset + bi);
            tensor_set_real(&mut r, lin, match op {
                0 => av + bv,
                1 => av - bv,
                2 => av * bv,
                _ => {
                    if bv == 0.0 {
                        return Err(LanaError::InvalidParameters);
                    }
                    av / bv
                }
            });
        }
    }
    Ok(r)
}

/// LIP-027: element-wise between a tensor and a scalar number. The scalar
/// adopts the tensor's dtype (cast to it), so `f16_tensor + 1.0` is "f16".
pub fn tensor_elementwise_scalar(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    t: &Tensor,
    s: f64,
    op: u32,
) -> Result<Tensor, LanaError> {
    let mut r = tensor_new_dtype(alloc, t.ndim, &t.shape, t.dtype)?;
    let total: usize = t.shape.iter().product();
    let mut idx = vec![0usize; t.ndim];
    for lin in 0..total {
        let mut rem = lin;
        for i in (0..t.ndim).rev() {
            idx[i] = if t.shape[i] == 0 { 0 } else { rem % t.shape[i] };
            rem /= t.shape[i];
        }
        let mut src = 0usize;
        for i in 0..t.ndim {
            src += idx[i] * t.strides[i];
        }
        if t.is_complex {
            let ar = tensor_get_real(t, t.offset + src);
            let aim = tensor_get_imag(t, t.offset + src);
            let (rr, ri) = match op {
                0 => (ar + s, aim),
                1 => (ar - s, aim),
                2 => (ar * s, aim * s),
                _ => {
                    if s == 0.0 {
                        return Err(LanaError::InvalidParameters);
                    }
                    (ar / s, aim / s)
                }
            };
            tensor_set_real(&mut r, lin, rr);
            tensor_set_imag(&mut r, lin, ri);
        } else {
            let v = tensor_get_real(t, t.offset + src);
            tensor_set_real(&mut r, lin, match op {
                0 => v + s,
                1 => v - s,
                2 => v * s,
                _ => {
                    if s == 0.0 {
                        return Err(LanaError::InvalidParameters);
                    }
                    v / s
                }
            });
        }
    }
    Ok(r)
}

/// General matmul following NumPy semantics, mirroring `tensor_matmul`.
/// LIP-027: relative precision rank used to pick the default matmul output
/// dtype for a mixed-dtype pair. Higher rank wins on a tie; f16 and bf16 are
/// equal (both 16-bit). Complex is handled separately by the caller.
fn dtype_rank(d: TensorDtype) -> i32 {
    match d {
        TensorDtype::F64 => 3,
        TensorDtype::F32 => 2,
        TensorDtype::F16 | TensorDtype::Bf16 => 1,
        TensorDtype::Complex => 0,
    }
}

/// LIP-027: default matmul output dtype. Same dtype -> that dtype; otherwise
/// the higher-precision operand (f64/f32 mix -> f64, f32/f16 -> f32).
pub fn matmul_default_dtype(a: &Tensor, b: &Tensor) -> TensorDtype {
    if a.dtype == b.dtype {
        a.dtype
    } else if dtype_rank(a.dtype) >= dtype_rank(b.dtype) {
        a.dtype
    } else {
        b.dtype
    }
}

/// LIP-027: whether matmul accumulates in binary32. f16/bf16 inputs always
/// accumulate in fp32 regardless of the output dtype; f32 x f32 also uses fp32.
/// A binary64 (or complex) input forces fp64 accumulation so it is never
/// silently downcast.
pub fn matmul_accumulation_fp32(a: &Tensor, b: &Tensor) -> bool {
    if a.dtype == TensorDtype::F16
        || a.dtype == TensorDtype::Bf16
        || b.dtype == TensorDtype::F16
        || b.dtype == TensorDtype::Bf16
    {
        return true;
    }
    a.dtype == TensorDtype::F32 && b.dtype == TensorDtype::F32
}

pub fn tensor_matmul(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    a: &Tensor,
    b: &Tensor,
    out_dtype: TensorDtype,
) -> Result<Tensor, LanaError> {
    if a.is_complex != b.is_complex {
        return Err(LanaError::Type);
    }
    if a.ndim == 0 || b.ndim == 0 {
        return Err(LanaError::InvalidParameters);
    }
    let complex = a.is_complex;
    // LIP-027: the output dtype must agree with the complexness of the inputs:
    // complex inputs require a complex output, real inputs a real output.
    if complex != (out_dtype == TensorDtype::Complex) {
        return Err(LanaError::Type);
    }
    let a_ndim = a.ndim;
    let b_ndim = b.ndim;
    let a_rows = if a_ndim == 1 { 1 } else { a.shape[a_ndim - 2] };
    let a_cols = a.shape[a_ndim - 1];
    let b_rows = if b_ndim == 1 { b.shape[0] } else { b.shape[b_ndim - 2] };
    let b_cols = if b_ndim == 1 { 1 } else { b.shape[b_ndim - 1] };
    if a_cols != b_rows {
        return Err(LanaError::InvalidParameters);
    }
    let k = a_cols;
    let a_batch = if a_ndim <= 2 { 0 } else { a_ndim - 2 };
    let b_batch = if b_ndim <= 2 { 0 } else { b_ndim - 2 };
    let batch_ndim = a_batch.max(b_batch);
    if batch_ndim > 0
        && alloc(batch_ndim * std::mem::size_of::<usize>()) != LanaError::Ok
    {
        return Err(LanaError::Oom);
    }
    let mut batch_shape = vec![0usize; batch_ndim];
    for i in 0..batch_ndim {
        let ai = if i < batch_ndim - a_batch {
            1
        } else {
            a.shape[i - (batch_ndim - a_batch)]
        };
        let bi = if i < batch_ndim - b_batch {
            1
        } else {
            b.shape[i - (batch_ndim - b_batch)]
        };
        if ai == bi {
            batch_shape[i] = ai;
        } else if ai == 1 {
            batch_shape[i] = bi;
        } else if bi == 1 {
            batch_shape[i] = ai;
        } else {
            return Err(LanaError::InvalidParameters);
        }
    }
    let a_vec = a_ndim == 1;
    let b_vec = b_ndim == 1;
    let out_ndim = batch_ndim + if a_vec { 0 } else { 1 } + if b_vec { 0 } else { 1 };
    if out_ndim > 0
        && alloc(out_ndim * std::mem::size_of::<usize>()) != LanaError::Ok
    {
        return Err(LanaError::Oom);
    }
    let mut out_shape = vec![0usize; out_ndim];
    for i in 0..batch_ndim {
        out_shape[i] = batch_shape[i];
    }
    let mut pos = batch_ndim;
    if !a_vec {
        out_shape[pos] = a_rows;
        pos += 1;
    }
    if !b_vec {
        out_shape[pos] = b_cols;
    }
    let mut r = tensor_new_dtype(alloc, out_ndim, &out_shape, out_dtype)?;
    if batch_ndim > 0
        && alloc(batch_ndim * std::mem::size_of::<usize>()) != LanaError::Ok
    {
        return Err(LanaError::Oom);
    }
    let mut a_batch_strides = vec![0usize; batch_ndim];
    if batch_ndim > 0
        && alloc(batch_ndim * std::mem::size_of::<usize>()) != LanaError::Ok
    {
        return Err(LanaError::Oom);
    }
    let mut b_batch_strides = vec![0usize; batch_ndim];
    for i in 0..batch_ndim {
        let a_dim = if i < batch_ndim - a_batch {
            1
        } else {
            a.shape[i - (batch_ndim - a_batch)]
        };
        let a_stride = if i < batch_ndim - a_batch {
            0
        } else {
            a.strides[i - (batch_ndim - a_batch)]
        };
        a_batch_strides[i] = if a_dim == 1 && batch_shape[i] != 1 {
            0
        } else {
            a_stride
        };
        let b_dim = if i < batch_ndim - b_batch {
            1
        } else {
            b.shape[i - (batch_ndim - b_batch)]
        };
        let b_stride = if i < batch_ndim - b_batch {
            0
        } else {
            b.strides[i - (batch_ndim - b_batch)]
        };
        b_batch_strides[i] = if b_dim == 1 && batch_shape[i] != 1 {
            0
        } else {
            b_stride
        };
    }
    let mut batch_total: usize = 1;
    for i in 0..batch_ndim {
        batch_total *= batch_shape[i];
    }
    let a_row_stride = if a_ndim == 1 { 0 } else { a.strides[a_ndim - 2] };
    let a_col_stride = a.strides[a_ndim - 1];
    let b_row_stride = if b_ndim == 1 { b.strides[0] } else { b.strides[b_ndim - 2] };
    let b_col_stride = if b_ndim == 1 { 0 } else { b.strides[b_ndim - 1] };
    let m = a_rows;
    let n = b_cols;
    let mult = if complex { 2 } else { 1 };
    // BLAS needs row-major contiguous cores. A promoted vector is contiguous
    // when its one live stride is 1; a matrix operand when its column stride
    // is 1. Anything else is gathered into accounted scratch per batch
    // element (LIP-004 section 5).
    // LIP-027: a non-f64 real dtype stores compact bytes (2/4 per element), so
    // the direct `*const f64` fast path is invalid; gather it into a double
    // scratch buffer via tensor_get_real. Complex (interleaved doubles) and
    // f64 read directly.
    let pack_a = a_col_stride != 1
        || (a.dtype != TensorDtype::F64 && a.dtype != TensorDtype::Complex);
    let pack_b = b_col_stride != 1
        || (b.dtype != TensorDtype::F64 && b.dtype != TensorDtype::Complex);
    // Leading dimensions for direct (non-packed) operands: the true row
    // stride of the view, which BLAS accepts instead of a copy. A promoted
    // vector has one row (lda = K) or one column (ldb = its row stride).
    let a_ld = if pack_a || a_vec { k } else { a_row_stride };
    let b_ld = if pack_b { n } else { b_row_stride };
    let a_core = m * k * mult;
    let b_core = k * n * mult;
    let mut a_packed: Vec<f64> = Vec::new();
    let mut b_packed: Vec<f64> = Vec::new();
    if pack_a && a_core > 0 {
        if alloc(a_core * std::mem::size_of::<f64>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        a_packed = vec![0.0; a_core];
    }
    if pack_b && b_core > 0 {
        if alloc(b_core * std::mem::size_of::<f64>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        b_packed = vec![0.0; b_core];
    }
    // LIP-027: the backend gemm accumulates in binary64. When the output dtype
    // is narrower than f64 (f32/f16/bf16), the result is staged in a scratch
    // buffer and converted element-wise into the compact output buffer.
    let c_core = m * n * mult;
    let mut c_scratch: Vec<f64> = Vec::new();
    if c_core > 0 {
        if alloc(c_core * std::mem::size_of::<f64>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        c_scratch = vec![0.0; c_core];
    }
    for batch in 0..batch_total {
        let mut rem = batch;
        let mut a_off = 0usize;
        let mut b_off = 0usize;
        for i in (0..batch_ndim).rev() {
            let bi = if batch_shape[i] == 0 { 0 } else { rem % batch_shape[i] };
            rem /= batch_shape[i];
            a_off += bi * a_batch_strides[i];
            b_off += bi * b_batch_strides[i];
        }
        let mut a_ptr = unsafe { (a.data.as_ptr() as *const f64).add((a.offset + a_off) * mult) };
        let mut b_ptr = unsafe { (b.data.as_ptr() as *const f64).add((b.offset + b_off) * mult) };
        if pack_a && a_core > 0 {
            for i in 0..m {
                for kk in 0..k {
                    let elem = a.offset + a_off + i * a_row_stride + kk * a_col_stride;
                    let dst = (i * k + kk) * mult;
                    a_packed[dst] = tensor_get_real(&a, elem);
                    if mult == 2 {
                        a_packed[dst + 1] = tensor_get_imag(&a, elem);
                    }
                }
            }
            a_ptr = a_packed.as_ptr();
        }
        if pack_b && b_core > 0 {
            for kk in 0..k {
                for j in 0..n {
                    let elem = b.offset + b_off + kk * b_row_stride + j * b_col_stride;
                    let dst = (kk * n + j) * mult;
                    b_packed[dst] = tensor_get_real(&b, elem);
                    if mult == 2 {
                        b_packed[dst + 1] = tensor_get_imag(&b, elem);
                    }
                }
            }
            b_ptr = b_packed.as_ptr();
        }
        let c_ptr = c_scratch.as_mut_ptr();
        backend::backend_gemm(
            m, k, n, complex, matmul_accumulation_fp32(a, b), a_ptr, a_ld, b_ptr, b_ld, c_ptr,
        );
        // Stage the binary64 result into the compact output buffer.
        for i in 0..c_core {
            let elem = batch * m * n + i / mult;
            if mult == 2 {
                if i % 2 == 0 {
                    tensor_set_real(&mut r, elem, c_scratch[i]);
                } else {
                    tensor_set_imag(&mut r, elem, c_scratch[i]);
                }
            } else {
                tensor_set_real(&mut r, elem, c_scratch[i]);
            }
        }
    }
    Ok(r)
}

/// Explicit GPU matmul (LIP-004 section 5). Mirrors `tensor_matmul`'s shape,
/// batch, and broadcasting logic, but dispatches each batch element to the
/// Metal device in float32. Complex operands are rejected (float32 complex
/// Metal is a later extension); the binary64 -> float32 downcast is explicit
/// and the float32 -> binary64 upcast is exact. The caller attaches the
/// APPROXIMATE derivation.
pub fn tensor_gpu_matmul(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    a: &Tensor,
    b: &Tensor,
) -> Result<Tensor, LanaError> {
    if a.is_complex || b.is_complex {
        return Err(LanaError::Type);
    }
    if a.ndim == 0 || b.ndim == 0 {
        return Err(LanaError::InvalidParameters);
    }
    let a_ndim = a.ndim;
    let b_ndim = b.ndim;
    let a_rows = if a_ndim == 1 { 1 } else { a.shape[a_ndim - 2] };
    let a_cols = a.shape[a_ndim - 1];
    let b_rows = if b_ndim == 1 { b.shape[0] } else { b.shape[b_ndim - 2] };
    let b_cols = if b_ndim == 1 { 1 } else { b.shape[b_ndim - 1] };
    if a_cols != b_rows {
        return Err(LanaError::InvalidParameters);
    }
    let k = a_cols;
    let a_batch = if a_ndim <= 2 { 0 } else { a_ndim - 2 };
    let b_batch = if b_ndim <= 2 { 0 } else { b_ndim - 2 };
    let batch_ndim = a_batch.max(b_batch);
    if batch_ndim > 0
        && alloc(batch_ndim * std::mem::size_of::<usize>()) != LanaError::Ok
    {
        return Err(LanaError::Oom);
    }
    let mut batch_shape = vec![0usize; batch_ndim];
    for i in 0..batch_ndim {
        let ai = if i < batch_ndim - a_batch {
            1
        } else {
            a.shape[i - (batch_ndim - a_batch)]
        };
        let bi = if i < batch_ndim - b_batch {
            1
        } else {
            b.shape[i - (batch_ndim - b_batch)]
        };
        if ai == bi {
            batch_shape[i] = ai;
        } else if ai == 1 {
            batch_shape[i] = bi;
        } else if bi == 1 {
            batch_shape[i] = ai;
        } else {
            return Err(LanaError::InvalidParameters);
        }
    }
    let a_vec = a_ndim == 1;
    let b_vec = b_ndim == 1;
    let out_ndim = batch_ndim + if a_vec { 0 } else { 1 } + if b_vec { 0 } else { 1 };
    if out_ndim > 0
        && alloc(out_ndim * std::mem::size_of::<usize>()) != LanaError::Ok
    {
        return Err(LanaError::Oom);
    }
    let mut out_shape = vec![0usize; out_ndim];
    for i in 0..batch_ndim {
        out_shape[i] = batch_shape[i];
    }
    let mut pos = batch_ndim;
    if !a_vec {
        out_shape[pos] = a_rows;
        pos += 1;
    }
    if !b_vec {
        out_shape[pos] = b_cols;
    }
    let mut r = tensor_new(alloc, out_ndim, &out_shape, false)?;
    if batch_ndim > 0
        && alloc(batch_ndim * std::mem::size_of::<usize>()) != LanaError::Ok
    {
        return Err(LanaError::Oom);
    }
    let mut a_batch_strides = vec![0usize; batch_ndim];
    if batch_ndim > 0
        && alloc(batch_ndim * std::mem::size_of::<usize>()) != LanaError::Ok
    {
        return Err(LanaError::Oom);
    }
    let mut b_batch_strides = vec![0usize; batch_ndim];
    for i in 0..batch_ndim {
        let a_dim = if i < batch_ndim - a_batch {
            1
        } else {
            a.shape[i - (batch_ndim - a_batch)]
        };
        let a_stride = if i < batch_ndim - a_batch {
            0
        } else {
            a.strides[i - (batch_ndim - a_batch)]
        };
        a_batch_strides[i] = if a_dim == 1 && batch_shape[i] != 1 {
            0
        } else {
            a_stride
        };
        let b_dim = if i < batch_ndim - b_batch {
            1
        } else {
            b.shape[i - (batch_ndim - b_batch)]
        };
        let b_stride = if i < batch_ndim - b_batch {
            0
        } else {
            b.strides[i - (batch_ndim - b_batch)]
        };
        b_batch_strides[i] = if b_dim == 1 && batch_shape[i] != 1 {
            0
        } else {
            b_stride
        };
    }
    let mut batch_total: usize = 1;
    for i in 0..batch_ndim {
        batch_total *= batch_shape[i];
    }
    let a_row_stride = if a_ndim == 1 { 0 } else { a.strides[a_ndim - 2] };
    let a_col_stride = a.strides[a_ndim - 1];
    let b_row_stride = if b_ndim == 1 { b.strides[0] } else { b.strides[b_ndim - 2] };
    let b_col_stride = if b_ndim == 1 { 0 } else { b.strides[b_ndim - 1] };
    let m = a_rows;
    let n = b_cols;
    // Metal needs contiguous row-major float32 cores, so the binary64 operands
    // are downcast into float32 scratch per batch element (LIP-004 section 5:
    // the downcast is explicit, never silent).
    let a_core = m * k;
    let b_core = k * n;
    let c_core = m * n;
    if alloc(a_core * std::mem::size_of::<f32>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    if alloc(b_core * std::mem::size_of::<f32>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    if alloc(c_core * std::mem::size_of::<f32>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    let mut a_float = vec![0.0f32; a_core];
    let mut b_float = vec![0.0f32; b_core];
    let mut c_float = vec![0.0f32; c_core];
    for batch in 0..batch_total {
        let mut rem = batch;
        let mut a_off = 0usize;
        let mut b_off = 0usize;
        for i in (0..batch_ndim).rev() {
            let bi = if batch_shape[i] == 0 { 0 } else { rem % batch_shape[i] };
            rem /= batch_shape[i];
            a_off += bi * a_batch_strides[i];
            b_off += bi * b_batch_strides[i];
        }
        for i in 0..m {
            for kk in 0..k {
                a_float[i * k + kk] =
                    tensor_get_real(&a, a.offset + a_off + i * a_row_stride + kk * a_col_stride) as f32;
            }
        }
        for kk in 0..k {
            for j in 0..n {
                b_float[kk * n + j] =
                    tensor_get_real(&b, b.offset + b_off + kk * b_row_stride + j * b_col_stride) as f32;
            }
        }
        if !metal::metal_sgemm(m, k, n, a_float.as_ptr(), b_float.as_ptr(), c_float.as_mut_ptr()) {
            return Err(LanaError::UnsupportedOperation);
        }
        for i in 0..c_core {
            tensor_set_real(&mut r, batch * c_core + i, c_float[i] as f64);
        }
    }
    Ok(r)
}

/// Reduce one strided fiber. `op`: 0=sum 1=mean 2=max 3=min. `offset` is
/// relative to the tensor's first element; the tensor's own view offset is
/// folded in here so every caller is automatically view-correct.
/// LIP-027: sum/mean accumulate in binary32 for f16/bf16 inputs.
fn reduction_accumulation_fp32(t: &Tensor) -> bool {
    t.dtype == TensorDtype::F16 || t.dtype == TensorDtype::Bf16
}

fn tensor_reduce_fiber(
    t: &Tensor, offset: usize, count: usize, stride: usize, op: u32, fp32: bool,
) -> Result<(f64, f64), LanaError> {
    if t.is_complex && op >= 2 { return Err(LanaError::Type); }
    if count == 0 && op != 0 { return Err(LanaError::InvalidParameters); }
    let mut re = if op == 2 { f64::NEG_INFINITY } else if op == 3 { f64::INFINITY } else { 0.0 };
    let mut re32 = 0.0f32;
    let mut im = 0.0;
    for i in 0..count {
        let index = t.offset + offset + i * stride;
        let v = tensor_get_real(&t, index);
        if !v.is_finite() { return Err(LanaError::InvalidParameters); }
        if op < 2 {
            if fp32 { re32 += v as f32; } else { re += v; }
        }
        else if if op == 2 { v > re } else { v < re } { re = v; }
        if t.is_complex {
            let component = tensor_get_imag(&t, index);
            if !component.is_finite() { return Err(LanaError::InvalidParameters); }
            im += component;
        }
    }
    if fp32 && op < 2 { re = re32 as f64; }
    if op == 1 { re /= count as f64; im /= count as f64; }
    if !re.is_finite() || !im.is_finite() { return Err(LanaError::InvalidParameters); }
    Ok((re, im))
}

/// Full reductions return a number or a rank-zero complex tensor. A view may
/// be non-contiguous, so this traverses the strided layout per dimension
/// instead of assuming a flat fiber.
pub fn tensor_reduce(
    alloc: &mut dyn FnMut(usize) -> LanaError, t: &Tensor, op: u32,
) -> Result<Value, LanaError> {
    if t.is_complex && op >= 2 { return Err(LanaError::Type); }
    let total: usize = t.shape.iter().product();
    if total == 0 && op != 0 { return Err(LanaError::InvalidParameters); }
    let fp32 = reduction_accumulation_fp32(t);
    let mut re = if op == 2 { f64::NEG_INFINITY } else if op == 3 { f64::INFINITY } else { 0.0 };
    let mut re32 = 0.0f32;
    let mut im = 0.0;
    for lin in 0..total {
        let mut rem = lin;
        let mut index = t.offset;
        for d in (0..t.ndim).rev() {
            index += if t.shape[d] == 0 { 0 } else { rem % t.shape[d] } * t.strides[d];
            rem /= t.shape[d];
        }
        let v = tensor_get_real(&t, index);
        if !v.is_finite() { return Err(LanaError::InvalidParameters); }
        if op < 2 {
            if fp32 { re32 += v as f32; } else { re += v; }
        }
        else if if op == 2 { v > re } else { v < re } { re = v; }
        if t.is_complex {
            let component = tensor_get_imag(&t, index);
            if !component.is_finite() { return Err(LanaError::InvalidParameters); }
            im += component;
        }
    }
    if fp32 && op < 2 { re = re32 as f64; }
    if op == 1 { re /= total as f64; im /= total as f64; }
    if !re.is_finite() || !im.is_finite() { return Err(LanaError::InvalidParameters); }
    if !t.is_complex { return Ok(Value::number(re)); }
    let mut r = tensor_new(alloc, 0, &[], true)?;
    tensor_set_real(&mut r, 0, re);
    tensor_set_imag(&mut r, 0, im);
    Ok(Value::tensor(Arc::new(r)))
}

/// Axis reductions remove the selected dimension, preserving the element type.
pub fn tensor_reduce_axis(
    alloc: &mut dyn FnMut(usize) -> LanaError, t: &Tensor, op: u32, axis: &Value,
) -> Result<Value, LanaError> {
    if t.is_complex && op >= 2 { return Err(LanaError::Type); }
    let ValueKind::Number(n) = axis.kind else { return Err(LanaError::Type); };
    if !n.is_finite() || n.floor() != n || n < -(t.ndim as f64) || n >= t.ndim as f64 {
        return Err(LanaError::InvalidParameters);
    }
    let axis = if n < 0.0 { n + t.ndim as f64 } else { n } as usize;
    let count = t.shape[axis];
    if count == 0 && op != 0 { return Err(LanaError::InvalidParameters); }
    let mut shape = [0; TENSOR_MAX_RANK];
    let mut j = 0;
    for (i, &dim) in t.shape.iter().enumerate() {
        if i != axis { shape[j] = dim; j += 1; }
    }
    let mut r = tensor_new_dtype(alloc, t.ndim - 1, &shape[..j], t.dtype)?;
    let total = r.shape.iter().product();
    let fp32 = reduction_accumulation_fp32(t);
    for i in 0..total {
        let mut offset = 0;
        let mut remaining = i;
        for d in (0..t.ndim).rev() {
            if d == axis { continue; }
            offset += (remaining % t.shape[d]) * t.strides[d];
            remaining /= t.shape[d];
        }
        let (re, im) = tensor_reduce_fiber(t, offset, count, t.strides[axis], op, fp32)?;
        tensor_set_real(&mut r, i, re);
        if t.is_complex { tensor_set_imag(&mut r, i, im); }
    }
    Ok(Value::tensor(Arc::new(r)))
}

/// Total number of real elements in a tensor (autodiff is real-only),
/// mirroring `tensor_element_count` in `vm/c/vm.c`.
pub fn tensor_element_count(t: &Tensor) -> usize {
    let mut total = 1usize;
    for i in 0..t.ndim {
        total *= t.shape[i];
    }
    total
}

/// Whether two tensors have identical shape, mirroring `tensor_shape_equal`.
pub fn tensor_shape_equal(a: &Tensor, b: &Tensor) -> bool {
    if a.ndim != b.ndim {
        return false;
    }
    for i in 0..a.ndim {
        if a.shape[i] != b.shape[i] {
            return false;
        }
    }
    true
}

/* ===== LIP-008 uncertainty-carrying tensor helpers ===== */

/// Detect an uncertain tensor: a map with exactly the "prediction" and
/// "uncertainty" keys, both tensors. Returns (prediction, variance, is_uncertain).
/// A bare tensor is certain (variance `None`). Any other value is a type error.
pub fn tensor_uncertainty_unpack(
    v: &Value,
) -> Result<(Arc<Tensor>, Option<Arc<Tensor>>, bool), LanaError> {
    match &v.kind {
        ValueKind::Tensor(t) => Ok((t.clone(), None, false)),
        ValueKind::Map(map) => {
            let map = map.lock().unwrap();
            if map.entries.len() != 2 {
                return Err(LanaError::Type);
            }
            let pred = map.get("prediction").ok_or(LanaError::Type)?;
            let var = map.get("uncertainty").ok_or(LanaError::Type)?;
            let (ValueKind::Tensor(p), ValueKind::Tensor(v)) = (&pred.kind, &var.kind) else {
                return Err(LanaError::Type);
            };
            Ok((p.clone(), Some(v.clone()), true))
        }
        _ => Err(LanaError::Type),
    }
}

/// Build the { prediction, uncertainty } map result, mirroring
/// `tensor_uncertain_result`.
pub fn tensor_uncertain_result(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    pred: Arc<Tensor>,
    var: Arc<Tensor>,
) -> Result<Value, LanaError> {
    if alloc(std::mem::size_of::<Map>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    let mut map = Map::new(2);
    map.set(Arc::from("prediction"), Value::tensor(pred), false)?;
    map.set(Arc::from("uncertainty"), Value::tensor(var), false)?;
    Ok(Value::map(Arc::new(Mutex::new(map))))
}

/// A zero real tensor with the same shape as `t` (the variance of a certain
/// operand), mirroring `tensor_zeros_like`.
pub fn tensor_zeros_like(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    t: &Tensor,
) -> Result<Tensor, LanaError> {
    tensor_new(alloc, t.ndim, &t.shape, false)
}

/// Whether every element of a freshly-allocated (contiguous) tensor is finite,
/// mirroring `tensor_all_finite`.
pub fn tensor_all_finite(t: &Tensor) -> bool {
    let total: usize = t.shape.iter().product();
    let mult = if t.is_complex { 2 } else { 1 };
    for i in 0..total * mult {
        if !tensor_get_real(&t, t.offset + i).is_finite() {
            return false;
        }
    }
    true
}

/// First-order variance propagation for element-wise + - * / (real-only),
/// mirroring `tensor_elementwise_uncertain`.
pub fn tensor_elementwise_uncertain(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    a_pred: &Tensor,
    a_var: &Tensor,
    b_pred: &Tensor,
    b_var: &Tensor,
    op: u32,
) -> Result<Value, LanaError> {
    if a_pred.is_complex || b_pred.is_complex || a_var.is_complex || b_var.is_complex {
        return Err(LanaError::Type);
    }
    let pred = tensor_elementwise(alloc, a_pred, b_pred, op)?;
    let var = if op == 0 || op == 1 {
        tensor_elementwise(alloc, a_var, b_var, 0)?
    } else if op == 2 {
        let b_sq = tensor_elementwise(alloc, b_pred, b_pred, 2)?;
        let a_sq = tensor_elementwise(alloc, a_pred, a_pred, 2)?;
        let t1 = tensor_elementwise(alloc, a_var, &b_sq, 2)?;
        let t2 = tensor_elementwise(alloc, b_var, &a_sq, 2)?;
        tensor_elementwise(alloc, &t1, &t2, 0)?
    } else {
        let b_sq = tensor_elementwise(alloc, b_pred, b_pred, 2)?;
        let b_4 = tensor_elementwise(alloc, &b_sq, &b_sq, 2)?;
        let a_sq = tensor_elementwise(alloc, a_pred, a_pred, 2)?;
        let t1 = tensor_elementwise(alloc, a_var, &b_sq, 3)?;
        let t2 = tensor_elementwise(alloc, b_var, &a_sq, 2)?;
        let t3 = tensor_elementwise(alloc, &t2, &b_4, 3)?;
        tensor_elementwise(alloc, &t1, &t3, 0)?
    };
    if !tensor_all_finite(&var) {
        return Err(LanaError::InvalidParameters);
    }
    tensor_uncertain_result(alloc, Arc::new(pred), Arc::new(var))
}

/// First-order variance propagation for matmul (real-only), mirroring
/// `tensor_matmul_uncertain`.
pub fn tensor_matmul_uncertain(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    a_pred: &Tensor,
    a_var: &Tensor,
    b_pred: &Tensor,
    b_var: &Tensor,
) -> Result<Value, LanaError> {
    if a_pred.is_complex || b_pred.is_complex || a_var.is_complex || b_var.is_complex {
        return Err(LanaError::Type);
    }
    let pred = tensor_matmul(alloc, a_pred, b_pred, matmul_default_dtype(a_pred, b_pred))?;
    let b_sq = tensor_elementwise(alloc, b_pred, b_pred, 2)?;
    let a_sq = tensor_elementwise(alloc, a_pred, a_pred, 2)?;
    let t1 = tensor_matmul(alloc, a_var, &b_sq, matmul_default_dtype(a_var, &b_sq))?;
    let t2 = tensor_matmul(alloc, &a_sq, b_var, matmul_default_dtype(&a_sq, b_var))?;
    let var = tensor_elementwise(alloc, &t1, &t2, 0)?;
    if !tensor_all_finite(&var) {
        return Err(LanaError::InvalidParameters);
    }
    tensor_uncertain_result(alloc, Arc::new(pred), Arc::new(var))
}

/// Reduce to a tensor, wrapping a full-reduction number in a rank-0 tensor,
/// mirroring `tensor_reduce_tensor`.
pub fn tensor_reduce_tensor(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    t: &Tensor,
    op: u32,
    axis: Option<&Value>,
) -> Result<Arc<Tensor>, LanaError> {
    let result = match axis {
        Some(axis) => tensor_reduce_axis(alloc, t, op, axis)?,
        None => tensor_reduce(alloc, t, op)?,
    };
    match result.kind {
        ValueKind::Tensor(t) => Ok(t),
        ValueKind::Number(n) => {
            let mut r = tensor_new(alloc, 0, &[], false)?;
            tensor_set_real(&mut r, 0, n);
            Ok(Arc::new(r))
        }
        _ => Err(LanaError::Type),
    }
}

/// Scale a freshly-allocated (contiguous) tensor by a constant factor,
/// mirroring `tensor_scale`.
pub fn tensor_scale(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    t: &Tensor,
    factor: f64,
) -> Result<Tensor, LanaError> {
    let mut r = tensor_new(alloc, t.ndim, &t.shape, t.is_complex)?;
    let total: usize = t.shape.iter().product();
    for e in 0..total {
        tensor_set_real(&mut r, e, tensor_get_real(&t, t.offset + e) * factor);
        tensor_set_imag(&mut r, e, tensor_get_imag(&t, t.offset + e) * factor);
    }
    Ok(r)
}

/// First-order variance propagation for sum/mean reductions (real-only),
/// mirroring `tensor_reduce_uncertain`.
pub fn tensor_reduce_uncertain(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    pred: &Tensor,
    var: &Tensor,
    op: u32,
    axis: Option<&Value>,
) -> Result<Value, LanaError> {
    if pred.is_complex || var.is_complex {
        return Err(LanaError::Type);
    }
    let pred_tensor = tensor_reduce_tensor(alloc, pred, op, axis)?;
    let mut var_tensor = tensor_reduce_tensor(alloc, var, 0, axis)?;
    if op == 1 {
        let n = match axis {
            None => pred.shape.iter().product::<usize>(),
            Some(axis_value) => {
                let n = axis_value.as_number();
                let axis = if n < 0.0 { n + pred.ndim as f64 } else { n } as usize;
                pred.shape[axis]
            }
        };
        let factor = (n as f64) * (n as f64);
        var_tensor = Arc::new(tensor_scale(alloc, &var_tensor, 1.0 / factor)?);
    }
    if !tensor_all_finite(&var_tensor) {
        return Err(LanaError::InvalidParameters);
    }
    tensor_uncertain_result(alloc, pred_tensor, var_tensor)
}

/// A view of `t` with its last two axes swapped, mirroring
/// `tensor_transpose_last_two`. Shares the source buffer.
pub fn tensor_transpose_last_two(t: &Tensor) -> Tensor {
    if t.ndim < 2 {
        return t.clone();
    }
    let mut shape = t.shape.clone();
    let mut strides = t.strides.clone();
    let last = t.ndim - 1;
    let second = t.ndim - 2;
    shape.swap(last, second);
    strides.swap(last, second);
    Tensor {
        ndim: t.ndim,
        shape,
        strides,
        is_complex: t.is_complex,
        dtype: t.dtype,
        data: Arc::clone(&t.data),
        offset: t.offset,
        is_state: t.is_state,
    }
}

/// Outer product of two 1-D tensors: `out[i,j] = u[i] * v[j]`, mirroring
/// `tensor_outer`.
pub fn tensor_outer(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    u: &Tensor,
    v: &Tensor,
) -> Result<Tensor, LanaError> {
    let k = u.shape[0];
    let n = v.shape[0];
    let shape = [k, n];
    let mut r = tensor_new(alloc, 2, &shape, false)?;
    for i in 0..k {
        for j in 0..n {
            tensor_set_real(
                &mut r,
                i * n + j,
                tensor_get_real(&u, u.offset + i * u.strides[0])
                    * tensor_get_real(&v, v.offset + j * v.strides[0]),
            );
        }
    }
    Ok(r)
}

/// Negate a tensor into a fresh base tensor (view-correct), mirroring
/// `tensor_negate`.
pub fn tensor_negate(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    t: &Tensor,
) -> Result<Tensor, LanaError> {
    let mut r = tensor_new(alloc, t.ndim, &t.shape, false)?;
    let total = tensor_element_count(t);
    if t.ndim > 0 && alloc(t.ndim * std::mem::size_of::<usize>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    for lin in 0..total {
        let mut rem = lin;
        let mut index = t.offset;
        for d in (0..t.ndim).rev() {
            index += (if t.shape[d] == 0 { 0 } else { rem % t.shape[d] }) * t.strides[d];
            rem /= t.shape[d];
        }
        tensor_set_real(&mut r, lin, -tensor_get_real(&t, index));
    }
    Ok(r)
}

/// Sum `g` over the broadcast dimensions so the result has `target`'s shape,
/// mirroring `tensor_unbroadcast`. `g` is a contiguous base tensor; `target`
/// supplies only its shape.
pub fn tensor_unbroadcast(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    g: &Tensor,
    target: &Tensor,
) -> Result<Tensor, LanaError> {
    let out_ndim = target.ndim;
    let mut r = tensor_new(alloc, out_ndim, &target.shape, false)?;
    let total = tensor_element_count(g);
    if g.ndim > 0 && alloc(g.ndim * std::mem::size_of::<usize>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    let mut idx = vec![0usize; g.ndim];
    let offset = g.ndim - out_ndim;
    for lin in 0..total {
        let mut rem = lin;
        for i in (0..g.ndim).rev() {
            idx[i] = if g.shape[i] == 0 { 0 } else { rem % g.shape[i] };
            rem /= g.shape[i];
        }
        let mut ti = 0usize;
        for i in 0..out_ndim {
            let gi = i + offset;
            let coord = if target.shape[i] == 1 { 0 } else { idx[gi] };
            ti = ti * target.shape[i] + coord;
        }
        let cur = tensor_get_real(&r, ti);
        tensor_set_real(&mut r, ti, cur + tensor_get_real(&g, g.offset + lin));
    }
    Ok(r)
}

/// Broadcast a reduced gradient `g` back to `input`'s shape, scaled by `scale`,
/// mirroring `tensor_broadcast_reduce`. `axis` is the reduced axis (-1 for a
/// full reduction).
pub fn tensor_broadcast_reduce(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    g: &Tensor,
    input: &Tensor,
    axis: i32,
    scale: f64,
) -> Result<Tensor, LanaError> {
    let mut r = tensor_new(alloc, input.ndim, &input.shape, false)?;
    let total = tensor_element_count(input);
    if input.ndim > 0 && alloc(input.ndim * std::mem::size_of::<usize>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    let mut idx = vec![0usize; input.ndim];
    let mut g_strides = [0usize; TENSOR_MAX_RANK];
    let mut stride = 1usize;
    for i in (0..g.ndim).rev() {
        g_strides[i] = stride;
        stride *= g.shape[i];
    }
    for lin in 0..total {
        let mut rem = lin;
        for i in (0..input.ndim).rev() {
            idx[i] = if input.shape[i] == 0 { 0 } else { rem % input.shape[i] };
            rem /= input.shape[i];
        }
        let mut gi = 0usize;
        if axis < 0 {
            gi = 0;
        } else {
            let mut gd = 0usize;
            for i in 0..input.ndim {
                if i == axis as usize {
                    continue;
                }
                gi += idx[i] * g_strides[gd];
                gd += 1;
            }
        }
        tensor_set_real(&mut r, lin, tensor_get_real(&g, g.offset + gi) * scale);
    }
    Ok(r)
}

/// Recursively infer the shape of a nested array of numbers, mirroring
/// `tensor_infer_shape`. Ragged input raises `InvalidParameters`; non-number
/// leaves raise `Type`. Every per-level shape buffer is accounted through
/// `alloc`.
pub fn tensor_infer_shape(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    v: &Value,
) -> Result<Vec<usize>, LanaError> {
    tensor_infer_shape_at(alloc, v, 0)
}

fn tensor_infer_shape_at(
    alloc: &mut dyn FnMut(usize) -> LanaError,
    v: &Value,
    depth: usize,
) -> Result<Vec<usize>, LanaError> {
    match &v.kind {
        ValueKind::Number(_) => Ok(Vec::new()),
        ValueKind::Array(array) => {
            if depth == TENSOR_MAX_RANK {
                return Err(LanaError::InvalidParameters);
            }
            // A recursive lock here means an array contains itself.
            let array = array.try_lock().map_err(|_| LanaError::InvalidParameters)?;
            if array.items.is_empty() {
                if alloc(std::mem::size_of::<usize>()) != LanaError::Ok {
                    return Err(LanaError::Oom);
                }
                return Ok(vec![0]);
            }
            let sub_shape = tensor_infer_shape_at(alloc, &array.items[0], depth + 1)?;
            for item in &array.items[1..] {
                let s_shape = tensor_infer_shape_at(alloc, item, depth + 1)?;
                if s_shape.len() != sub_shape.len() {
                    return Err(LanaError::InvalidParameters);
                }
                for d in 0..sub_shape.len() {
                    if s_shape[d] != sub_shape[d] {
                        return Err(LanaError::InvalidParameters);
                    }
                }
            }
            if alloc((sub_shape.len() + 1) * std::mem::size_of::<usize>()) != LanaError::Ok {
                return Err(LanaError::Oom);
            }
            let mut shape = Vec::with_capacity(sub_shape.len() + 1);
            shape.push(array.items.len());
            shape.extend_from_slice(&sub_shape);
            Ok(shape)
        }
        _ => Err(LanaError::Type),
    }
}

/// Recursively fill a tensor's data buffer in row-major order from a nested
/// array of numbers, mirroring `tensor_fill_data`. Writes through
/// `tensor_set_real`, which converts to the storage dtype (rounding f16/bf16).
pub fn tensor_fill_data(v: &Value, t: &mut Tensor, offset: &mut usize) -> Result<(), LanaError> {
    match &v.kind {
        ValueKind::Number(n) => {
            tensor_set_real(t, *offset, *n);
            *offset += 1;
            Ok(())
        }
        ValueKind::Array(array) => {
            let array = array.lock().unwrap();
            for item in &array.items {
                tensor_fill_data(item, t, offset)?;
            }
            Ok(())
        }
        _ => Err(LanaError::Type),
    }
}

/// Recursively fill a complex tensor's interleaved `[re, im]` buffer from two
/// parallel nested arrays, mirroring `tensor_fill_complex`. Shape mismatch
/// raises `InvalidParameters`.
pub fn tensor_fill_complex(
    re: &Value,
    im: &Value,
    t: &mut Tensor,
    offset: &mut usize,
) -> Result<(), LanaError> {
    match (&re.kind, &im.kind) {
        (ValueKind::Number(re_n), ValueKind::Number(im_n)) => {
            tensor_set_real(t, *offset, *re_n);
            tensor_set_imag(t, *offset, *im_n);
            *offset += 1;
            Ok(())
        }
        (ValueKind::Array(ra), ValueKind::Array(ia)) => {
            // Components may alias. Never hold two array locks, or keep a
            // parent's lock while recursively reading a component.
            let len = ra.lock().unwrap().items.len();
            if len != ia.lock().unwrap().items.len() {
                return Err(LanaError::InvalidParameters);
            }
            for i in 0..len {
                let re = ra.lock().unwrap().items[i].clone();
                let im = ia.lock().unwrap().items[i].clone();
                tensor_fill_complex(&re, &im, t, offset)?;
            }
            Ok(())
        }
        _ => Err(LanaError::Type),
    }
}

/// LIP-004 indexing: resolve one integer position against a dimension of
/// length `dim`. Negative counts from the end; a non-integer or non-finite
/// number is `Type`, an adjusted position outside `[0, dim)` is `Key`.
fn tensor_resolve_index(n: f64, dim: usize) -> Result<usize, LanaError> {
    if !n.is_finite() || n.floor() != n { return Err(LanaError::Type); }
    if n < -(dim as f64) || n >= dim as f64 { return Err(LanaError::Key); }
    Ok((if n < 0.0 { n + dim as f64 } else { n }) as usize)
}

/// LIP-004 slicing: resolve one slice bound against a dimension of length
/// `dim`. Negative counts from the end, then clamps into `[0, dim]`; slice
/// bounds never range-error. Non-integer numbers are `Type`.
fn tensor_resolve_slice_bound(n: f64, dim: usize) -> Result<usize, LanaError> {
    if !n.is_finite() || n.floor() != n { return Err(LanaError::Type); }
    let adjusted = if n < 0.0 { n + dim as f64 } else { n };
    Ok(adjusted.clamp(0.0, dim as f64) as usize)
}

/// LIP-004 indexing and slicing: build a view (or scalar) of `t` from a spec,
/// mirroring `tensor_index` in `vm/c/vm.c`. The spec is a single number (one
/// integer position) or an array of positions, each a number (integer) or a
/// two-element array `[start, end]` (slice). Fewer positions than the rank
/// keep the remaining trailing axes whole. Integer positions drop their axis;
/// slices keep it. Every non-fully-integer result shares the source buffer
/// (through `Arc`) as a view.
pub fn tensor_index(
    alloc: &mut dyn FnMut(usize) -> LanaError, t: &Tensor, spec: &Value,
) -> Result<Value, LanaError> {
    let mut is_int = [false; TENSOR_MAX_RANK];
    let mut start = [0usize; TENSOR_MAX_RANK];
    let mut count = [0usize; TENSOR_MAX_RANK];
    let positions: usize;
    match &spec.kind {
        ValueKind::Number(n) => {
            if t.ndim == 0 { return Err(LanaError::InvalidParameters); }
            start[0] = tensor_resolve_index(*n, t.shape[0])?;
            is_int[0] = true;
            count[0] = 1;
            positions = 1;
        }
        ValueKind::Array(array) => {
            let array = array.lock().unwrap();
            if array.items.len() > t.ndim { return Err(LanaError::InvalidParameters); }
            if array.items.len() > TENSOR_MAX_RANK { return Err(LanaError::InvalidParameters); }
            for (i, item) in array.items.iter().enumerate() {
                match &item.kind {
                    ValueKind::Number(n) => {
                        start[i] = tensor_resolve_index(*n, t.shape[i])?;
                        is_int[i] = true;
                        count[i] = 1;
                    }
                    ValueKind::Array(pair) => {
                        let pair = pair.lock().unwrap();
                        if pair.items.len() != 2
                            || !matches!(pair.items[0].kind, ValueKind::Number(_))
                            || !matches!(pair.items[1].kind, ValueKind::Number(_))
                        {
                            return Err(LanaError::Type);
                        }
                        let ValueKind::Number(s) = &pair.items[0].kind else {
                            unreachable!()
                        };
                        let ValueKind::Number(e) = &pair.items[1].kind else {
                            unreachable!()
                        };
                        let from = tensor_resolve_slice_bound(*s, t.shape[i])?;
                        let to = tensor_resolve_slice_bound(*e, t.shape[i])?;
                        is_int[i] = false;
                        start[i] = from;
                        count[i] = to.saturating_sub(from);
                    }
                    _ => return Err(LanaError::Type),
                }
            }
            positions = array.items.len();
        }
        _ => return Err(LanaError::Type),
    }

    // Fold every axis into the view: integer axes contribute their offset and
    // drop out; slice axes keep their (clamped) extent; trailing axes beyond
    // the position list are full slices.
    let mut all_int = true;
    let mut view_offset = t.offset;
    let mut view_shape = [0usize; TENSOR_MAX_RANK];
    let mut view_strides = [0usize; TENSOR_MAX_RANK];
    let mut view_ndim = 0usize;
    for i in 0..t.ndim {
        let integer = i < positions && is_int[i];
        let axis_start = if i < positions { start[i] } else { 0 };
        let axis_count = if i < positions { count[i] } else { t.shape[i] };
        view_offset += axis_start * t.strides[i];
        if integer { continue; }
        all_int = false;
        view_shape[view_ndim] = axis_count;
        view_strides[view_ndim] = t.strides[i];
        view_ndim += 1;
    }

    if all_int {
        // A full set of integer positions selects one element: a number, or
        // the established rank-zero complex tensor for complex tensors.
        if !t.is_complex {
            return Ok(Value::number(tensor_get_real(&t, view_offset)));
        }
        let mut r = tensor_new(alloc, 0, &[], true)?;
        tensor_set_real(&mut r, 0, tensor_get_real(&t, view_offset));
        tensor_set_imag(&mut r, 0, tensor_get_imag(&t, view_offset));
        return Ok(Value::tensor(Arc::new(r)));
    }

    if alloc(std::mem::size_of::<Tensor>()) != LanaError::Ok {
        return Err(LanaError::Oom);
    }
    if view_ndim > 0 {
        if alloc(view_ndim * std::mem::size_of::<usize>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
        if alloc(view_ndim * std::mem::size_of::<usize>()) != LanaError::Ok {
            return Err(LanaError::Oom);
        }
    }
    Ok(Value::tensor(Arc::new(Tensor {
        ndim: view_ndim,
        shape: view_shape[..view_ndim].to_vec(),
        strides: view_strides[..view_ndim].to_vec(),
        is_complex: t.is_complex,
        dtype: t.dtype,
        data: Arc::clone(&t.data),
        offset: view_offset,
        is_state: t.is_state,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_alloc(_bytes: usize) -> LanaError {
        LanaError::Ok
    }

    fn array(items: Vec<Value>) -> Value {
        Value::array(Arc::new(std::sync::Mutex::new(crate::value::Array { items })))
    }

    #[test]
    fn memory_accounting_uses_element_width() {
        // LIP-027 B18: a tensor's buffer accounts the actual element width, so
        // an f16 buffer is a quarter of an f64 buffer. The data-buffer
        // allocation dominates the struct/shape overhead.
        let mut f16_bytes = 0usize;
        let mut alloc_f16 = |bytes: usize| {
            f16_bytes = f16_bytes.max(bytes);
            LanaError::Ok
        };
        let t16 = tensor_new_dtype(&mut alloc_f16, 1, &[100000], TensorDtype::F16).unwrap();
        assert_eq!(t16.dtype, TensorDtype::F16);

        let mut f64_bytes = 0usize;
        let mut alloc_f64 = |bytes: usize| {
            f64_bytes = f64_bytes.max(bytes);
            LanaError::Ok
        };
        let t64 = tensor_new_dtype(&mut alloc_f64, 1, &[100000], TensorDtype::F64).unwrap();
        assert_eq!(t64.dtype, TensorDtype::F64);

        // f16 = 100000 * 2 = 200000 bytes; f64 = 100000 * 8 = 800000 bytes.
        assert!(f16_bytes >= 200000, "f16 buffer too small: {f16_bytes}");
        assert!(f64_bytes >= 800000, "f64 buffer too small: {f64_bytes}");
        assert!(f64_bytes >= f16_bytes * 3, "f64 not ~4x f16: {f64_bytes} vs {f16_bytes}");
    }

    #[test]
    fn indexing_selects_positions_slices_and_views() {
        let t = tensor(&[2, 3], &[1., 2., 3., 4., 5., 6.]);

        // One integer position selects a row: a rank-1 view sharing the buffer.
        let row = tensor_index(&mut ok_alloc, &t, &Value::number(1.)).unwrap();
        let ValueKind::Tensor(r) = &row.kind else { panic!("expected tensor") };
        assert_eq!(r.shape, [3]); assert_eq!(r.strides, [1]); assert_eq!(r.offset, 3);
        assert_eq!(tensor_f64(r), [1., 2., 3.]); // logical elements of the row view
        assert!(Arc::ptr_eq(&r.data, &t.data));
        let neg = tensor_index(&mut ok_alloc, &t, &Value::number(-1.)).unwrap();
        let ValueKind::Tensor(r) = &neg.kind else { panic!("expected tensor") };
        assert_eq!(r.shape.as_slice(), [3].as_slice());
        assert_eq!(r.offset, 3);

        // Out-of-range integer positions are Key errors (after negative wrap).
        for n in [2., -3., 1e100] {
            assert!(matches!(tensor_index(&mut ok_alloc, &t, &Value::number(n)),
                             Err(LanaError::Key)));
        }

        // Non-number, non-integer, and non-finite positions are Type errors.
        for n in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.5, -0.5] {
            assert!(matches!(tensor_index(&mut ok_alloc, &t, &Value::number(n)),
                             Err(LanaError::Type)));
        }
        assert!(matches!(tensor_index(&mut ok_alloc, &t, &Value::boolean(true)),
                         Err(LanaError::Type)));

        // A full set of integer positions selects one element as a number.
        let full = array(vec![Value::number(1.), Value::number(2.)]);
        assert!(matches!(tensor_index(&mut ok_alloc, &t, &full).unwrap().kind,
                         ValueKind::Number(6.0)));
        let wrapped = array(vec![Value::number(-1.), Value::number(-3.)]);
        assert!(matches!(tensor_index(&mut ok_alloc, &t, &wrapped).unwrap().kind,
                         ValueKind::Number(4.0)));

        // More positions than the rank is InvalidParameters.
        let extra = array(vec![Value::number(0.), Value::number(0.), Value::number(0.)]);
        assert!(matches!(tensor_index(&mut ok_alloc, &t, &extra),
                         Err(LanaError::InvalidParameters)));

        // Slices keep their axis and may be non-contiguous: t[0:2, 1].
        let col_spec = array(vec![
            array(vec![Value::number(0.), Value::number(2.)]),
            Value::number(1.),
        ]);
        let col_value = tensor_index(&mut ok_alloc, &t, &col_spec).unwrap();
        let ValueKind::Tensor(col) = &col_value.kind else { panic!("expected tensor") };
        assert_eq!(col.shape, [2]); assert_eq!(col.strides, [3]); assert_eq!(col.offset, 1);
        assert!(Arc::ptr_eq(&col.data, &t.data));

        // Views chain: selecting the column view again still reads through
        // offset and strides.
        assert!(matches!(tensor_index(&mut ok_alloc, col, &Value::number(1.)).unwrap().kind,
                         ValueKind::Number(5.0)));
        assert!(matches!(tensor_index(&mut ok_alloc, col, &Value::number(-1.)).unwrap().kind,
                         ValueKind::Number(5.0)));
        assert!(matches!(tensor_index(&mut ok_alloc, col, &Value::number(2.)),
                         Err(LanaError::Key)));

        // Strided traversal: arithmetic and full reduction over the view
        // follow its strides rather than assuming a contiguous fiber.
        let doubled = tensor_elementwise(&mut ok_alloc, col, col, 0).unwrap();
        assert_eq!(tensor_f64(&doubled), [4.0, 10.0]);
        assert!(matches!(tensor_reduce(&mut ok_alloc, col, 0).unwrap().kind,
                         ValueKind::Number(7.0)));

        // Fewer positions than the rank keep trailing axes whole; slice bounds
        // clamp and never range-error; start > end is an empty axis.
        let whole = tensor_index(&mut ok_alloc, &t, &array(vec![
            array(vec![Value::number(-10.), Value::number(10.)]),
        ])).unwrap();
        let ValueKind::Tensor(r) = &whole.kind else { panic!("expected tensor") };
        assert_eq!(r.shape, [2, 3]); assert_eq!(r.offset, 0);
        let empty = tensor_index(&mut ok_alloc, &t, &array(vec![
            array(vec![Value::number(5.), Value::number(10.)]),
        ])).unwrap();
        let ValueKind::Tensor(r) = &empty.kind else { panic!("expected tensor") };
        assert_eq!(r.shape, [0, 3]);
        let back = tensor_index(&mut ok_alloc, &t, &array(vec![
            array(vec![Value::number(1.), Value::number(0.)]),
        ])).unwrap();
        let ValueKind::Tensor(r) = &back.kind else { panic!("expected tensor") };
        assert_eq!(r.shape, [0, 3]);

        // Malformed specs are Type errors: a non-number position, a pair
        // holding a non-number bound, and a pair that is not two elements long.
        assert!(matches!(tensor_index(&mut ok_alloc, &t,
                         &array(vec![Value::boolean(true), Value::number(0.)])),
                         Err(LanaError::Type)));
        assert!(matches!(tensor_index(&mut ok_alloc, &t, &array(vec![
            array(vec![Value::number(0.), Value::boolean(true)]), Value::number(0.),
        ])), Err(LanaError::Type)));
        assert!(matches!(tensor_index(&mut ok_alloc, &t, &array(vec![
            array(vec![Value::number(0.), Value::number(0.), Value::number(0.)]),
        ])), Err(LanaError::Type)));

        // A position on a rank-zero tensor has no axis to select.
        let scalar = tensor_new(&mut ok_alloc, 0, &[], false).unwrap();
        assert!(matches!(tensor_index(&mut ok_alloc, &scalar, &Value::number(0.)),
                         Err(LanaError::InvalidParameters)));

        // A complex element selection yields the rank-zero complex form.
        let mut c = tensor_new(&mut ok_alloc, 1, &[1], true).unwrap();
        set_f64(&mut c, &[3., 4.]);
        let picked = tensor_index(&mut ok_alloc, &c, &Value::number(0.)).unwrap();
        let ValueKind::Tensor(r) = &picked.kind else { panic!("expected tensor") };
        assert_eq!(r.ndim, 0); assert!(r.is_complex); assert_eq!(tensor_f64(r), [3., 4.]);
    }

    #[test]
    fn axis_reductions_cover_shapes_values_and_errors() {
        let mut t = tensor_new(&mut ok_alloc, 2, &[2, 3], false).unwrap();
        set_f64(&mut t, &[1., 2., 3., 4., 5., 6.]);
        for (op, axis, shape, data) in [
            (0, 0., vec![3], vec![5., 7., 9.]),
            (1, 0., vec![3], vec![2.5, 3.5, 4.5]),
            (2, -1., vec![2], vec![3., 6.]),
            (3, -1., vec![2], vec![1., 4.]),
        ] {
            let result = tensor_reduce_axis(&mut ok_alloc, &t, op, &Value::number(axis)).unwrap();
            let ValueKind::Tensor(r) = result.kind else { panic!("expected tensor"); };
            assert_eq!(r.shape, shape); assert_eq!(tensor_f64(&r), data);
        }
        for axis in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.5, -3., 2., 1e100] {
            assert!(matches!(tensor_reduce_axis(&mut ok_alloc, &t, 0, &Value::number(axis)), Err(LanaError::InvalidParameters)));
        }
        assert!(matches!(tensor_reduce_axis(&mut ok_alloc, &t, 0, &Value::boolean(false)), Err(LanaError::Type)));
        let empty = tensor_new(&mut ok_alloc, 2, &[0, 3], false).unwrap();
        let result = tensor_reduce_axis(&mut ok_alloc, &empty, 0, &Value::number(0.)).unwrap();
        let ValueKind::Tensor(r) = result.kind else { panic!("expected tensor"); };
        assert_eq!(r.shape, [3]); assert_eq!(tensor_f64(&r), [0., 0., 0.]);
        for op in 1..4 {
            assert!(matches!(tensor_reduce_axis(&mut ok_alloc, &empty, op, &Value::number(0.)), Err(LanaError::InvalidParameters)));
            let result = tensor_reduce_axis(&mut ok_alloc, &empty, op, &Value::number(1.)).unwrap();
            let ValueKind::Tensor(r) = result.kind else { panic!("expected tensor"); };
            assert_eq!(r.shape, [0]); assert!(r.data.is_empty());
        }
        let scalar = tensor_new(&mut ok_alloc, 0, &[], false).unwrap();
        assert!(matches!(tensor_reduce_axis(&mut ok_alloc, &scalar, 0, &Value::number(0.)), Err(LanaError::InvalidParameters)));
        let mut complex = tensor_new(&mut ok_alloc, 1, &[2], true).unwrap();
        set_f64(&mut complex, &[1., 2., 3., 4.]);
        for (op, data) in [(0, vec![4., 6.]), (1, vec![2., 3.])] {
            let result = tensor_reduce_axis(&mut ok_alloc, &complex, op, &Value::number(-1.)).unwrap();
            let ValueKind::Tensor(r) = result.kind else { panic!("expected tensor"); };
            assert_eq!(r.ndim, 0); assert!(r.is_complex); assert_eq!(tensor_f64(&r), data);
        }
        assert!(matches!(tensor_reduce_axis(&mut ok_alloc, &complex, 2, &Value::number(0.)), Err(LanaError::Type)));
        tensor_set_imag(&mut complex, 0, f64::INFINITY);
        assert!(matches!(tensor_reduce_axis(&mut ok_alloc, &complex, 0, &Value::number(0.)), Err(LanaError::InvalidParameters)));
    }

    #[test]
    fn construction_rejects_invalid_dimensions_rank_and_cycles() {
        for n in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.5,
                  18446744073709551616.0] {
            assert_eq!(tensor_dimension(n), Err(LanaError::InvalidParameters));
            assert_eq!(tensor_shape_from_array(&mut ok_alloc, &array(vec![Value::number(n)])),
                       Err(LanaError::InvalidParameters));
        }
        assert!(tensor_new(&mut ok_alloc, 32, &[1; 32], false).is_ok());
        assert!(matches!(tensor_new(&mut ok_alloc, 33, &[1; 33], false),
                         Err(LanaError::InvalidParameters)));
        assert!(matches!(tensor_new(&mut ok_alloc, 1, &[], false),
                         Err(LanaError::InvalidParameters)));
        assert!(matches!(tensor_new(&mut ok_alloc, 1, &[usize::MAX / 8 + 1], false),
                         Err(LanaError::Oom)));
        assert!(matches!(tensor_new(&mut ok_alloc, 1, &[usize::MAX / 8 + 1], true),
                         Err(LanaError::Oom)));
        let cyclic = array(vec![]);
        let ValueKind::Array(a) = &cyclic.kind else { unreachable!() };
        a.lock().unwrap().items.push(cyclic.clone());
        assert!(!cyclic.is_unresolved());
        let result = tensor_infer_shape(&mut ok_alloc, &cyclic);
        a.lock().unwrap().items.clear();
        assert_eq!(result, Err(LanaError::InvalidParameters));
        let mut nested = Value::number(1.0);
        for _ in 0..32 { nested = array(vec![nested]); }
        assert_eq!(tensor_infer_shape(&mut ok_alloc, &nested).unwrap().len(), 32);
        assert_eq!(tensor_infer_shape(&mut ok_alloc, &array(vec![nested])), Err(LanaError::InvalidParameters));
    }

    #[test]
    fn complex_components_may_share_arrays() {
        let a = array(vec![array(vec![Value::number(1.0), Value::number(2.0)])]);
        let mut t = tensor_new(&mut ok_alloc, 1, &[2], true).unwrap();
        tensor_fill_complex(&a, &a, &mut t, &mut 0).unwrap();
        assert_eq!(tensor_get_real(&t, 0), 1.0);
        assert_eq!(tensor_get_imag(&t, 0), 1.0);
        assert_eq!(tensor_get_real(&t, 1), 2.0);
        assert_eq!(tensor_get_imag(&t, 1), 2.0);
    }

    #[test]
    fn reductions_reject_empty_domains_and_nonfinite_results() {
        let empty = tensor(&[0], &[]);
        assert!(matches!(tensor_reduce(&mut ok_alloc, &empty, 0).unwrap().kind,
                         ValueKind::Number(0.0)));
        for op in 1..4 {
            assert!(matches!(tensor_reduce(&mut ok_alloc, &empty, op),
                             Err(LanaError::InvalidParameters)));
        }
        let overflow = tensor(&[2], &[f64::MAX, f64::MAX]);
        assert!(matches!(tensor_reduce(&mut ok_alloc, &overflow, 0),
                         Err(LanaError::InvalidParameters)));
        let mut complex = tensor_new(&mut ok_alloc, 1, &[2], true).unwrap();
        set_f64(&mut complex, &[1.0, f64::INFINITY, 2.0, 0.0]);
        assert!(matches!(tensor_reduce(&mut ok_alloc, &complex, 0),
                         Err(LanaError::InvalidParameters)));
        set_f64(&mut complex, &[f64::MAX, 0.0, f64::MAX, 0.0]);
        assert!(matches!(tensor_reduce(&mut ok_alloc, &complex, 0),
                         Err(LanaError::InvalidParameters)));
    }

    /// Read a tensor's elements back as f64s: the real parts for a real tensor,
    /// or the interleaved [re, im] pairs for a complex tensor.
    fn tensor_f64(t: &Tensor) -> Vec<f64> {
        let count = tensor_element_count(t);
        if t.is_complex {
            let mut out = Vec::with_capacity(count * 2);
            for i in 0..count {
                out.push(tensor_get_real(t, i));
                out.push(tensor_get_imag(t, i));
            }
            out
        } else {
            (0..count).map(|i| tensor_get_real(t, i)).collect()
        }
    }

    /// Write f64s into a tensor: the real parts for a real tensor, or the
    /// interleaved [re, im] pairs for a complex tensor.
    fn set_f64(t: &mut Tensor, data: &[f64]) {
        if t.is_complex {
            for (i, pair) in data.chunks(2).enumerate() {
                tensor_set_real(t, i, pair[0]);
                tensor_set_imag(t, i, pair[1]);
            }
        } else {
            for (i, v) in data.iter().enumerate() {
                tensor_set_real(t, i, *v);
            }
        }
    }

    fn tensor(shape: &[usize], data: &[f64]) -> Tensor {
        let mut t = tensor_new(&mut ok_alloc, shape.len(), shape, false).unwrap();
        set_f64(&mut t, data);
        t
    }

    #[test]
    fn new_computes_row_major_strides() {
        let t = tensor_new(&mut ok_alloc, 2, &[2, 3], false).unwrap();
        assert_eq!(t.ndim, 2);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.strides, vec![3, 1]);
        assert_eq!(t.data.len(), 48);
        assert!(!t.is_complex);
    }

    #[test]
    fn new_zero_dim_is_scalar() {
        let t = tensor_new(&mut ok_alloc, 0, &[], false).unwrap();
        assert_eq!(t.ndim, 0);
        assert_eq!(t.data.len(), 8);
    }

    #[test]
    fn elementwise_add_broadcasts() {
        let a = tensor(&[2, 1], &[1.0, 2.0]);
        let b = tensor(&[1, 3], &[10.0, 20.0, 30.0]);
        let r = tensor_elementwise(&mut ok_alloc, &a, &b, 0).unwrap();
        assert_eq!(r.shape, vec![2, 3]);
        assert_eq!(tensor_f64(&r), vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
    }

    #[test]
    fn elementwise_div_by_zero_is_invalid() {
        let a = tensor(&[1], &[1.0]);
        let b = tensor(&[1], &[0.0]);
        assert!(matches!(
            tensor_elementwise(&mut ok_alloc, &a, &b, 3),
            Err(LanaError::InvalidParameters)
        ));
    }

    #[test]
    fn elementwise_mismatched_complex_is_type_error() {
        let a = tensor(&[1], &[1.0]);
        let mut b = tensor(&[1], &[1.0]);
        b.is_complex = true;
        b.dtype = TensorDtype::Complex;
        assert!(matches!(
            tensor_elementwise(&mut ok_alloc, &a, &b, 0),
            Err(LanaError::Type)
        ));
    }

    #[test]
    fn matmul_1d_dot_product() {
        let a = tensor(&[3], &[1.0, 2.0, 3.0]);
        let b = tensor(&[3], &[4.0, 5.0, 6.0]);
        let r = tensor_matmul(&mut ok_alloc, &a, &b, TensorDtype::F64).unwrap();
        assert_eq!(r.ndim, 0);
        assert_eq!(tensor_f64(&r), vec![32.0]);
    }

    #[test]
    fn matmul_2d() {
        let a = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
        let r = tensor_matmul(&mut ok_alloc, &a, &b, TensorDtype::F64).unwrap();
        assert_eq!(r.shape, vec![2, 2]);
        assert_eq!(tensor_f64(&r), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_incompatible_is_invalid() {
        let a = tensor(&[2, 3], &[1.0; 6]);
        let b = tensor(&[2, 2], &[1.0; 4]);
        assert!(matches!(
            tensor_matmul(&mut ok_alloc, &a, &b, TensorDtype::F64),
            Err(LanaError::InvalidParameters)
        ));
    }

    /// A view over `base` sharing its buffer: custom shape/strides/offset.
    fn view(base: &Tensor, shape: &[usize], strides: &[usize], offset: usize) -> Tensor {
        Tensor {
            ndim: shape.len(),
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            is_complex: base.is_complex,
            dtype: base.dtype,
            data: Arc::clone(&base.data),
            offset,
            is_state: base.is_state,
        }
    }

    #[test]
    fn matmul_vector_matrix_promotions() {
        // 1d x 2d promotes the vector to a row: [1,2,3] . [[1,2],[3,4],[5,6]].
        let a = tensor(&[3], &[1.0, 2.0, 3.0]);
        let b = tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = tensor_matmul(&mut ok_alloc, &a, &b, TensorDtype::F64).unwrap();
        assert_eq!(r.shape, vec![2]);
        assert_eq!(tensor_f64(&r), vec![22.0, 28.0]);

        // 2d x 1d promotes the vector to a column: [[1,2,3],[4,5,6]] . [1,2,3].
        let m = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = tensor_matmul(&mut ok_alloc, &m, &a, TensorDtype::F64).unwrap();
        assert_eq!(r.shape, vec![2]);
        assert_eq!(tensor_f64(&r), vec![14.0, 32.0]);
    }

    #[test]
    fn matmul_batched_broadcast() {
        // a[2,2,2] . b[1,2,2] -> [2,2,2]; b's batch dim 1 broadcasts to 2.
        let a = tensor(&[2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = tensor(&[1, 2, 2], &[1.0, 0.0, 0.0, 1.0]); // identity
        let r = tensor_matmul(&mut ok_alloc, &a, &b, TensorDtype::F64).unwrap();
        assert_eq!(r.shape, vec![2, 2, 2]);
        assert_eq!(tensor_f64(&r), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn matmul_complex_zgemm() {
        // (iI) . (iI) = -I, interleaved [re, im] per element.
        let mut a = tensor_new(&mut ok_alloc, 2, &[2, 2], true).unwrap();
        set_f64(&mut a, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let mut b = tensor_new(&mut ok_alloc, 2, &[2, 2], true).unwrap();
        set_f64(&mut b, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let r = tensor_matmul(&mut ok_alloc, &a, &b, TensorDtype::Complex).unwrap();
        assert!(r.is_complex);
        assert_eq!(tensor_f64(&r), vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]);
    }

    #[test]
    fn matmul_direct_strided_view() {
        // Columns 0:3 of a 2x4 base keep column stride 1 but row stride 4, so
        // lda = 4 > K = 3 and BLAS reads the true row stride.
        let base = tensor(&[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let a = view(&base, &[2, 3], &[4, 1], 0);
        let b = tensor(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let r = tensor_matmul(&mut ok_alloc, &a, &b, TensorDtype::F64).unwrap();
        assert_eq!(r.shape, vec![2, 2]);
        assert_eq!(tensor_f64(&r), vec![4.0, 5.0, 12.0, 13.0]);
    }

    #[test]
    fn matmul_packed_noncontiguous_core() {
        // Columns 0:4:2 of a 2x4 base have column stride 2, so the operand is
        // gathered into scratch before the backend call.
        let base = tensor(&[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let a = view(&base, &[2, 2], &[4, 2], 0);
        let b = tensor(&[2, 2], &[1.0, 1.0, 0.0, 1.0]);
        let r = tensor_matmul(&mut ok_alloc, &a, &b, TensorDtype::F64).unwrap();
        assert_eq!(r.shape, vec![2, 2]);
        assert_eq!(tensor_f64(&r), vec![1.0, 4.0, 5.0, 12.0]);
    }

    #[test]
    fn matmul_empty_contraction() {
        // [2,0] . [0,3] -> [2,3] of zeros (the sum over an empty contraction).
        let a = tensor(&[2, 0], &[]);
        let b = tensor(&[0, 3], &[]);
        let r = tensor_matmul(&mut ok_alloc, &a, &b, TensorDtype::F64).unwrap();
        assert_eq!(r.shape, vec![2, 3]);
        assert_eq!(tensor_f64(&r), vec![0.0; 6]);
    }

    #[test]
    fn matmul_mismatched_complex_is_type_error() {
        let mut a = tensor_new(&mut ok_alloc, 2, &[2, 2], true).unwrap();
        set_f64(&mut a, &[0.0; 8]);
        let b = tensor(&[2, 2], &[1.0; 4]);
        assert!(matches!(
            tensor_matmul(&mut ok_alloc, &a, &b, TensorDtype::F64),
            Err(LanaError::Type)
        ));
    }

    #[test]
    fn matmul_rank_zero_is_invalid() {
        let a = tensor(&[], &[1.0]);
        let b = tensor(&[2, 2], &[1.0; 4]);
        assert!(matches!(
            tensor_matmul(&mut ok_alloc, &a, &b, TensorDtype::F64),
            Err(LanaError::InvalidParameters)
        ));
    }

    #[test]
    fn matmul_incompatible_batch_is_invalid() {
        let a = tensor(&[3, 2, 2], &[1.0; 12]);
        let b = tensor(&[2, 2, 2], &[1.0; 8]);
        assert!(matches!(
            tensor_matmul(&mut ok_alloc, &a, &b, TensorDtype::F64),
            Err(LanaError::InvalidParameters)
        ));
    }

    #[test]
    fn reduce_sum_mean_max_min() {
        let t = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let sum = tensor_reduce(&mut ok_alloc, &t, 0).unwrap();
        let mean = tensor_reduce(&mut ok_alloc, &t, 1).unwrap();
        let max = tensor_reduce(&mut ok_alloc, &t, 2).unwrap();
        let min = tensor_reduce(&mut ok_alloc, &t, 3).unwrap();
        assert!(matches!(sum.kind, ValueKind::Number(n) if n == 10.0));
        assert!(matches!(mean.kind, ValueKind::Number(n) if n == 2.5));
        assert!(matches!(max.kind, ValueKind::Number(n) if n == 4.0));
        assert!(matches!(min.kind, ValueKind::Number(n) if n == 1.0));
    }

    #[test]
    fn reduce_complex_max_is_type_error() {
        let mut t = tensor(&[2], &[1.0, 2.0]);
        t.is_complex = true;
        assert!(matches!(
            tensor_reduce(&mut ok_alloc, &t, 2),
            Err(LanaError::Type)
        ));
    }

    #[test]
    fn infer_shape_nested() {
        let v = Value::array(Arc::new(std::sync::Mutex::new(crate::value::Array {
            items: vec![
                Value::array(Arc::new(std::sync::Mutex::new(crate::value::Array {
                    items: vec![Value::number(1.0), Value::number(2.0)],
                }))),
                Value::array(Arc::new(std::sync::Mutex::new(crate::value::Array {
                    items: vec![Value::number(3.0), Value::number(4.0)],
                }))),
            ],
        })));
        assert_eq!(tensor_infer_shape(&mut ok_alloc, &v).unwrap(), vec![2, 2]);
    }

    #[test]
    fn infer_shape_ragged_is_invalid() {
        let v = Value::array(Arc::new(std::sync::Mutex::new(crate::value::Array {
            items: vec![
                Value::array(Arc::new(std::sync::Mutex::new(crate::value::Array {
                    items: vec![Value::number(1.0), Value::number(2.0)],
                }))),
                Value::array(Arc::new(std::sync::Mutex::new(crate::value::Array {
                    items: vec![Value::number(3.0)],
                }))),
            ],
        })));
        assert_eq!(tensor_infer_shape(&mut ok_alloc, &v), Err(LanaError::InvalidParameters));
    }

    #[test]
    fn shape_from_array_rejects_non_number() {
        let v = Value::array(Arc::new(std::sync::Mutex::new(crate::value::Array {
            items: vec![Value::string(Arc::from("x"))],
        })));
        assert_eq!(tensor_shape_from_array(&mut ok_alloc, &v), Err(LanaError::Type));
    }

    #[test]
    fn shape_from_array_rejects_fractional() {
        let v = Value::array(Arc::new(std::sync::Mutex::new(crate::value::Array {
            items: vec![Value::number(2.5)],
        })));
        assert_eq!(
            tensor_shape_from_array(&mut ok_alloc, &v),
            Err(LanaError::InvalidParameters)
        );
    }

    #[test]
    fn gpu_matmul_2d_within_float32_tolerance() {
        let a = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
        let r = match tensor_gpu_matmul(&mut ok_alloc, &a, &b) {
            Ok(r) => r,
            // No Metal device on this host; the GPU path is optional.
            Err(LanaError::UnsupportedOperation) => return,
            Err(e) => panic!("unexpected error: {e:?}"),
        };
        assert_eq!(r.shape, vec![2, 2]);
        let expected = [19.0, 22.0, 43.0, 50.0];
        for (got, want) in tensor_f64(&r).iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5 * (1.0 + want.abs()));
        }
    }

    #[test]
    fn gpu_matmul_rejects_complex() {
        let a = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let mut ca = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        ca.is_complex = true;
        assert!(matches!(
            tensor_gpu_matmul(&mut ok_alloc, &ca, &a),
            Err(LanaError::Type)
        ));
    }

    fn uncertain(pred: &Tensor, var: &Tensor) -> Value {
        let mut map = Map::new(2);
        map.set(Arc::from("prediction"), Value::tensor(Arc::new(pred.clone())), false).unwrap();
        map.set(Arc::from("uncertainty"), Value::tensor(Arc::new(var.clone())), false).unwrap();
        Value::map(Arc::new(Mutex::new(map)))
    }

    fn unpack_uncertain(v: &Value) -> (Tensor, Tensor) {
        let (pred, var, unc) = tensor_uncertainty_unpack(v).unwrap();
        assert!(unc);
        ((*pred).clone(), (*var.unwrap()).clone())
    }

    #[test]
    fn uncertainty_elementwise_add_sub() {
        let a = tensor(&[2], &[1.0, 2.0]);
        let va = tensor(&[2], &[0.5, 0.25]);
        let b = tensor(&[2], &[3.0, 4.0]);
        let vb = tensor(&[2], &[0.125, 0.0625]);

        let (pred, var) = unpack_uncertain(
            &tensor_elementwise_uncertain(&mut ok_alloc, &a, &va, &b, &vb, 0).unwrap());
        assert_eq!(tensor_f64(&pred), [4.0, 6.0]);
        assert_eq!(tensor_f64(&var), [0.625, 0.3125]);

        let (pred, var) = unpack_uncertain(
            &tensor_elementwise_uncertain(&mut ok_alloc, &a, &va, &b, &vb, 1).unwrap());
        assert_eq!(tensor_f64(&pred), [-2.0, -2.0]);
        assert_eq!(tensor_f64(&var), [0.625, 0.3125]);
    }

    #[test]
    fn uncertainty_elementwise_mul_div() {
        let a = tensor(&[2], &[1.0, 2.0]);
        let va = tensor(&[2], &[0.5, 0.25]);
        let b = tensor(&[2], &[3.0, 4.0]);
        let vb = tensor(&[2], &[0.125, 0.0625]);

        let (pred, var) = unpack_uncertain(
            &tensor_elementwise_uncertain(&mut ok_alloc, &a, &va, &b, &vb, 2).unwrap());
        assert_eq!(tensor_f64(&pred), [3.0, 8.0]);
        assert_eq!(tensor_f64(&var), [4.625, 4.25]);

        let d = tensor(&[2], &[1.0, 1.0]);
        let vd = tensor(&[2], &[1.0, 1.0]);
        let e = tensor(&[2], &[2.0, 2.0]);
        let ve = tensor(&[2], &[1.0, 1.0]);
        let (pred, var) = unpack_uncertain(
            &tensor_elementwise_uncertain(&mut ok_alloc, &d, &vd, &e, &ve, 3).unwrap());
        assert_eq!(tensor_f64(&pred), [0.5, 0.5]);
        assert_eq!(tensor_f64(&var), [0.3125, 0.3125]);
    }

    #[test]
    fn uncertainty_matmul_dot() {
        let a = tensor(&[2], &[1.0, 2.0]);
        let va = tensor(&[2], &[0.5, 0.25]);
        let b = tensor(&[2], &[3.0, 4.0]);
        let vb = tensor(&[2], &[0.125, 0.0625]);
        let (pred, var) = unpack_uncertain(
            &tensor_matmul_uncertain(&mut ok_alloc, &a, &va, &b, &vb).unwrap());
        assert_eq!(tensor_f64(&pred), [11.0]);
        assert_eq!(tensor_f64(&var), [8.875]);
    }

    #[test]
    fn uncertainty_reduce_sum_mean() {
        let s = tensor(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let vs = tensor(&[4], &[0.5, 0.25, 0.125, 0.0625]);

        let (pred, var) = unpack_uncertain(
            &tensor_reduce_uncertain(&mut ok_alloc, &s, &vs, 0, None).unwrap());
        assert_eq!(tensor_f64(&pred), [10.0]);
        assert_eq!(tensor_f64(&var), [0.9375]);

        let (pred, var) = unpack_uncertain(
            &tensor_reduce_uncertain(&mut ok_alloc, &s, &vs, 1, None).unwrap());
        assert_eq!(tensor_f64(&pred), [2.5]);
        assert_eq!(tensor_f64(&var), [0.05859375]);

        // Axis reduction over a 2x2 tensor.
        let m = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let vm = tensor(&[2, 2], &[0.5, 0.25, 0.125, 0.0625]);
        let axis = Value::number(0.0);
        let (pred, var) = unpack_uncertain(
            &tensor_reduce_uncertain(&mut ok_alloc, &m, &vm, 0, Some(&axis)).unwrap());
        assert_eq!(tensor_f64(&pred), [4.0, 6.0]);
        assert_eq!(tensor_f64(&var), [0.625, 0.3125]);

        let (pred, var) = unpack_uncertain(
            &tensor_reduce_uncertain(&mut ok_alloc, &m, &vm, 1, Some(&axis)).unwrap());
        assert_eq!(tensor_f64(&pred), [2.0, 3.0]);
        assert_eq!(tensor_f64(&var), [0.15625, 0.078125]);
    }

    #[test]
    fn uncertainty_rejects_complex_and_nonfinite() {
        let a = tensor(&[2], &[1.0, 2.0]);
        let va = tensor(&[2], &[0.5, 0.25]);
        let b = tensor(&[2], &[3.0, 4.0]);
        let vb = tensor(&[2], &[0.125, 0.0625]);

        let mut ca = tensor(&[2], &[1.0, 2.0]);
        ca.is_complex = true;
        assert!(matches!(
            tensor_elementwise_uncertain(&mut ok_alloc, &ca, &va, &b, &vb, 0),
            Err(LanaError::Type)
        ));

        let vinf = tensor(&[2], &[0.5, f64::INFINITY]);
        assert!(matches!(
            tensor_elementwise_uncertain(&mut ok_alloc, &a, &vinf, &b, &vb, 0),
            Err(LanaError::InvalidParameters)
        ));
    }

    #[test]
    fn uncertainty_unpack_rejects_malformed_maps() {
        let a = tensor(&[2], &[1.0, 2.0]);
        let va = tensor(&[2], &[0.5, 0.25]);

        // Wrong entry count.
        let mut map1 = Map::new(1);
        map1.set(Arc::from("prediction"), Value::tensor(Arc::new(a.clone())), false).unwrap();
        assert!(matches!(
            tensor_uncertainty_unpack(&Value::map(Arc::new(Mutex::new(map1)))),
            Err(LanaError::Type)
        ));

        // Wrong key.
        let mut map2 = Map::new(2);
        map2.set(Arc::from("prediction"), Value::tensor(Arc::new(a.clone())), false).unwrap();
        map2.set(Arc::from("variance"), Value::tensor(Arc::new(va.clone())), false).unwrap();
        assert!(matches!(
            tensor_uncertainty_unpack(&Value::map(Arc::new(Mutex::new(map2)))),
            Err(LanaError::Type)
        ));

        // Non-tensor value.
        let mut map3 = Map::new(2);
        map3.set(Arc::from("prediction"), Value::tensor(Arc::new(a.clone())), false).unwrap();
        map3.set(Arc::from("uncertainty"), Value::number(1.0), false).unwrap();
        assert!(matches!(
            tensor_uncertainty_unpack(&Value::map(Arc::new(Mutex::new(map3)))),
            Err(LanaError::Type)
        ));

        // A bare tensor is certain.
        let (pred, var, unc) = tensor_uncertainty_unpack(&Value::tensor(Arc::new(a.clone()))).unwrap();
        assert!(!unc);
        assert!(var.is_none());
        assert_eq!(tensor_f64(&pred), [1.0, 2.0]);
    }

    #[test]
    fn dtype_rounding_and_string_roundtrip() {
        // f16 round-to-nearest-even: 0.1 -> 0.0999755859375.
        assert_eq!(round_f16(0.1), 0.0999755859375);
        // bf16 round-to-nearest-even: 3.14159 -> 3.140625.
        assert_eq!(round_bf16(3.14159), 3.140625);
        // f16 overflow to inf (max finite f16 is 65504).
        assert_eq!(round_f16(65520.0), f64::INFINITY);
        // f16 halfway rounds to even: 0.5 is exact; 1.0 + 2^-11 rounds to 1.0.
        assert_eq!(round_f16(0.5), 0.5);
        assert_eq!(round_f16(1.0 + 2f64.powi(-11)), 1.0);
        // Non-finite and zero pass through unchanged.
        assert!(round_f16(f64::NAN).is_nan());
        assert_eq!(round_f16(0.0), 0.0);
        assert_eq!(round_bf16(0.0), 0.0);

        // TensorDtype string round-trip.
        for (dtype, name) in [
            (TensorDtype::F64, "f64"),
            (TensorDtype::F32, "f32"),
            (TensorDtype::F16, "f16"),
            (TensorDtype::Bf16, "bf16"),
            (TensorDtype::Complex, "complex"),
        ] {
            assert_eq!(dtype.as_str(), name);
            assert_eq!(TensorDtype::from_str(name), Some(dtype));
        }
        assert_eq!(TensorDtype::from_str("int8"), None);
    }
}
