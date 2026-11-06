//! Tensor host calls (LIP-004, LIP-027): construction, shape, elementwise,
//! reductions, matmul, and device transfer.

use super::*;

/// LIP-027: parse an optional trailing dtype string argument (the last of
/// `argc` arguments). Returns `None` for an invalid dtype string; callers must
/// already have checked argc is 1 or 2.
fn tensor_optional_dtype(arguments: &[Value], argc: usize) -> Option<TensorDtype> {
    if argc == 1 {
        return Some(TensorDtype::F64);
    }
    let ValueKind::String(s) = &arguments[1].kind else {
        return None;
    };
    TensorDtype::from_str(s)
}

impl Vm<'_> {
    pub(super) fn host_tensor(&mut self, host_id: u32, arguments: &[Value], out: &mut Value) -> LanaError {
        let argc = arguments.len();
        match host_id {
            LANA_HOST_TENSOR_ALLOC => {
                // Arguments: shape array (VAL_ARRAY), optional is_complex (VAL_BOOL).
                if !(1..=2).contains(&argc) {
                    return LanaError::Type;
                }
                let mut is_complex = false;
                if argc == 2 {
                    let ValueKind::Bool(b) = arguments[1].kind else {
                        return LanaError::Type;
                    };
                    is_complex = b;
                }
                let alloc = self.heap.clone();
                let shape = match tensor::tensor_shape_from_array(&alloc, &arguments[0]) {
                    Ok(shape) => shape,
                    Err(error) => return error,
                };
                let t = match tensor::tensor_new(&alloc, shape.len(), &shape, is_complex) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                *out = Value::tensor(Arc::new(t));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_ZEROS => {
                if argc != 1 && argc != 2 {
                    return LanaError::Type;
                }
                let dtype = match tensor_optional_dtype(arguments, argc) {
                    Some(d) => d,
                    None => return LanaError::InvalidParameters,
                };
                let alloc = self.heap.clone();
                let shape = match tensor::tensor_shape_from_array(&alloc, &arguments[0]) {
                    Ok(shape) => shape,
                    Err(error) => return error,
                };
                let mut t = match tensor::tensor_new(&alloc, shape.len(), &shape, false) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                t.dtype = dtype;
                *out = Value::tensor(Arc::new(t));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_ONES => {
                if argc != 1 && argc != 2 {
                    return LanaError::Type;
                }
                let dtype = match tensor_optional_dtype(arguments, argc) {
                    Some(d) => d,
                    None => return LanaError::InvalidParameters,
                };
                let alloc = self.heap.clone();
                let shape = match tensor::tensor_shape_from_array(&alloc, &arguments[0]) {
                    Ok(shape) => shape,
                    Err(error) => return error,
                };
                let mut t = match tensor::tensor_new(&alloc, shape.len(), &shape, false) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                t.dtype = dtype;
                let mut total: usize = 1;
                for i in 0..t.ndim {
                    total *= t.shape[i];
                }
                for i in 0..total {
                    tensor_set_real(&mut t, i, 1.0);
                }
                *out = Value::tensor(Arc::new(t));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_EYE => {
                if argc != 1 && argc != 2 {
                    return LanaError::Type;
                }
                let ValueKind::Number(d) = arguments[0].kind else {
                    return LanaError::Type;
                };
                let dtype = match tensor_optional_dtype(arguments, argc) {
                    Some(d) => d,
                    None => return LanaError::InvalidParameters,
                };
                let n = match tensor::tensor_dimension(d) {
                    Ok(n) => n,
                    Err(error) => return error,
                };
                let shape = [n, n];
                let alloc = self.heap.clone();
                let mut t = match tensor::tensor_new(&alloc, 2, &shape, false) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                t.dtype = dtype;
                for i in 0..n {
                    tensor_set_real(&mut t, i * n + i, 1.0);
                }
                *out = Value::tensor(Arc::new(t));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_DTYPE => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let ValueKind::Tensor(t) = &arguments[0].kind else {
                    return LanaError::Type;
                };
                *out = Value::string(Arc::from(t.dtype.as_str()));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_CAST => {
                if argc != 2 {
                    return LanaError::Type;
                }
                let ValueKind::Tensor(t) = &arguments[0].kind else {
                    return LanaError::Type;
                };
                let ValueKind::String(s) = &arguments[1].kind else {
                    return LanaError::Type;
                };
                let Some(dtype) = TensorDtype::from_str(s) else {
                    return LanaError::InvalidParameters;
                };
                let alloc = self.heap.clone();
                match tensor::tensor_cast(&alloc, t, dtype) {
                    Ok(t) => {
                        *out = Value::tensor(Arc::new(t));
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_RESHAPE => {
                if argc != 2 { return LanaError::Type; }
                let ValueKind::Tensor(source) = &arguments[0].kind else { return LanaError::Type; };
                let result = { let alloc = self.heap.clone(); tensor::tensor_reshape(&alloc, source, &arguments[1]) };
                match result {
                    Ok(value) => {
                        *out = value;
                        if arguments[0].derivation.is_some() { self.ad_record(10, &arguments[0], None, -1, out) } else { LanaError::Ok }
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_TRANSPOSE => {
                if argc != 1 { return LanaError::Type; }
                let ValueKind::Tensor(source) = &arguments[0].kind else { return LanaError::Type; };
                let result = { let alloc = self.heap.clone(); tensor::tensor_transpose(&alloc, source) };
                match result {
                    Ok(value) => {
                        *out = value;
                        if arguments[0].derivation.is_some() { self.ad_record(11, &arguments[0], None, -1, out) } else { LanaError::Ok }
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_EXP | LANA_HOST_TENSOR_LOG | LANA_HOST_TENSOR_SQRT | LANA_HOST_TENSOR_RELU => {
                if argc != 1 { return LanaError::Type; }
                let ValueKind::Tensor(source) = &arguments[0].kind else { return LanaError::Type; };
                let op = host_id - LANA_HOST_TENSOR_EXP;
                let result = { let alloc = self.heap.clone(); tensor::tensor_unary_math(&alloc, source, op) };
                match result {
                    Ok(value) => {
                        *out = value;
                        if arguments[0].derivation.is_some() { self.ad_record(12 + op as i32, &arguments[0], None, -1, out) } else { LanaError::Ok }
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_SOFTMAX | LANA_HOST_TENSOR_LOGSUMEXP => {
                if argc != 1 && argc != 2 { return LanaError::Type; }
                let ValueKind::Tensor(source) = &arguments[0].kind else { return LanaError::Type; };
                let axis_value = if argc == 2 { Some(&arguments[1]) } else { None };
                let result = { let alloc = self.heap.clone(); tensor::tensor_softmax_like(&alloc, source, axis_value, host_id == LANA_HOST_TENSOR_LOGSUMEXP) };
                match result {
                    Ok(value) => {
                        *out = value;
                        if arguments[0].derivation.is_none() { return LanaError::Ok; }
                        let mut axis_number = match axis_value { None => -1.0, Some(v) => v.as_number() };
                        if axis_number < 0.0 { axis_number += source.ndim as f64; }
                        self.ad_record(if host_id == LANA_HOST_TENSOR_LOGSUMEXP { 17 } else { 16 }, &arguments[0], None, axis_number as i32, out)
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_ARGMAX => {
                if argc != 1 && argc != 2 { return LanaError::Type; }
                let ValueKind::Tensor(source) = &arguments[0].kind else { return LanaError::Type; };
                let alloc = self.heap.clone();
                match tensor::tensor_argmax(&alloc, source, if argc == 2 { Some(&arguments[1]) } else { None }) {
                    Ok(value) => { *out = value; LanaError::Ok }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_COMPARE => {
                if argc != 3 { return LanaError::Type; }
                let (ValueKind::Tensor(a), ValueKind::Tensor(b), ValueKind::String(operation)) =
                    (&arguments[0].kind, &arguments[1].kind, &arguments[2].kind) else { return LanaError::Type; };
                let alloc = self.heap.clone();
                match tensor::tensor_compare_values(&alloc, a, b, operation) {
                    Ok(value) => { *out = value; LanaError::Ok }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_SELECT => {
                if argc != 3 { return LanaError::Type; }
                let (ValueKind::Tensor(mask), ValueKind::Tensor(yes), ValueKind::Tensor(no)) =
                    (&arguments[0].kind, &arguments[1].kind, &arguments[2].kind) else { return LanaError::Type; };
                let result = { let alloc = self.heap.clone(); tensor::tensor_select_values(&alloc, mask, yes, no) };
                match result {
                    Ok(value) => {
                        *out = value;
                        if arguments[1].derivation.is_none() && arguments[2].derivation.is_none() { return LanaError::Ok; }
                        let shape = match &out.kind { ValueKind::Tensor(t) => t.shape.clone(), _ => unreachable!() };
                        let alloc = self.heap.clone();
                        let mut yes_mask = match tensor::tensor_new_dtype(&alloc, shape.len(), &shape, yes.dtype) { Ok(t) => t, Err(error) => return error };
                        let mut no_mask = match tensor::tensor_new_dtype(&alloc, shape.len(), &shape, no.dtype) { Ok(t) => t, Err(error) => return error };
                        for i in 0..shape.iter().product() {
                            let selected = if tensor_get_real(mask, tensor::broadcast_index(mask, &shape, i)) != 0.0 { 1.0 } else { 0.0 };
                            tensor_set_real(&mut yes_mask, i, selected); tensor_set_real(&mut no_mask, i, 1.0 - selected);
                        }
                        drop(alloc);
                        let yes_mask_value = Value::tensor(Arc::new(match yes_mask.try_clone(&self.heap) { Ok(t) => t, Err(error) => return error }));
                        let no_mask_value = Value::tensor(Arc::new(match no_mask.try_clone(&self.heap) { Ok(t) => t, Err(error) => return error }));
                        let mut yes_product = { let alloc = self.heap.clone(); match tensor::tensor_elementwise(&alloc, yes, &yes_mask, 2) { Ok(t) => Value::tensor(Arc::new(t)), Err(error) => return error } };
                        if arguments[1].derivation.is_some() { let error = self.ad_record(2, &arguments[1], Some(&yes_mask_value), -1, &mut yes_product); if error != LanaError::Ok { return error; } }
                        let mut no_product = { let alloc = self.heap.clone(); match tensor::tensor_elementwise(&alloc, no, &no_mask, 2) { Ok(t) => Value::tensor(Arc::new(t)), Err(error) => return error } };
                        if arguments[2].derivation.is_some() { let error = self.ad_record(2, &arguments[2], Some(&no_mask_value), -1, &mut no_product); if error != LanaError::Ok { return error; } }
                        self.ad_record(0, &yes_product, Some(&no_product), -1, out)
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_GATHER => {
                if argc != 3 { return LanaError::Type; }
                let (ValueKind::Tensor(source), ValueKind::Tensor(indices)) =
                    (&arguments[0].kind, &arguments[1].kind) else { return LanaError::Type; };
                let result = { let alloc = self.heap.clone(); tensor::tensor_gather_values(&alloc, source, indices, &arguments[2]) };
                match result {
                    Ok(value) => {
                        *out = value;
                        if arguments[0].derivation.is_none() { return LanaError::Ok; }
                        let mut axis = arguments[2].as_number(); if axis < 0.0 { axis += source.ndim as f64; }
                        self.ad_record(18, &arguments[0], Some(&arguments[1]), axis as i32, out)
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_CHOLESKY_SOLVE => {
                if argc != 2 { return LanaError::Type; }
                let (ValueKind::Tensor(matrix), ValueKind::Tensor(rhs)) =
                    (&arguments[0].kind, &arguments[1].kind) else { return LanaError::Type; };
                let result = { let alloc = self.heap.clone(); tensor::tensor_cholesky_solve(&alloc, matrix, rhs) };
                match result {
                    Ok(value) => {
                        *out = value;
                        if arguments[0].derivation.is_some() || arguments[1].derivation.is_some() { self.ad_record(19, &arguments[0], Some(&arguments[1]), -1, out) } else { LanaError::Ok }
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_RANDOM_UNIFORM | LANA_HOST_RANDOM_NORMAL => {
                if argc != 2 { return LanaError::Type; }
                let alloc = self.heap.clone();
                match tensor::tensor_random(&alloc, &arguments[0], &arguments[1], host_id == LANA_HOST_RANDOM_NORMAL) {
                    Ok(value) => { *out = value; LanaError::Ok }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR_DEVICE => {
                if argc != 1 { return LanaError::Type; }
                let ValueKind::Tensor(tensor) = &arguments[0].kind else { return LanaError::Type; };
                *out = Value::string(Arc::from(if tensor.device == TensorDevice::Metal { "metal" } else { "cpu" }));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_TO_DEVICE => {
                if argc != 3 { return LanaError::Type; }
                let (ValueKind::Tensor(source), ValueKind::String(device), ValueKind::Capability(token)) =
                    (&arguments[0].kind, &arguments[1].kind, &arguments[2].kind) else { return LanaError::Type; };
                let target = match &**device { "cpu" => TensorDevice::Cpu, "metal" => TensorDevice::Metal, _ => return LanaError::InvalidParameters };
                if target == TensorDevice::Metal {
                    let shared = &token.shared;
                    let named_gpu = matches!(&shared.base_snapshot.kind, ValueKind::String(name) if &**name == "gpu");
                    if !named_gpu || !capability_allows_locked(shared, token, LANA_CAPABILITY_READ) { return LanaError::Capability; }
                    if source.dtype == TensorDtype::F64 || source.is_complex || !crate::metal::metal_available() { return LanaError::UnsupportedOperation; }
                }
                let result = self.tensor_copy_contiguous(source);
                match result { Ok(mut tensor) => {
                    if target == TensorDevice::Metal {
                        let charge = match self.heap.reserve(source.data.len()) { Ok(charge) => charge, Err(error) => return error };
                        let Some(buffer) = crate::metal::ResidentBuffer::new(&tensor.data) else { return LanaError::UnsupportedOperation; };
                        let tensor_mut = Arc::get_mut(&mut tensor).expect("new tensor is unique");
                        tensor_mut.metal_buffer = Some(Arc::new(buffer));
                        tensor_mut.metal_charge = Some(Arc::new(charge));
                        tensor_mut.device = target;
                    } else { Arc::get_mut(&mut tensor).expect("new tensor is unique").device = target; }
                    *out = Value::tensor(tensor); LanaError::Ok
                }, Err(error) => error }
            }
            LANA_HOST_TENSOR_TO_CPU => {
                if argc != 1 { return LanaError::Type; }
                let ValueKind::Tensor(source) = &arguments[0].kind else { return LanaError::Type; };
                match self.tensor_copy_contiguous(source) { Ok(mut tensor) => {
                    let tensor_mut = Arc::get_mut(&mut tensor).expect("new tensor is unique");
                    if let Some(buffer) = &source.metal_buffer {
                        buffer.copy_into(Arc::get_mut(&mut tensor_mut.data).expect("new tensor buffer is unique"));
                    }
                    tensor_mut.metal_buffer = None;
                    tensor_mut.metal_charge = None;
                    tensor_mut.device = TensorDevice::Cpu;
                    *out = Value::tensor(tensor); LanaError::Ok
                }, Err(error) => error }
            }
            LANA_HOST_TENSOR_SHAPE => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let ValueKind::Tensor(t) = &arguments[0].kind else {
                    return LanaError::Type;
                };
                let mut items = match self.allocate_array_items(t.shape.len()) {
                    Ok(items) => items, Err(error) => return error,
                };
                if let Err(error) = items.extend(t.shape.iter().map(|&dim| Value::number(dim as f64))) { return error; }
                *out = Value::array(Arc::new(Mutex::new(match Array::from_buffer(items) { Ok(array) => array, Err(error) => return error })));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_NDIM => {
                if argc != 1 {
                    return LanaError::Type;
                }
                let ValueKind::Tensor(t) = &arguments[0].kind else {
                    return LanaError::Type;
                };
                *out = Value::number(t.ndim as f64);
                LanaError::Ok
            }
            LANA_HOST_TENSOR_ADD | LANA_HOST_TENSOR_SUB | LANA_HOST_TENSOR_MUL | LANA_HOST_TENSOR_DIV => {
                if argc != 2 {
                    return LanaError::Type;
                }
                let op = host_id - LANA_HOST_TENSOR_ADD;
                let (a_pred, a_var, a_unc) = match tensor::tensor_uncertainty_unpack(&arguments[0]) {
                    Ok(x) => x,
                    Err(error) => return error,
                };
                let (b_pred, b_var, b_unc) = match tensor::tensor_uncertainty_unpack(&arguments[1]) {
                    Ok(x) => x,
                    Err(error) => return error,
                };
                let alloc = self.heap.clone();
                if a_unc || b_unc {
                    let a_var = match a_var {
                        Some(v) => v,
                        None => match tensor::tensor_zeros_like(&alloc, &a_pred) {
                            Ok(t) => Arc::new(t),
                            Err(error) => return error,
                        },
                    };
                    let b_var = match b_var {
                        Some(v) => v,
                        None => match tensor::tensor_zeros_like(&alloc, &b_pred) {
                            Ok(t) => Arc::new(t),
                            Err(error) => return error,
                        },
                    };
                    return match tensor::tensor_elementwise_uncertain(
                        &alloc, &a_pred, &a_var, &b_pred, &b_var, op,
                    ) {
                        Ok(value) => {
                            *out = value;
                            LanaError::Ok
                        }
                        Err(error) => error,
                    };
                }
                let t = match tensor::tensor_elementwise(&alloc, &a_pred, &b_pred, op) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                *out = Value::tensor(Arc::new(t));
                if self.ad_recording {
                    return self.ad_record(op as i32, &arguments[0], Some(&arguments[1]), -1, out);
                }
                LanaError::Ok
            }
            LANA_HOST_TENSOR_MATMUL => {
                if argc != 2 && argc != 3 {
                    return LanaError::Type;
                }
                if argc == 3 && !matches!(arguments[2].kind, ValueKind::String(_)) {
                    return LanaError::Type;
                }
                let (a_pred, a_var, a_unc) = match tensor::tensor_uncertainty_unpack(&arguments[0]) {
                    Ok(x) => x,
                    Err(error) => return error,
                };
                let (b_pred, b_var, b_unc) = match tensor::tensor_uncertainty_unpack(&arguments[1]) {
                    Ok(x) => x,
                    Err(error) => return error,
                };
                let alloc = self.heap.clone();
                if a_unc || b_unc {
                    let a_var = match a_var {
                        Some(v) => v,
                        None => match tensor::tensor_zeros_like(&alloc, &a_pred) {
                            Ok(t) => Arc::new(t),
                            Err(error) => return error,
                        },
                    };
                    let b_var = match b_var {
                        Some(v) => v,
                        None => match tensor::tensor_zeros_like(&alloc, &b_pred) {
                            Ok(t) => Arc::new(t),
                            Err(error) => return error,
                        },
                    };
                    return match tensor::tensor_matmul_uncertain(
                        &alloc, &a_pred, &a_var, &b_pred, &b_var,
                    ) {
                        Ok(value) => {
                            *out = value;
                            LanaError::Ok
                        }
                        Err(error) => error,
                    };
                }
                // LIP-027: optional out_dtype: named parameter. Defaults to the
                // input dtype when both match, else the higher-precision operand.
                let out_dtype = if argc == 3 {
                    let ValueKind::String(s) = &arguments[2].kind else {
                        return LanaError::Type;
                    };
                    match TensorDtype::from_str(s) {
                        Some(d) => d,
                        None => return LanaError::InvalidParameters,
                    }
                } else {
                    tensor::matmul_default_dtype(&a_pred, &b_pred)
                };
                let t = match tensor::tensor_matmul(&alloc, &a_pred, &b_pred, out_dtype) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                *out = Value::tensor(Arc::new(t));
                if self.ad_recording {
                    return self.ad_record(4, &arguments[0], Some(&arguments[1]), -1, out);
                }
                LanaError::Ok
            }
            LANA_HOST_GPU_MATMUL => {
                if argc != 3 {
                    return LanaError::Type;
                }
                let (ValueKind::Tensor(a), ValueKind::Tensor(b), ValueKind::String(precision)) =
                    (&arguments[0].kind, &arguments[1].kind, &arguments[2].kind)
                else {
                    return LanaError::Type;
                };
                if &**precision != "float32" {
                    return LanaError::Type;
                }
                let alloc = self.heap.clone();
                let t = match tensor::tensor_gpu_matmul(&alloc, a, b) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                *out = Value::tensor(Arc::new(t));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_SUM | LANA_HOST_TENSOR_MEAN | LANA_HOST_TENSOR_MAX | LANA_HOST_TENSOR_MIN => {
                if argc != 1 && argc != 2 {
                    return LanaError::Type;
                }
                let op = host_id - LANA_HOST_TENSOR_SUM;
                let (pred, var, unc) = match tensor::tensor_uncertainty_unpack(&arguments[0]) {
                    Ok(x) => x,
                    Err(error) => return error,
                };
                let alloc = self.heap.clone();
                if unc {
                    if op >= 2 {
                        return LanaError::Type;
                    }
                    let var = var.expect("uncertain tensor has a variance");
                    let axis = if argc == 2 { Some(&arguments[1]) } else { None };
                    return match tensor::tensor_reduce_uncertain(&alloc, &pred, &var, op, axis) {
                        Ok(value) => {
                            *out = value;
                            LanaError::Ok
                        }
                        Err(error) => error,
                    };
                }
                let result = if argc == 2 {
                    tensor::tensor_reduce_axis(&alloc, &pred, op, &arguments[1])
                } else {
                    tensor::tensor_reduce(&alloc, &pred, op)
                };
                match result {
                    Ok(value) => {
                        *out = value;
                        if self.ad_recording && op < 2 {
                            let mut ad_axis = -1;
                            if argc == 2 {
                                let axis_number = arguments[1].as_number();
                                let ndim = pred.ndim;
                                ad_axis = if axis_number < 0.0 {
                                    (axis_number + ndim as f64) as i32
                                } else {
                                    axis_number as i32
                                };
                            }
                            return self.ad_record(5 + op as i32, &arguments[0], None, ad_axis, out);
                        }
                        LanaError::Ok
                    }
                    Err(error) => error,
                }
            }
            LANA_HOST_TENSOR => {
                if argc != 1 && argc != 2 {
                    return LanaError::Type;
                }
                let dtype = match tensor_optional_dtype(arguments, argc) {
                    Some(d) => d,
                    None => return LanaError::InvalidParameters,
                };
                let alloc = self.heap.clone();
                let shape = match tensor::tensor_infer_shape(&alloc, &arguments[0]) {
                    Ok(shape) => shape,
                    Err(error) => return error,
                };
                let mut t = match tensor::tensor_new(&alloc, shape.len(), &shape, false) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                t.dtype = dtype;
                let mut offset = 0usize;
                if let Err(error) = tensor::tensor_fill_data(&arguments[0], &mut t, &mut offset) {
                    return error;
                }
                *out = Value::tensor(Arc::new(t));
                LanaError::Ok
            }
            LANA_HOST_TENSOR_COMPLEX => {
                if argc != 2 {
                    return LanaError::Type;
                }
                let alloc = self.heap.clone();
                let shape = match tensor::tensor_infer_shape(&alloc, &arguments[0]) {
                    Ok(shape) => shape,
                    Err(error) => return error,
                };
                let mut t = match tensor::tensor_new(&alloc, shape.len(), &shape, true) {
                    Ok(t) => t,
                    Err(error) => return error,
                };
                let mut offset = 0usize;
                if let Err(error) =
                    tensor::tensor_fill_complex(&arguments[0], &arguments[1], &mut t, &mut offset)
                {
                    return error;
                }
                *out = Value::tensor(Arc::new(t));
                LanaError::Ok
            }
            _ => unreachable!("host_tensor called for a non-tensor host call"),
        }
    }
}
