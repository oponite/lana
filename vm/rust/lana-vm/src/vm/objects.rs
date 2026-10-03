//! Immutable v6 values and statically checked pure methods.
use super::*;
use lana_bytecode::objects::Descriptor;
use crate::value::ObjectValue;

impl Vm<'_> {
    pub(super) fn prepare_object_values(&mut self) -> Result<(), LanaError> {
        if self.object_descriptors.is_some() { return Ok(()); }
        lana_bytecode::verifier::verify(&self.chunk).map_err(|e| e.code)?;
        let descriptors = lana_bytecode::objects::descriptors(&self.chunk).map_err(|e| e.code)?;
        self.verify_pure_object_methods(&descriptors)?;
        for (descriptor, value) in &descriptors {
            for (index, method) in value.methods.iter().enumerate() {
                if let Some(function) = method.function_index {
                    self.object_methods.insert(function, (*descriptor, index as u32));
                }
            }
            for (index, field) in value.fields.iter().enumerate() {
                if let Some(function) = field.default_function {
                    self.object_methods.insert(function, (*descriptor, (value.methods.len() + index) as u32));
                }
            }
        }
        self.object_function_entries = self.chunk.functions.iter().enumerate()
            .map(|(index, function)| (function.entry as usize, self.object_methods.get(&(index as u32)).copied())).collect();
        self.object_function_entries.sort_unstable_by_key(|entry| entry.0);
        self.object_descriptors = Some(descriptors.into_iter().map(|(key, value)| (key, Arc::new(value))).collect());
        Ok(())
    }

    pub(super) fn owns_object(&self, descriptor: u32) -> bool {
        self.current_frame().object_method.is_some_and(|(owner, _)| owner == descriptor)
    }

    fn verify_pure_object_methods(&self, descriptors: &std::collections::BTreeMap<u32, Descriptor>) -> Result<(), LanaError> {
        use OpCode::*;
        let mut roots = Vec::new();
        for descriptor in descriptors.values() {
            for method in &descriptor.methods {
                if let Some(function) = method.function_index {
                    roots.push((function, if method.is_init { method.effect_mask & 8 } else { method.effect_mask }, method.is_init));
                }
            }
            for field in &descriptor.fields { if let Some(function) = field.default_function { roots.push((function, 0, true)); } }
        }
        let mut entries: Vec<_> = self.chunk.functions.iter().enumerate().map(|(i, f)| (f.entry as usize, i)).collect();
        entries.sort_unstable();
        let mut ranges = vec![(0, 0); entries.len()];
        for (position, (start, index)) in entries.iter().enumerate() {
            ranges[*index] = (*start, entries.get(position + 1).map_or(self.chunk.code.len(), |v| v.0));
        }
        for (root, allowed, initializer) in roots {
            let mut pending = vec![root];
            let mut seen = std::collections::HashSet::new();
            while let Some(index) = pending.pop() {
                if !seen.insert(index) { continue; }
                let (start, end) = *ranges.get(index as usize).ok_or(LanaError::Format)?;
                for ins in &self.chunk.code[start..end] {
                    let effect = object_effects::instruction_effect(ins)?;
                    if effect & !allowed != 0 || (initializer && effect != 0 && !matches!(ins.opcode, ObjectNew | OoSet)) {
                        return Err(LanaError::UnsupportedOperation);
                    }
                    match ins.opcode {
                        Call | LoadFunction | Fork | Generator | Async | Lazy => pending.push(ins.b),
                        OoCall | OoStaticCall => {
                            let method = descriptors.get(&ins.c).and_then(|d| d.methods.get(ins.imm as usize)).ok_or(LanaError::Format)?;
                            if method.effect_mask & !allowed != 0 { return Err(LanaError::UnsupportedOperation); }
                            if let Some(function) = method.function_index { pending.push(function); }
                        }
                        _ => {},
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn implemented_interface(&self, receiver: &Value, interface: &Descriptor) -> Result<Arc<Descriptor>, LanaError> {
        let descriptor = match &receiver.kind {
            ValueKind::ObjectValue(value) => value.descriptor.clone(),
            ValueKind::ClassObject(reference) => { self.class_storage(reference)?; reference.descriptor.clone() },
            _ => return Err(LanaError::Type),
        };
        if !descriptor.implements.contains(&interface.qualified_name) { return Err(LanaError::Type); }
        Ok(descriptor)
    }

    pub(super) fn execute_interface_conversion(&mut self, ins: &Instruction) -> LanaError {
        let result = (|| {
            let interface = self.object_descriptors.as_ref().and_then(|d| d.get(&ins.c)).cloned().ok_or(LanaError::Format)?;
            let receiver = self.current_frame().registers[ins.b as usize].clone();
            if receiver.reactive.is_some() { return Err(LanaError::UnresolvedValue); }
            self.check_complete_objects(&receiver)?;
            self.implemented_interface(&receiver, &interface)?;
            Ok(receiver)
        })();
        match result {
            Ok(receiver) => { self.current_frame_mut().registers[ins.a as usize] = receiver; LanaError::Ok },
            Err(error) => error,
        }
    }

    pub(super) fn execute_object_method(&mut self, ins: &Instruction) -> LanaError {
        let result = (|| {
            let descriptor = self.object_descriptors.as_ref().and_then(|d| d.get(&ins.c))
                .cloned().ok_or(LanaError::Format)?;
            if descriptor.kind == "interface" {
                let packed = self.current_frame().registers[ins.b as usize].clone();
                if packed.reactive.is_some() { return Err(LanaError::UnresolvedValue); }
                let ValueKind::Array(array) = &packed.kind else { return Err(LanaError::Type); };
                let receiver = array.lock().unwrap().items.first().cloned().ok_or(LanaError::Type)?;
                if receiver.reactive.is_some() { return Err(LanaError::UnresolvedValue); }
                let concrete = self.implemented_interface(&receiver, &descriptor)?;
                let promise = descriptor.methods.get(ins.imm as usize).ok_or(LanaError::Format)?;
                let member = concrete.methods.iter().position(|m| m.name == promise.name && !m.is_static
                    && m.parameter_types == promise.parameter_types && m.result_type == promise.result_type)
                    .ok_or(LanaError::Type)?;
                let owner = self.object_descriptors.as_ref().unwrap().iter().find(|(_, d)| ***d == *concrete)
                    .map(|(index, _)| *index).ok_or(LanaError::Type)?;
                let mut call = *ins;
                call.c = owner;
                call.imm = member as u32;
                let error = self.execute_object_method(&call);
                return if error == LanaError::Ok { Ok(()) } else { Err(error) };
            }
            let method = descriptor.methods.get(ins.imm as usize).ok_or(LanaError::Format)?;
            if method.is_init || method.is_static != (ins.opcode == OpCode::OoStaticCall)
                || (method.visibility != "public" && !self.owns_object(ins.c)) {
                return Err(LanaError::UnsupportedOperation);
            }
            if method.effect_mask != 0 && (self.active_path_count > 1 || self.pure_callback_depth > 0) {
                return Err(LanaError::UnsupportedOperation);
            }
            let packed = self.current_frame().registers[ins.b as usize].clone();
            if packed.reactive.is_some() { return Err(LanaError::UnresolvedValue); }
            let ValueKind::Array(array) = &packed.kind else { return Err(LanaError::Type); };
            let receiver_count = usize::from(!method.is_static);
            let count = method.parameter_types.len() + receiver_count;
            if array.lock().unwrap().items.len() != count { return Err(LanaError::Type); }
            if !method.is_static {
                let receiver = array.lock().unwrap().items[0].clone();
                if receiver.reactive.is_some() || is_snapshot(&receiver)
                    || matches!(receiver.kind, ValueKind::Possibility(_) | ValueKind::PathSet(_) | ValueKind::Joint(_)) {
                    if descriptor.kind != "value" || method.effect_mask != 0 { return Err(LanaError::UnsupportedOperation); }
                    let parameters = self.array_value(array.lock().unwrap().items[1..].to_vec())?;
                    // Captured parameters must not smuggle mutable/live dependencies into replay.
                    self.validate_value_field(&parameters, false, &mut Vec::new(), 0)?;
                    let function = method.function_index.ok_or(LanaError::Format)?;
                    let current = self.reactive_value(&receiver);
                    let mut result = Value::null();
                    let error = self.lift_object_method_raw(&current, &parameters, function, ins.a, &mut result);
                    if error != LanaError::Ok { return Err(error); }
                    result.derivation = Some(self.record_derivation(DerivationKind::Operation,
                        "method", &[&receiver, &parameters], "", ins.line, DerivationExactness::Exact,
                        &method.name, DerivationOutcome::Success, "none").ok_or(LanaError::Oom)?);
                    if receiver.reactive.is_some() {
                        let error = self.reactive_derived_value(&receiver, Some(&parameters), ReactiveKind::ObjectMethod, function, &mut result);
                        if error != LanaError::Ok { return Err(error); }
                    } else { result = self.information_snapshot(&result)?; }
                    self.current_frame_mut().registers[ins.a as usize] = result;
                    return Ok(());
                }
            }
            if self.frames.len() >= LANA_MAX_CALL_FRAMES as usize { return Err(LanaError::Limit); }
            let function_index = method.function_index.ok_or(LanaError::Format)?;
            let register_count = self.max_registers[function_index as usize];
            let mut callee = Frame::charged(register_count, &self.heap)?;
            for index in 0..count {
                let value = array.lock().unwrap().items[index].clone();
                self.check_complete_objects(&value)?;
                if index == 0 && !method.is_static {
                    if value.reactive.is_some() { return Err(LanaError::UnresolvedValue); }
                    let actual = match &value.kind {
                        ValueKind::ObjectValue(receiver) => &receiver.descriptor,
                        ValueKind::ClassObject(receiver) => &receiver.descriptor,
                        _ => return Err(LanaError::Type),
                    };
                    if **actual != *descriptor { return Err(LanaError::Type); }
                } else {
                    self.check_class_type(&value, &method.parameter_types[index - receiver_count], &descriptor)?;
                }
                callee.registers[index] = value;
            }
            callee.function = function_index;
            callee.return_ip = self.ip;
            callee.return_register = ins.a;
            callee.object_method = Some((ins.c, ins.imm));
            self.frames.push(callee);
            self.ip = self.chunk.functions[function_index as usize].entry as usize;
            Ok(())
        })();
        result.err().unwrap_or(LanaError::Ok)
    }

    pub(super) fn lift_object_method_raw(&mut self, receiver: &Value, parameters: &Value, function: u32, scratch: u32, out: &mut Value) -> LanaError {
        if let ValueKind::Joint(joint) = &receiver.kind {
            // Map named coordinates within the existing law, never construct an independent joint.
            let mut mapped = (**joint).clone();
            for value in &mut mapped.values {
                let mut result = Value::null();
                let error = self.lift_object_method_raw(value, parameters, function, scratch, &mut result);
                if error != LanaError::Ok { return error; }
                *value = result;
            }
            for row in &mut mapped.rows {
                for value in &mut row.values {
                    let mut result = Value::null();
                    let error = self.lift_object_method_raw(value, parameters, function, scratch, &mut result);
                    if error != LanaError::Ok { return error; }
                    *value = result;
                }
            }
            let values = mapped.rows.first().map_or(mapped.values.as_slice(), |row| row.values.as_slice());
            mapped.domains = values.iter().map(Value::value_type).collect();
            *out = Value::joint(Arc::new(mapped));
            return LanaError::Ok;
        }
        let Some((owner, member)) = self.object_methods.get(&function).copied() else { return LanaError::Format; };
        let descriptor = self.object_descriptors.as_ref().unwrap()[&owner].clone();
        let method = &descriptor.methods[member as usize];
        if descriptor.kind != "value" || method.effect_mask != 0 || method.is_static { return LanaError::UnsupportedOperation; }
        let ValueKind::Array(parameters) = &parameters.kind else { return LanaError::Type; };
        let count = parameters.lock().unwrap().items.len();
        let _reservation = match self.heap.reserve((count + 1).saturating_mul(std::mem::size_of::<Value>())) {
            Ok(reservation) => reservation, Err(error) => return error,
        };
        let parameters = parameters.lock().unwrap().items.to_vec();
        if parameters.len() != method.parameter_types.len() { return LanaError::Type; }
        self.lift_pointwise(receiver, out, 0, &mut |vm, receiver, out| {
            let ValueKind::ObjectValue(object) = &receiver.kind else { return LanaError::UnsupportedOperation; };
            if *object.descriptor != *descriptor { return LanaError::Type; }
            for (value, expected) in parameters.iter().zip(&method.parameter_types) {
                if let Err(error) = vm.check_class_type(value, expected, &descriptor) { return error; }
            }
            let mut args = vec![receiver.clone()];
            args.extend(parameters.iter().cloned());
            match vm.run_owned_body(owner, member, function, &args, scratch) {
                Ok(value) => { *out = value; LanaError::Ok },
                Err(error) => error,
            }
        })
    }

    pub(super) fn check_object_return(&mut self, returned: &Value) -> Result<(), LanaError> {
        let Some((owner, member)) = self.current_frame().object_method else { return Ok(()); };
        let descriptor = self.object_descriptors.as_ref().and_then(|d| d.get(&owner)).cloned().ok_or(LanaError::Format)?;
        let result_type = if let Some(method) = descriptor.methods.get(member as usize) {
            if method.is_init { return if matches!(returned.kind, ValueKind::Null) { Ok(()) } else { Err(LanaError::Type) }; }
            method.result_type.as_deref().ok_or(LanaError::Format)?
        } else {
            &descriptor.fields.get(member as usize - descriptor.methods.len()).ok_or(LanaError::Format)?.field_type
        };
        self.check_complete_objects(returned)?;
        self.check_class_type(returned, result_type, &descriptor)
    }

    pub(super) fn execute_object_value(&mut self, ins: &Instruction) -> LanaError {
        let result = (|| {
            let descriptor = self.object_descriptors.as_ref().and_then(|d| d.get(&ins.c))
                .cloned().ok_or(LanaError::Format)?;
            if ins.opcode == OpCode::OoGet {
                let receiver = &self.current_frame().registers[ins.b as usize];
                if receiver.reactive.is_some() { return Err(LanaError::UnresolvedValue); }
                let field = descriptor.fields.get(ins.imm as usize).ok_or(LanaError::Format)?;
                if field.visibility != "public" && !self.owns_object(ins.c) { return Err(LanaError::UnsupportedOperation); }
                return match &receiver.kind {
                    ValueKind::ObjectValue(value) if *value.descriptor == *descriptor => value.fields.get(ins.imm as usize).cloned().ok_or(LanaError::Format),
                    ValueKind::ClassObject(reference) if *reference.descriptor == *descriptor => {
                        let object = self.class_storage(reference)?;
                        if !object.initialized.load(Ordering::Acquire) && !self.constructing.last().is_some_and(|current|
                            std::sync::Weak::ptr_eq(&current.object, &reference.object)) { return Err(LanaError::UnsupportedOperation); }
                        let value = object.fields.lock().unwrap().get(ins.imm as usize).cloned().flatten().ok_or(LanaError::Type);
                        value
                    }
                    _ => Err(LanaError::Type),
                };
            }
            if descriptor.kind != "value" || descriptor.fields.len() != ins.imm as usize
                || (!self.owns_object(ins.c) && descriptor.fields.iter().any(|f| f.visibility != "public")) {
                return Err(LanaError::Type);
            }
            let _bytes = descriptor.fields.len().checked_mul(std::mem::size_of::<Value>())
                .and_then(|n| n.checked_add(std::mem::size_of::<ObjectValue>())).ok_or(LanaError::Oom)?;
            let mut fields = Vec::new();
            fields.try_reserve_exact(descriptor.fields.len()).map_err(|_| LanaError::Oom)?;
            let mut memo = DeepCloneMemo { freeze: true, ..DeepCloneMemo::default() };
            for (index, field) in descriptor.fields.iter().enumerate() {
                let value = self.current_frame().registers[ins.b as usize + index].clone();
                self.validate_value_field(&value, false, &mut Vec::new(), 0)?;
                self.check_value_field_type(&value, &field.field_type, &descriptor, 0)?;
                fields.push(self.deep_clone_value(&value, &mut memo)?);
            }
            let mut value = Value::null();
            value.kind = ValueKind::ObjectValue(self.managed_payload(ObjectValue { descriptor, fields })?);
            Ok(value)
        })();
        match result {
            Ok(value) => { self.current_frame_mut().registers[ins.a as usize] = value; LanaError::Ok }
            Err(error) => error,
        }
    }

    fn validate_value_field(&mut self, value: &Value, captured: bool, ancestors: &mut Vec<usize>, depth: usize) -> Result<(), LanaError> {
        if depth >= 64 { return Err(LanaError::Limit); }
        self.charge_bounded_work(1)?;
        if value.reactive.is_some() || value.claim.is_some() || value.planned_effect.is_some() {
            return Err(LanaError::UnsupportedValue);
        }
        let captured = captured || is_snapshot(value);
        match &value.kind {
            ValueKind::Null | ValueKind::Number(_) | ValueKind::Bool(_) | ValueKind::String(_)
            | ValueKind::State(_) | ValueKind::StateDist(_) | ValueKind::Tensor(_) => return Ok(()),
            ValueKind::ObjectValue(_) | ValueKind::Array(_) | ValueKind::Map(_) | ValueKind::Adt(_) => {},
            ValueKind::Possibility(_) | ValueKind::PathSet(_) | ValueKind::Joint(_) if captured => {},
            _ => return Err(LanaError::UnsupportedValue),
        }
        let identity = value.container_identity().ok_or(LanaError::UnsupportedValue)?;
        if ancestors.contains(&identity) { return Err(LanaError::UnsupportedValue); }
        ancestors.push(identity);
        let mut index = 0;
        while let Some(child) = value.inspection_child(index) {
            self.validate_value_field(&child, captured, ancestors, depth + 1)?;
            index += 1;
        }
        ancestors.pop();
        Ok(())
    }

    pub(super) fn check_value_field_type(&mut self, value: &Value, field_type: &str, owner: &Descriptor, depth: usize) -> Result<(), LanaError> {
        if depth >= 64 { return Err(LanaError::Limit); }
        self.charge_bounded_work(1)?;
        if let Some(inner) = field_type.strip_prefix("Information<").and_then(|s| s.strip_suffix('>')) {
            if !is_snapshot(value) { return Err(LanaError::Type); }
            match &value.kind {
                ValueKind::Possibility(p) => {
                    for item in &p.values { self.check_value_field_type(item, inner, owner, depth + 1)?; }
                }
                ValueKind::PathSet(p) => {
                    for item in &p.alternatives { self.check_value_field_type(&item.result, inner, owner, depth + 1)?; }
                }
                // Joint assignments are named records, not scalar alternatives.
                ValueKind::Joint(_) if matches!(inner, "map" | "Dynamic") => {},
                _ => self.check_value_field_type(value, inner, owner, depth + 1)?,
            }
            return Ok(());
        }
        let expected = if field_type.contains('/') { field_type.to_owned() }
            else { format!("{}/{}", owner.qualified_name.rsplit_once('/').unwrap().0, field_type) };
        if let Some(interface) = self.object_descriptors.as_ref().and_then(|all| all.values().find(|d|
            d.kind == "interface" && d.qualified_name == expected)) {
            self.implemented_interface(value, interface)?;
            return Ok(());
        }
        let matches = match (field_type, &value.kind) {
            ("Dynamic", _) | ("null", ValueKind::Null) | ("number", ValueKind::Number(_))
            | ("bool", ValueKind::Bool(_)) | ("string", ValueKind::String(_))
            | ("STATE", ValueKind::State(_)) | ("STATE_DIST", ValueKind::StateDist(_))
            | ("array", ValueKind::Array(_)) | ("map", ValueKind::Map(_))
            | ("Tensor", ValueKind::Tensor(_)) => true,
            ("Shape", _) => { tensor::tensor_shape_from_array(&self.heap, value)?; true },
            (_, ValueKind::ObjectValue(object)) => {
                let name = if field_type == "Self" { owner.qualified_name.clone() }
                    else if field_type.contains('/') { field_type.to_string() }
                    else { format!("{}/{}", owner.qualified_name.rsplit_once('/').unwrap().0, field_type) };
                object.descriptor.qualified_name == name
                    && self.object_descriptors.as_ref().is_some_and(|all|
                        all.values().any(|d| **d == *object.descriptor))
            }
            _ => false,
        };
        if matches { Ok(()) } else { Err(LanaError::Type) }
    }
}

fn is_snapshot(value: &Value) -> bool {
    value.reactive.is_none() && value.derivation.as_ref().is_some_and(|d| d.operation.as_ref() == "snapshot")
}

pub(super) fn values_equal(left: &Value, right: &Value, out: &mut bool) -> LanaError {
    // Bound traversal even when many fields alias the same nested snapshot.
    fn charge(work: &mut usize, depth: usize) -> Result<(), LanaError> {
        if depth >= 64 || *work == 0 { return Err(LanaError::Limit); }
        *work -= 1;
        Ok(())
    }
    fn supported(value: &Value, depth: usize, work: &mut usize) -> Result<(), LanaError> {
        charge(work, depth)?;
        match &value.kind {
            ValueKind::ObjectValue(value) => {
                for field in &value.fields { supported(field, depth + 1, work)?; }
                Ok(())
            }
            ValueKind::Null | ValueKind::Number(_) | ValueKind::Bool(_) | ValueKind::String(_)
            | ValueKind::State(_) | ValueKind::Array(_) => Ok(()),
            _ => Err(LanaError::UnsupportedOperation),
        }
    }
    fn compare(left: &Value, right: &Value, depth: usize, work: &mut usize) -> Result<bool, LanaError> {
        charge(work, depth)?;
        match (&left.kind, &right.kind) {
            (ValueKind::ObjectValue(a), ValueKind::ObjectValue(b)) => {
                if a.descriptor != b.descriptor { return Ok(false); }
                let mut equal = true;
                for (a, b) in a.fields.iter().zip(&b.fields) { equal &= compare(a, b, depth + 1, work)?; }
                Ok(equal)
            }
            (ValueKind::ObjectValue(_), _) | (_, ValueKind::ObjectValue(_)) => Ok(false),
            _ => {
                let mut equal = false;
                let error = super::values_equal(left, right, &mut equal);
                if error == LanaError::Ok { Ok(equal) } else { Err(error) }
            }
        }
    }
    let mut work = 100_000;
    let result = supported(left, 0, &mut work).and_then(|_| supported(right, 0, &mut work))
        .and_then(|_| compare(left, right, 0, &mut work));
    match result { Ok(equal) => { *out = equal; LanaError::Ok }, Err(error) => error }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lana_bytecode::assembler;

    fn chunk(types: &[&str]) -> Chunk {
        let descriptor = serde_json::json!({"schema_version":1,"kind":"value","qualified_name":"file/main.lana/Reading",
            "fields":types.iter().enumerate().map(|(i, t)| serde_json::json!({"name":format!("f{i}"), "type":t,
                "visibility":"public", "mutable":false,"default_function":null})).collect::<Vec<_>>(),
            "methods":[],"implements":[]}).to_string();
        let hex: String = descriptor.bytes().map(|b| format!("{b:02x}")).collect();
        assembler::assemble(&format!(".version 6\n.function main 0 16\nVALUE_NEW R10 {} {hex} {}\nHALT\n",
            if types.is_empty() { "-" } else { "R0" }, types.len())).unwrap()
    }

    fn construct(vm: &mut Vm<'_>, args: &[Value]) -> Result<Value, LanaError> {
        for (i, arg) in args.iter().enumerate() { vm.current_frame_mut().registers[i] = arg.clone(); }
        let error = vm.execute(&vm.chunk.code[0].clone());
        if error == LanaError::Ok { Ok(vm.current_frame().registers[10].clone()) } else { Err(error) }
    }

    fn method_source(factory_prefix: &str, getter: &str, helper: &str) -> String {
        let method = |name: &str, static_: bool, visibility: &str, parameters: Vec<&str>, result: &str, function| {
            serde_json::json!({"name":name,"visibility":visibility,"static":static_,"parameter_types":parameters,
                "result_type":result,"effect_mask":0,"function_index":function,"is_init":false})
        };
        let descriptor = serde_json::json!({"schema_version":1,"kind":"value","qualified_name":"file/main.lana/Reading",
            "fields":[{"name":"state","type":"STATE","visibility":"private","mutable":false,"default_function":null}],
            "methods":[method("create", true, "public", vec!["STATE"], "Self", 1),
                       method("read", false, "public", vec![], "STATE", 2),
                       method("hidden", false, "private", vec![], "STATE", 3)], "implements":[]}).to_string();
        let hex: String = descriptor.bytes().map(|b| format!("{b:02x}")).collect();
        format!(".version 6\n.function main 0 6\nSTATE_NEW R0 0.5 0.3 0.4\nARRAY_NEW R1 R0 1\nOO_STATIC_CALL R2 R1 {hex} 0\nARRAY_NEW R3 R2 1\nOO_CALL R4 R3 {hex} 1\nRETURN R4\n.function create 1 2\n{factory_prefix}VALUE_NEW R1 R0 {hex} 1\nRETURN R1\n.function read 1 2\nARRAY_NEW R1 R0 1\nOO_CALL R1 R1 {hex} 2\nRETURN R1\n.function hidden 1 2\n{getter}\nRETURN R1\n{helper}",
            getter=getter.replace("DESCRIPTOR", &hex))
    }

    #[test]
    fn interface_dispatch_preserves_value_and_class_identity() {
        for kind in ["value", "class"] {
            let method = serde_json::json!({"name":"read","visibility":"public","static":false,
                "parameter_types":[],"result_type":"STATE","effect_mask":0,"function_index":1,"is_init":false});
            let mut promise = method.clone();
            promise["function_index"] = serde_json::Value::Null;
            let interface = serde_json::json!({"schema_version":1,"kind":"interface","qualified_name":"file/main.lana/Readable",
                "fields":[],"methods":[promise],"implements":[]});
            let descriptor = serde_json::json!({"schema_version":1,"kind":kind,"qualified_name":"file/main.lana/Reading",
                "fields":[{"name":"state","type":"STATE","visibility":"public","mutable":false,
                    "default_function":if kind == "class" { Some(2) } else { None }}],
                "methods":[method],"implements":["file/main.lana/Readable"]});
            let hex = |value: &serde_json::Value| value.to_string().bytes().map(|b| format!("{b:02x}")).collect::<String>();
            let (d, i) = (hex(&descriptor), hex(&interface));
            let construct = if kind == "class" { format!("OBJECT_NEW R1 - {d} 0") } else { format!("VALUE_NEW R1 R0 {d} 1") };
            let source = format!(".version 6\n.function main 0 6\nSTATE_NEW R0 0.5 0.3 0.4\n{construct}\nOO_AS_INTERFACE R2 R1 {i} -\nCOMPARE R1 == R2 R3\nARRAY_NEW R4 R2 1\nOO_CALL R5 R4 {i} 0\nRETURN R5\n.function read 1 2\nOO_GET R1 R0 {d} 0\nRETURN R1\n.function default 0 1\nSTATE_NEW R0 0.5 0.3 0.4\nRETURN R0\n");
            let chunk = assembler::assemble(&source).unwrap();
            let mut vm = Vm::new(&chunk);
            assert_eq!(vm.run(), LanaError::Ok);
            assert!(vm.frames[0].registers[3].as_bool());
            assert_eq!(vm.result.as_state().state, State { p: 0.5, d_re: 0.3, d_im: 0.4 });
            let mut wrong = descriptor.clone();
            wrong["implements"] = serde_json::json!([]);
            let chunk = assembler::assemble(&source.replace(&d, &hex(&wrong))).unwrap();
            let mut vm = Vm::new(&chunk);
            assert_eq!(vm.run(), LanaError::Type);
            assert!(matches!(vm.frames[0].registers[2].kind, ValueKind::Null));
        }
    }

    #[test]
    fn value_methods_lift_preserving_dependencies_and_live_replay() {
        let chunk = assembler::assemble(&method_source("", "OO_GET R1 R0 DESCRIPTOR 0", "")).unwrap();
        let mut vm = Vm::new(&chunk);
        assert_eq!(vm.run(), LanaError::Ok);
        let first = vm.frames[0].registers[2].clone();
        let ValueKind::ObjectValue(object) = &first.kind else { panic!("value"); };
        let mut second = Value::null();
        second.kind = ValueKind::ObjectValue(Arc::new(ObjectValue {
            descriptor: object.descriptor.clone(), fields: vec![Value::state(StateValue::default())],
        }));
        let alternatives = Value::possibility(Arc::new(Possibility {
            values: vec![first.clone(), second], weights: Some(vec![0.25, 0.75]), dependency_id: 91,
        }));
        let root = vm.information_root(&alternatives).unwrap();
        let call = chunk.code[4];
        for receiver in [alternatives.clone(), root.clone(), vm.information_snapshot(&first).unwrap()] {
            vm.frames[0].registers[3] = vm.array_value(vec![receiver]).unwrap();
            vm.ip = 5;
            vm.running = true;
            assert_eq!(vm.execute_object_method(&call), LanaError::Ok);
            let result = vm.frames[0].registers[4].clone();
            if let ValueKind::Possibility(p) = vm.reactive_value(&result).kind {
                assert_eq!(p.dependency_id, 91);
                assert_eq!(p.weights, Some(vec![0.25, 0.75]));
                assert_eq!(p.values[0].as_state().state, object.fields[0].as_state().state);
            }
            if result.reactive.is_some() {
                vm.reactive_recompute_transaction(root.reactive.as_ref().unwrap(), &first, 5).unwrap();
                assert_eq!(vm.reactive_value(&result).as_state().state, object.fields[0].as_state().state);
            }
        }
        let paths = Value::paths(Arc::new(PathSet { dependency_id: 42, alternatives: vec![
            PathAlternative { guard: true, weight: 0.4, result: first.clone() },
        ] }));
        let joint = Value::joint(Arc::new(JointState { names: vec![Arc::from("reading")],
            domains: vec![first.value_type()], values: vec![], rows: vec![JointRow {
                values: vec![first.clone()], weight: 1.0,
            }], kind: JointKind::FiniteLaw, capabilities: LANA_JOINT_CAN_PROJECT,
        }));
        for receiver in [paths, joint] {
            vm.frames[0].registers[3] = vm.array_value(vec![receiver]).unwrap();
            vm.ip = 5;
            assert_eq!(vm.execute_object_method(&call), LanaError::Ok);
            match &vm.frames[0].registers[4].kind {
                ValueKind::PathSet(paths) => {
                    assert_eq!(paths.dependency_id, 42);
                    assert!(paths.alternatives[0].guard);
                    assert_eq!(paths.alternatives[0].weight, 0.4);
                }
                ValueKind::Joint(joint) => {
                    assert_eq!(joint.names[0].as_ref(), "reading");
                    assert_eq!(joint.rows[0].weight, 1.0);
                    assert_eq!(joint.rows[0].values[0].as_state().state, object.fields[0].as_state().state);
                }
                _ => panic!("lift changed Information form"),
            }
        }
        assert_eq!(vm.observation_count, 0);
    }

    #[test]
    fn declared_io_methods_and_additional_pure_operations_execute() {
        let mut chunk = assembler::assemble(&method_source("PRINT R0\n", "OO_GET R1 R0 DESCRIPTOR 0", "")).unwrap();
        for constant in &mut chunk.constants {
            if let ConstantValue::String(json) = constant {
                if json.contains("schema_version") { *json = json.replacen("\"effect_mask\":0", "\"effect_mask\":4", 1); }
            }
        }
        assert_eq!(Vm::new(&chunk).run(), LanaError::Ok);
        let chunk = assembler::assemble(&method_source("", "OO_GET R1 R0 DESCRIPTOR 0\nTRANSFORM R1 R1 invert", "")).unwrap();
        assert_eq!(Vm::new(&chunk).run(), LanaError::Ok);
    }

    #[test]
    fn value_methods_factory_private_access_and_checked_returns() {
        let source = method_source("", "OO_GET R1 R0 DESCRIPTOR 0", "");
        let chunk = assembler::assemble(&source).unwrap();
        let mut vm = Vm::new(&chunk);
        assert_eq!(vm.run(), LanaError::Ok);
        assert_eq!(vm.result.as_state().state, State { p:0.5, d_re:0.3, d_im:0.4 });
        let wrong = source.replace("STATE_NEW R0 0.5 0.3 0.4", "LOAD_CONST R0 true");
        let chunk = assembler::assemble(&wrong).unwrap();
        let mut vm = Vm::new(&chunk);
        assert_eq!(vm.run(), LanaError::Type);
        assert!(matches!(vm.frames[0].registers[2].kind, ValueKind::Null));
        for wrong in [source.replacen("ARRAY_NEW R1 R0 1", "ARRAY_NEW R1 R0 0", 1),
                      source.replace("ARRAY_NEW R3 R2 1", "ARRAY_NEW R3 R0 1")] {
            let chunk = assembler::assemble(&wrong).unwrap();
            assert_eq!(Vm::new(&chunk).run(), LanaError::Type);
        }
        let chunk = assembler::assemble(&source).unwrap();
        let mut limited = Vm::new(&chunk);
        assert_eq!(limited.set_memory_limit(1), Err(LanaError::Oom));
        limited.set_memory_limit(limited.allocated_bytes()).unwrap();
        assert_eq!(limited.run(), LanaError::Oom);
        assert!(matches!(limited.result.kind, ValueKind::Null));
        let wrong = method_source("", "LOAD_CONST R1 true", "");
        let chunk = assembler::assemble(&wrong).unwrap();
        let mut vm = Vm::new(&chunk);
        vm.frames[0].registers[4] = Value::number(77.0);
        assert_eq!(vm.run(), LanaError::Type);
        assert_eq!(vm.frames[0].registers[4].as_number(), 77.0);
        assert!(matches!(vm.result.kind, ValueKind::Null));
    }

    #[test]
    fn value_methods_reject_effects_transitively_before_first_instruction() {
        for (prefix, helper) in [
            ("PRINT R0\n", ""),
            ("HOST_CALL random R0 0 R1\n", ""),
            ("CALL helper R0 0 R1\n", ".function helper 0 1\nLOAD_CONST R0 42\nPRINT R0\nRETURN R0\n"),
            ("HALT\n", ""),
        ] {
            let chunk = assembler::assemble(&method_source(prefix, "OO_GET R1 R0 DESCRIPTOR 0", helper)).unwrap();
            for debug in [false, true] {
                let mut vm = Vm::new(&chunk);
                assert_eq!(if debug { vm.debug_step() } else { vm.run() }, LanaError::UnsupportedOperation);
                assert!(matches!(vm.frames[0].registers[0].kind, ValueKind::Null));
            }
        }
    }

    #[test]
    fn value_methods_named_entry_cannot_bypass_ownership() {
        let chunk = assembler::assemble(&method_source("", "OO_GET R1 R0 DESCRIPTOR 0", "")).unwrap();
        let mut vm = Vm::new(&chunk);
        assert_eq!(vm.run_pure_dataset_plan("create", &[Value::state(StateValue::default())]).unwrap_err(), LanaError::UnsupportedOperation);
        assert!(matches!(vm.result.kind, ValueKind::Null));
        // Even an abandoned callback's instruction pointer cannot grant ownership.
        vm.ip = chunk.functions[3].entry as usize + 1;
        let ins = chunk.code[vm.ip - 1];
        assert_eq!(vm.execute(&ins), LanaError::UnsupportedOperation);
    }

    #[test]
    fn immutable_values_tensor_and_shape_fields() {
        let chunk = chunk(&["Tensor", "Shape"]);
        let mut vm = Vm::new(&chunk);
        for dtype in [TensorDtype::F64, TensorDtype::F32, TensorDtype::F16, TensorDtype::Bf16, TensorDtype::Complex] {
            let mut tensor = tensor::tensor_new_dtype(&vm.heap, 2, &[1, 2], dtype).unwrap();
            tensor_set_real(&mut tensor, 0, 1.25);
            tensor_set_imag(&mut tensor, 0, 2.5);
            let tensor = Arc::new(tensor::tensor_transpose_last_two(&vm.heap, &tensor).unwrap());
            let input = Value::tensor(tensor.clone());
            let shape = vm.array_value(vec![Value::number(2.0), Value::number(1.0)]).unwrap();
            let result = construct(&mut vm, &[input.clone(), shape.clone()]).unwrap();
            let ValueKind::ObjectValue(value) = &result.kind else { panic!("value"); };
            let ValueKind::Tensor(captured) = &value.fields[0].kind else { panic!("tensor"); };
            assert!(Arc::ptr_eq(&tensor, captured)); // Published tensor storage is immutable.
            assert_eq!(captured.shape(), &[2, 1]);
            assert_eq!(captured.dtype(), dtype);
            assert_eq!(tensor_get_real(captured, 0), 1.25);
            assert_eq!(tensor_get_imag(captured, 0), if dtype == TensorDtype::Complex { 2.5 } else { 0.0 });
            let changed = tensor::tensor_elementwise_scalar(&vm.heap, &tensor, 1.0, 0).unwrap();
            assert_eq!(tensor_get_real(&changed, 0), 2.25);
            assert_eq!(tensor_get_real(captured, 0), 1.25);
            let ValueKind::Array(original) = shape.kind else { panic!("shape"); };
            original.lock().unwrap().items[0] = Value::number(9.0);
            let ValueKind::Array(frozen) = &value.fields[1].kind else { panic!("shape"); };
            assert!(frozen.lock().unwrap().frozen);
            assert_eq!(frozen.lock().unwrap().items[0].as_number(), 2.0);
            assert_eq!(values_equal(&result, &result, &mut false), LanaError::UnsupportedOperation);
            let snapshot = vm.information_snapshot(&input).unwrap();
            assert!(matches!(snapshot.kind, ValueKind::Tensor(_)));
            for (shape, error) in [
                (vec![Value::number(-1.0)], LanaError::InvalidParameters),
                (vec![Value::number(0.5)], LanaError::InvalidParameters),
                (vec![Value::number(f64::NAN)], LanaError::InvalidParameters),
                (vec![Value::number(1.0); 33], LanaError::InvalidParameters),
                (vec![Value::boolean(true)], LanaError::Type),
            ] {
                let shape = vm.array_value(shape).unwrap();
                vm.current_frame_mut().registers[10] = Value::number(77.0);
                assert_eq!(construct(&mut vm, &[input.clone(), shape]).unwrap_err(), error);
                assert_eq!(vm.current_frame().registers[10].as_number(), 77.0);
            }
            assert_eq!(construct(&mut vm, &[Value::number(1.0), Value::null()]).unwrap_err(), LanaError::Type);
        }
    }

    #[test]
    fn immutable_values_keep_core_v5_sampling_and_observation_rules() {
        let chunk = chunk(&["Dynamic"]);
        let mut vm = Vm::new(&chunk);
        let uncertain = Value::possibility(vm.possibility_build(&[Value::number(1.0), Value::number(2.0)]).unwrap());
        assert_eq!(vm.information_sample(&uncertain).unwrap_err(), LanaError::UnsupportedOperation);
        let root = vm.information_root(&uncertain).unwrap();
        assert_eq!(vm.information_observe(&root, &Value::number(3.0)).unwrap_err(), LanaError::InvalidConditioning);
        assert_eq!(vm.observation_count, 0);
        vm.information_observe(&root, &Value::number(1.0)).unwrap();
        assert_eq!(vm.observation_count, 1);
        let snapshot = vm.information_snapshot(&root).unwrap();
        construct(&mut vm, &[snapshot]).unwrap();
    }

    #[test]
    fn immutable_values_freeze_preserve_state_and_check_receivers() {
        let chunk = chunk(&["STATE", "array"]);
        let mut vm = Vm::new(&chunk);
        let state = Value::state(StateValue { state: State { p:0.5, d_re:0.3, d_im:0.4 }, ..StateValue::default() });
        let array = vm.array_value(vec![Value::number(7.0)]).unwrap();
        let result = construct(&mut vm, &[state.clone(), array.clone()]).unwrap();
        let ValueKind::ObjectValue(value) = &result.kind else { panic!("value"); };
        assert_eq!(value.fields[0].as_state().state, state.as_state().state);
        let ValueKind::Array(captured) = &value.fields[1].kind else { panic!("array"); };
        assert!(captured.lock().unwrap().frozen);
        let ValueKind::Array(original) = array.kind else { panic!("array"); };
        original.lock().unwrap().items[0] = Value::number(9.0);
        assert_eq!(captured.lock().unwrap().items[0].as_number(), 7.0);
        let get = Instruction::new(OpCode::OoGet, 11, 10, chunk.code[0].c, 0, 0);
        assert_eq!(vm.execute(&get), LanaError::Ok);
        assert_eq!(vm.current_frame().registers[11].as_state().state, state.as_state().state);
        vm.current_frame_mut().registers[10] = Value::null();
        assert_eq!(vm.execute(&get), LanaError::Type);
        assert!(result.print().starts_with("value<file/main.lana/Reading>"));
        assert_eq!(vm.observation_count, 0);
    }

    #[test]
    fn immutable_values_reject_live_cycles_handles_and_publish_nothing() {
        let chunk = chunk(&["Dynamic"]);
        let mut vm = Vm::new(&chunk);
        let live = vm.information_root(&Value::number(7.0)).unwrap();
        let nested = vm.array_value(vec![live.clone()]).unwrap();
        let cyclic = vm.array_value(vec![]).unwrap();
        let ValueKind::Array(array) = &cyclic.kind else { panic!("array"); };
        array.lock().unwrap().items.push(cyclic.clone()).unwrap();
        for input in [live, nested, cyclic.clone(), Value::function(0)] {
            vm.current_frame_mut().registers[10] = Value::number(77.0);
            assert_eq!(construct(&mut vm, &[input]).unwrap_err(), LanaError::UnsupportedValue);
            assert_eq!(vm.current_frame().registers[10].as_number(), 77.0);
            assert!(matches!(vm.result.kind, ValueKind::Null));
        }
        array.lock().unwrap().items.clear();
        assert_eq!(vm.set_memory_limit(0), Err(LanaError::Oom));
        vm.collect_classes().unwrap();
        vm.set_memory_limit(vm.allocated_bytes()).unwrap();
        assert_eq!(construct(&mut vm, &[Value::number(1.0)]).unwrap_err(), LanaError::Oom);
        assert_eq!(vm.current_frame().registers[10].as_number(), 77.0);
    }

    #[test]
    fn immutable_values_capture_information_and_reject_unsupported_equality() {
        let chunk = chunk(&["Information<number>"]);
        let mut vm = Vm::new(&chunk);
        let uncertain = Value::possibility(vm.possibility_build(&[Value::number(1.0), Value::number(2.0)]).unwrap());
        assert_eq!(construct(&mut vm, &[uncertain.clone()]).unwrap_err(), LanaError::UnsupportedValue);
        assert_eq!(construct(&mut vm, &[Value::number(1.0)]).unwrap_err(), LanaError::Type);
        let snapshot = vm.information_snapshot(&uncertain).unwrap();
        let captured = construct(&mut vm, &[snapshot.clone()]).unwrap();
        let ValueKind::ObjectValue(value) = &captured.kind else { panic!("value"); };
        let (ValueKind::Possibility(before), ValueKind::Possibility(after)) = (&snapshot.kind, &value.fields[0].kind) else { panic!("law"); };
        assert_eq!(before.dependency_id, after.dependency_id);
        assert_eq!(before.weights, after.weights);
        assert_eq!(after.values.len(), 2);
        assert_eq!(values_equal(&captured, &captured, &mut false), LanaError::UnsupportedOperation);
        let wrong = vm.information_snapshot(&Value::boolean(true)).unwrap();
        assert_eq!(construct(&mut vm, &[wrong]).unwrap_err(), LanaError::Type);
        let root = vm.information_root(&Value::number(7.0)).unwrap();
        let snapshot = vm.information_snapshot(&root).unwrap();
        let captured = construct(&mut vm, &[snapshot]).unwrap();
        assert_eq!(values_equal(&captured, &captured, &mut false), LanaError::Ok);
    }

    #[test]
    fn immutable_values_nominal_types_and_all_fields_determine_equality() {
        let chunk = chunk(&["number", "Dynamic"]);
        let mut vm = Vm::new(&chunk);
        let a = construct(&mut vm, &[Value::number(1.0), Value::boolean(true)]).unwrap();
        let b = construct(&mut vm, &[Value::number(1.0), Value::boolean(true)]).unwrap();
        let mut equal = false;
        assert_eq!(values_equal(&a, &b, &mut equal), LanaError::Ok);
        assert!(equal);
        let c = construct(&mut vm, &[Value::number(2.0), a.clone()]).unwrap();
        let d = construct(&mut vm, &[Value::number(2.0), b.clone()]).unwrap();
        assert_eq!(values_equal(&c, &d, &mut equal), LanaError::Ok);
        assert!(equal);
        let unsupported = Value::map(Arc::new(Mutex::new(Map::new(&vm.heap, 0).unwrap())));
        let e = construct(&mut vm, &[Value::number(9.0), unsupported]).unwrap();
        assert_eq!(values_equal(&a, &e, &mut equal), LanaError::UnsupportedOperation);
        assert_eq!(construct(&mut vm, &[Value::boolean(true), b]).unwrap_err(), LanaError::Type);
    }
}
