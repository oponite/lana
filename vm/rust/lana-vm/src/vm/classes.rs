//! Task-local class storage, transactional initialization, and graph transfer.
use super::*;
use crate::value::{ClassObject, ClassReference};
use lana_bytecode::objects::Descriptor;

pub(super) struct Construction {
    pub caller_depth: usize,
    pub checkpoint: usize,
    instruction: Instruction,
    return_ip: usize,
    args: Vec<Value>,
    next_field: usize,
    returned_field: Option<usize>,
    init_started: bool,
}

impl Vm<'_> {
    fn allocate_class(&mut self, descriptor: Arc<Descriptor>) -> Result<Arc<ClassReference>, LanaError> {
        if self.class_objects.len() == self.class_objects.capacity() {
            let capacity = self.class_objects.capacity().saturating_mul(2).max(4);
            let bytes = capacity.checked_mul(std::mem::size_of::<Arc<ClassObject>>()).ok_or(LanaError::Oom)?;
            let mut reservation = self.class_objects_reservation.lock().unwrap();
            let old_bytes = self.class_objects.capacity() * std::mem::size_of::<Arc<ClassObject>>();
            reservation.resize(bytes)?;
            if self.class_objects.try_reserve_exact(capacity - self.class_objects.len()).is_err() {
                reservation.resize(old_bytes)?;
                return Err(LanaError::Oom);
            }
            let actual = self.class_objects.capacity().checked_mul(std::mem::size_of::<Arc<ClassObject>>()).ok_or(LanaError::Oom)?;
            reservation.resize(actual)?;
        }
        let bytes = descriptor.fields.len().checked_mul(std::mem::size_of::<Option<Value>>())
            .and_then(|n| n.checked_add(std::mem::size_of::<ClassObject>() + std::mem::size_of::<ClassReference>()))
            .ok_or(LanaError::Oom)?;
        let mut allocation = self.heap.reserve(bytes)?;
        allocation.admit_gc(2, descriptor.fields.len().checked_mul(5).and_then(|n| n.checked_add(1)).ok_or(LanaError::Oom)?)?;
        let reference = Arc::new_cyclic(|reference| {
            let object = Arc::new(ClassObject { fields: Mutex::new(vec![None; descriptor.fields.len()]),
                initialized: AtomicBool::new(false), generation: std::sync::atomic::AtomicU8::new(crate::heap::Generation::Young as u8),
                remembered: AtomicBool::new(false), reference: reference.clone(), _allocation: allocation });
            let result = ClassReference { descriptor, owner: self.class_owner.clone(), object: Arc::downgrade(&object) };
            self.class_objects.push(object);
            result
        });
        self.heap.bump_mutation_epoch();
        self.class_allocations_since_gc += 1;
        Ok(reference)
    }

    pub(super) fn class_storage(&self, reference: &ClassReference) -> Result<Arc<ClassObject>, LanaError> {
        if !Arc::ptr_eq(&reference.owner, &self.class_owner) { return Err(LanaError::Task); }
        reference.object.upgrade().ok_or(LanaError::Task)
    }

    pub(super) fn check_complete_objects(&mut self, value: &Value) -> Result<(), LanaError> {
        let mut pending = vec![value.clone()];
        let mut seen = std::collections::HashSet::new();
        while let Some(value) = pending.pop() {
            self.charge_bounded_work(1)?;
            if let ValueKind::ClassObject(reference) = &value.kind {
                let object = self.class_storage(reference)?;
                if !object.initialized.load(Ordering::Acquire)
                    && !(self.constructing.iter().any(|c| std::sync::Weak::ptr_eq(&c.object, &reference.object))
                        && object.fields.lock().unwrap().iter().all(Option::is_some)) {
                    return Err(LanaError::UnsupportedOperation);
                }
            }
            let Some(identity) = value.container_identity() else { continue; };
            if !seen.insert(identity) { continue; }
            let mut index = 0;
            while let Some(child) = value.inspection_child(index) { pending.push(child); index += 1; }
        }
        Ok(())
    }

    pub(super) fn check_class_type(&mut self, value: &Value, field_type: &str, owner: &Descriptor) -> Result<(), LanaError> {
        if let Some(inner) = field_type.strip_prefix("Information<").and_then(|s| s.strip_suffix('>')) {
            if value.reactive.is_some() || matches!(value.kind, ValueKind::Possibility(_) | ValueKind::PathSet(_) | ValueKind::Joint(_))
                || value.derivation.as_ref().is_some_and(|d| d.operation.as_ref() == "snapshot") {
                let current = self.reactive_value(value);
                return self.check_class_information_inner(&current, inner, owner);
            }
        }
        let expected = if field_type.contains('/') { field_type.to_owned() }
            else { format!("{}/{}", owner.qualified_name.rsplit_once('/').unwrap().0, field_type) };
        let interface = self.object_descriptors.as_ref().and_then(|all| all.values().find(|d|
            d.kind == "interface" && d.qualified_name == expected)).cloned();
        if let Some(interface) = interface {
            if value.reactive.is_some() { return Err(LanaError::UnresolvedValue); }
            self.implemented_interface(value, &interface)?;
            return Ok(());
        }
        if field_type == "Dynamic" { return Ok(()); }
        if value.reactive.is_some() && field_type != "Dynamic" { return Err(LanaError::UnresolvedValue); }
        if let ValueKind::ClassObject(reference) = &value.kind {
            if value.reactive.is_some() { return Err(LanaError::UnresolvedValue); }
            self.class_storage(reference)?;
            let expected = if field_type == "Self" { owner.qualified_name.clone() }
                else if field_type.contains('/') { field_type.to_owned() }
                else { format!("{}/{}", owner.qualified_name.rsplit_once('/').unwrap().0, field_type) };
            if field_type == "Dynamic" || reference.descriptor.qualified_name == expected { return Ok(()); }
            return Err(LanaError::Type);
        }
        self.check_value_field_type(value, field_type, owner, 0)
    }

    fn check_class_information_inner(&mut self, value: &Value, inner: &str, owner: &Descriptor) -> Result<(), LanaError> {
        match &value.kind {
            ValueKind::Possibility(p) => { for v in &p.values { self.check_class_type(v, inner, owner)?; } Ok(()) }
            ValueKind::PathSet(p) => { for v in &p.alternatives { self.check_class_type(&v.result, inner, owner)?; } Ok(()) }
            ValueKind::Joint(_) if matches!(inner, "map" | "Dynamic") => Ok(()),
            _ => self.check_class_type(value, inner, owner),
        }
    }

    pub(super) fn run_owned_body(&mut self, owner: u32, member: u32, function: u32, args: &[Value], scratch: u32) -> Result<Value, LanaError> {
        let saved = self.current_frame().registers[scratch as usize].clone();
        let history = self.current_frame().histories[scratch as usize].clone();
        let mut result = Value::null();
        let error = self.run_function_args_owned(function, args, scratch, &mut result, Some((owner, member)));
        self.current_frame_mut().registers[scratch as usize] = saved;
        self.current_frame_mut().histories[scratch as usize] = history;
        if error == LanaError::Ok { Ok(result) } else { Err(error) }
    }

    pub(super) fn execute_class_instruction(&mut self, ins: &Instruction) -> LanaError {
        let result: Result<Option<Value>, LanaError> = (|| {
            if self.active_path_count > 1 || self.pure_callback_depth > 0 { return Err(LanaError::UnsupportedOperation); }
            let descriptor = self.object_descriptors.as_ref().and_then(|d| d.get(&ins.c)).cloned().ok_or(LanaError::Format)?;
            if descriptor.kind != "class" { return Err(LanaError::Type); }
            if ins.opcode == OpCode::OoSet {
                let receiver = self.current_frame().registers[ins.a as usize].clone();
                if receiver.reactive.is_some() { return Err(LanaError::UnresolvedValue); }
                let ValueKind::ClassObject(reference) = &receiver.kind else { return Err(LanaError::Type); };
                if *reference.descriptor != *descriptor { return Err(LanaError::Type); }
                let object = self.class_storage(reference)?;
                let field = descriptor.fields.get(ins.imm as usize).ok_or(LanaError::Format)?;
                if field.visibility != "public" && !self.owns_object(ins.c) { return Err(LanaError::UnsupportedOperation); }
                let constructing = self.constructing.last().is_some_and(|current| std::sync::Weak::ptr_eq(&current.object, &reference.object));
                if !self.constructing.is_empty() && !constructing { return Err(LanaError::UnsupportedOperation); }
                let initialized = object.initialized.load(Ordering::Acquire);
                if !initialized && !constructing { return Err(LanaError::UnsupportedOperation); }
                if !field.mutable && (initialized || object.fields.lock().unwrap()[ins.imm as usize].is_some()) {
                    return Err(LanaError::UnsupportedOperation);
                }
                let value = self.current_frame().registers[ins.b as usize].clone();
                self.check_class_type(&value, &field.field_type, &descriptor)?;
                // Incomplete self can be retained only inside its own candidate fields.
                if !constructing { self.check_complete_objects(&value)?; }
                if object.generation.load(Ordering::Acquire) == crate::heap::Generation::StableShared as u8 {
                    self.promote_shared_graph(&value)?;
                }
                object.set_field(ins.imm as usize, value);
                return Ok(None);
            }
            let init = descriptor.methods.iter().enumerate().find(|(_, m)| m.is_init);
            if ins.imm as usize != init.map_or(0, |(_, m)| m.parameter_types.len())
                || (init.is_some_and(|(_, m)| m.visibility != "public") && !self.owns_object(ins.c)) { return Err(LanaError::Type); }
            let _arguments = self.heap.reserve((ins.imm as usize + 1).saturating_mul(std::mem::size_of::<Value>()))?;
            let mut args = Vec::new();
            for index in 0..ins.imm as usize {
                let value = self.current_frame().registers[ins.b as usize + index].clone();
                self.check_complete_objects(&value)?;
                self.check_class_type(&value, &init.unwrap().1.parameter_types[index], &descriptor)?;
                args.push(value);
            }
            let checkpoint = self.class_objects.len();
            let candidate = self.allocate_class(descriptor.clone())?;
            self.constructing.push(candidate);
            self.constructions.push(Construction { caller_depth: self.frames.len(), checkpoint,
                instruction: *ins, return_ip: self.ip, args, next_field: 0, returned_field: None, init_started: false });
            self.resume_construction(None)?;
            Ok(None)
        })();
        match result {
            Ok(Some(value)) => { self.current_frame_mut().registers[ins.a as usize] = value; LanaError::Ok }
            Ok(None) => LanaError::Ok,
            Err(error) => error,
        }
    }

    // Defaults and init use ordinary frames, so debugger stepping follows their source lines.
    pub(super) fn resume_construction(&mut self, returned: Option<Value>) -> Result<(), LanaError> {
        let construction = self.constructions.last().ok_or(LanaError::Format)?;
        let ins = construction.instruction;
        let candidate = self.constructing.last().ok_or(LanaError::Format)?.clone();
        let descriptor = candidate.descriptor.clone();
        let storage = self.class_storage(&candidate)?;
        if let Some(index) = construction.returned_field {
            let value = returned.ok_or(LanaError::Format)?;
            self.check_class_type(&value, &descriptor.fields[index].field_type, &descriptor)?;
            self.check_complete_objects(&value)?;
            storage.set_field(index, value);
            self.constructions.last_mut().unwrap().returned_field = None;
        }
        let construction = self.constructions.last_mut().unwrap();
        let mut call = None;
        while construction.next_field < descriptor.fields.len() {
            let index = construction.next_field;
            construction.next_field += 1;
            if let Some(function) = descriptor.fields[index].default_function {
                construction.returned_field = Some(index);
                call = Some((function, (descriptor.methods.len() + index) as u32, Vec::new()));
                break;
            }
        }
        if call.is_none() && !construction.init_started {
            construction.init_started = true;
            if let Some((index, method)) = descriptor.methods.iter().enumerate().find(|(_, method)| method.is_init) {
                let mut receiver = Value::null(); receiver.kind = ValueKind::ClassObject(candidate.clone());
                let mut args = vec![receiver]; args.append(&mut construction.args);
                call = Some((method.function_index.ok_or(LanaError::Format)?, index as u32, args));
            }
        }
        if let Some((function, member, args)) = call {
            if self.frames.len() >= LANA_MAX_CALL_FRAMES as usize { return Err(LanaError::Limit); }
            let registers = self.max_registers[function as usize];
            let mut frame = Frame::charged(registers, &self.heap)?;
            frame.function = function; frame.object_method = Some((ins.c, member));
            frame.registers[..args.len()].clone_from_slice(&args);
            self.frames.push(frame);
            self.ip = self.chunk.functions[function as usize].entry as usize;
        } else {
            if storage.fields.lock().unwrap().iter().any(Option::is_none) { return Err(LanaError::Type); }
            storage.initialized.store(true, Ordering::Release);
            let construction = self.constructions.pop().unwrap();
            self.constructing.pop();
            let mut value = Value::null(); value.kind = ValueKind::ClassObject(candidate);
            self.current_frame_mut().registers[ins.a as usize] = value;
            self.ip = construction.return_ip;
        }
        Ok(())
    }

    pub(super) fn clone_class_reference(&mut self, source: &Arc<ClassReference>, memo: &mut DeepCloneMemo) -> Result<Arc<ClassReference>, LanaError> {
        if Arc::ptr_eq(&source.owner, &self.class_owner) { return Ok(source.clone()); }
        let key = source.object.as_ptr() as usize;
        if let Some(copy) = memo.classes.get(&key) { return Ok(copy.clone()); }
        let source_object = source.object.upgrade().ok_or(LanaError::Task)?;
        if !source_object.initialized.load(Ordering::Acquire) { return Err(LanaError::UnsupportedOperation); }
        let fields = source_object.fields.try_lock().map_err(|_| LanaError::UnsupportedOperation)?;
        let copy = self.allocate_class(source.descriptor.clone())?;
        memo.classes.insert(key, copy.clone());
        let count = fields.len();
        let storage = self.class_storage(&copy)?;
        let frozen = memo.freeze;
        memo.freeze = false;
        let result = (|| {
            for index in 0..count {
                let field = fields[index].clone().ok_or(LanaError::Type)?;
                let copied = self.deep_clone_value(&field, memo)?;
                storage.set_field(index, copied);
            }
            Ok::<(), LanaError>(())
        })();
        memo.freeze = frozen;
        result?;
        storage.initialized.store(true, Ordering::Release);
        Ok(copy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lana_bytecode::assembler;

    fn source(main: &str, init: &str, extra: &str) -> String {
        let descriptor = serde_json::json!({"schema_version":1,"kind":"class","qualified_name":"file/main.lana/Sensor",
            "fields":[
                {"name":"state","type":"STATE","visibility":"public","mutable":true,"default_function":null},
                {"name":"fixed","type":"number","visibility":"public","mutable":false,"default_function":2},
                {"name":"link","type":"Dynamic","visibility":"public","mutable":true,"default_function":3}],
            "methods":[{"name":"init","visibility":"public","static":false,"parameter_types":["STATE"],
                "result_type":null,"effect_mask":8,"function_index":1,"is_init":true}],"implements":[]}).to_string();
        let hex: String = descriptor.bytes().map(|b| format!("{b:02x}")).collect();
        format!(".version 6\n.function main 0 10\n{main}\n.function init 2 4\n{init}\n.function fixed 0 1\nLOAD_CONST R0 7\nRETURN R0\n.function link 0 1\nRETURN R0\n{extra}").replace("DESC", &hex)
    }

    const INIT: &str = "OO_SET R0 R1 DESC 0\nRETURN R2";
    const CREATE: &str = "STATE_NEW R0 0.5 0.3 0.4\nOBJECT_NEW R1 R0 DESC 1\n";

    fn take_result(vm: &mut Vm<'_>) -> Value {
        vm.frames.clear();
        std::mem::replace(&mut vm.result, Value::null())
    }

    #[test]
    fn rooted_class_result_survives_vm_and_releases_cycles_on_last_root() {
        let chunk = assembler::assemble(&source(&format!("{CREATE}RETURN R1"), INIT, "")).unwrap();
        let mut vm = Vm::new(&chunk);
        assert_eq!(vm.run(), LanaError::Ok);
        let root = vm.result().unwrap();
        let ValueKind::ClassObject(reference) = &root.value.kind else { panic!("class"); };
        let weak = reference.object.clone();
        let array = vm.array_value(vec![root.value.clone()]).unwrap();
        vm.class_storage(reference).unwrap().fields.lock().unwrap()[2] = Some(array.clone());
        let ValueKind::Array(items) = &array.kind else { panic!("array"); };
        let weak_array = Arc::downgrade(items);
        let alias = root.clone();
        let heap = vm.heap();
        drop(array);
        drop(vm);
        assert_eq!(root.child(1).unwrap().unwrap().as_number(), 7.0);
        drop(root);
        assert_eq!(alias.child(1).unwrap().unwrap().as_number(), 7.0);
        assert!(weak.upgrade().is_some());
        drop(alias);
        assert!(weak.upgrade().is_none());
        assert!(weak_array.upgrade().is_none());
        heap.collect_strings();
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn class_collection_preserves_external_container_aliases_and_reclaims_cycles() {
        let chunk = assembler::assemble(&source(&format!("{CREATE}RETURN R1"), INIT, "")).unwrap();
        let mut vm = Vm::new(&chunk);
        assert_eq!(vm.run(), LanaError::Ok);
        let value = take_result(&mut vm);
        let ValueKind::ClassObject(reference) = &value.kind else { panic!("class"); };
        let weak = reference.object.clone();
        let array = vm.array_value(vec![value.clone(), value.clone()]).unwrap();
        let ValueKind::Array(items) = &array.kind else { panic!("array"); };
        items.lock().unwrap().push(array.clone()).unwrap();
        vm.class_storage(reference).unwrap().fields.lock().unwrap()[2] = Some(array.clone());
        let alias = array.clone();
        drop(value);
        drop(array);
        vm.collect_classes().unwrap();
        assert!(weak.upgrade().is_some());
        assert_eq!(vm.class_objects.len(), 1);
        drop(alias);
        vm.collect_classes().unwrap();
        assert!(weak.upgrade().is_none());
        assert!(vm.class_objects.is_empty());
        let heap = vm.heap.clone();
        drop(vm);
        assert_eq!(heap.live_bytes(), 0, "class and cyclic array charges must be released");
    }

    #[test]
    fn class_collection_preserves_information_history_and_suspended_computations() {
        for holder in ["information", "generator", "future"] {
            let chunk = assembler::assemble(&source(&format!("{CREATE}RETURN R1"), INIT, "")).unwrap();
            let mut vm = Vm::new(&chunk);
            assert_eq!(vm.run(), LanaError::Ok);
            let value = take_result(&mut vm);
            let ValueKind::ClassObject(reference) = &value.kind else { panic!("class"); };
            let weak = reference.object.clone();
            let mut root = Value::null();
            match holder {
                "information" => {
                    root = vm.information_root(&Value::number(1.0)).unwrap();
                    let mut reactive = root.reactive.as_ref().unwrap().lock().unwrap();
                    reactive.history.push(ReactiveVersion { revision: 0, value: Some(value.clone()) });
                }
                "generator" => root.kind = ValueKind::Generator(Arc::new(Mutex::new(crate::value::Generator {
                    function: 0, ip: 0, registers: vec![value.clone()], exhausted: false,
                }))),
                _ => root.kind = ValueKind::Future(Arc::new(Mutex::new(Future {
                    function: 0, ip: 0, registers: vec![value.clone()], exhausted: false, ready: false, queued: false,
                }))),
            }
            vm.class_storage(reference).unwrap().fields.lock().unwrap()[2] = Some(root.clone());
            drop(value);
            vm.collect_classes().unwrap();
            assert!(weak.upgrade().is_some(), "{holder}");
            drop(root);
            vm.collect_classes().unwrap();
            assert!(weak.upgrade().is_none(), "{holder}");
            assert!(vm.class_objects.is_empty());
        }
    }

    #[test]
    fn class_collection_without_headroom_preserves_limit_and_cancellation() {
        let chunk = assembler::assemble(&source(&format!("{CREATE}OO_SET R1 R1 DESC 2\nRETURN R1"), INIT, "")).unwrap();
        let mut vm = Vm::new(&chunk);
        assert_eq!(vm.run(), LanaError::Ok);
        let value = take_result(&mut vm);
        let ValueKind::ClassObject(reference) = &value.kind else { panic!("class"); };
        let weak = reference.object.clone();
        drop(value);
        let bytes = vm.allocated_bytes();
        vm.set_instruction_limit(vm.instruction_count + 5);
        assert_eq!(vm.collect_classes(), Err(LanaError::Limit));
        assert_eq!(vm.allocated_bytes(), bytes);
        vm.set_instruction_limit(50_000_000);
        vm.cancelled.store(true, Ordering::Relaxed);
        assert_eq!(vm.collect_classes(), Err(LanaError::Cancelled));
        vm.cancelled.store(false, Ordering::Relaxed);
        assert_eq!(weak.upgrade().unwrap().fields.lock().unwrap().len(), 3);
        vm.set_memory_limit(vm.allocated_bytes()).unwrap();
        let allocations = vm.heap.allocations();
        vm.collect_classes().unwrap();
        assert_eq!(vm.heap.allocations(), allocations);
        assert!(weak.upgrade().is_none());
        assert!(vm.class_objects.is_empty());
    }

    #[test]
    fn class_collection_defers_locked_graphs_and_constructor_transactions() {
        let chunk = assembler::assemble(&source(&format!("{CREATE}RETURN R1"), INIT, "")).unwrap();
        let mut vm = Vm::new(&chunk);
        let line = chunk.code[chunk.functions[2].entry as usize].line;
        vm.set_breakpoint_line(line);
        // Step directly to the first suspended default frame.
        while vm.constructions.is_empty() { assert_eq!(vm.debug_step(), LanaError::Ok); }
        vm.collect_classes().unwrap();
        assert_eq!(vm.class_objects.len(), 1);
        assert_eq!(vm.debug_continue(), LanaError::Ok);
        let value = take_result(&mut vm);
        let ValueKind::ClassObject(reference) = &value.kind else { panic!("class"); };
        let storage = vm.class_storage(reference).unwrap();
        let fields = storage.fields.lock().unwrap();
        vm.collect_classes().unwrap();
        assert_eq!(fields.len(), 3);
        drop(fields);
        drop(storage);
        drop(value);
        vm.collect_classes().unwrap();
        assert!(vm.class_objects.is_empty());
    }

    #[test]
    fn class_collection_reuses_small_heap_across_returned_constructor_frames() {
        let main = format!("STATE_NEW R0 0.5 0.3 0.4\n{}RETURN R1",
            "OBJECT_NEW R1 R0 DESC 1\nOO_SET R1 R1 DESC 2\n".repeat(1000));
        let chunk = assembler::assemble(&source(&main, INIT, "")).unwrap();
        let mut vm = Vm::new(&chunk);
        vm.set_memory_limit(64 * 1024).unwrap();
        assert_eq!(vm.run(), LanaError::Ok, "{:?}", vm.error());
        assert!(vm.class_objects.len() < 65);
        assert!(vm.allocated_bytes() < 64 * 1024);
        let value = take_result(&mut vm);
        vm.collect_classes().unwrap();
        assert_eq!(vm.class_objects.len(), 1);
        drop(value);
        vm.collect_classes().unwrap();
        assert!(vm.class_objects.is_empty());
    }

    #[test]
    fn class_collection_rechecks_dropped_roots_before_allocation_under_pressure() {
        let chunk = assembler::assemble(&source(&format!("{CREATE}RETURN R1"), INIT, "")).unwrap();
        let mut vm = Vm::new(&chunk);
        assert_eq!(vm.run(), LanaError::Ok);
        let value = vm.result.clone();
        let ValueKind::ClassObject(reference) = &value.kind else { panic!("class"); };
        let weak = reference.object.clone();
        let payload = vm.array_value(vec![Value::number(1.0); 64]).unwrap();
        vm.class_storage(reference).unwrap().fields.lock().unwrap()[2] = Some(payload);
        vm.set_memory_limit(vm.allocated_bytes() + 1024).unwrap();
        vm.collect_classes().unwrap();
        assert_eq!(vm.class_allocations_since_gc, 0);
        vm.result = Value::null();
        vm.frames[0].registers.fill(Value::null());
        vm.frames[0].registers[0] = Value::state(StateValue::default());
        drop(value);
        let index = chunk.code.iter().position(|ins| ins.opcode == OpCode::ObjectNew).unwrap();
        vm.ip = index + 1;
        assert_eq!(vm.execute(&chunk.code[index]), LanaError::Ok);
        assert!(weak.upgrade().is_none(), "the previous live set became garbage before this allocation");
        assert_eq!(vm.debug_continue(), LanaError::Ok);
    }

    #[test]
    fn retained_task_child_keeps_its_original_class_after_join() {
        let main = format!("{CREATE}RETURN R1");
        let chunk = assembler::assemble(&source(&main, INIT, ".function echo 1 4\nRETURN R0\n")).unwrap();
        let mut vm = Vm::new(&chunk);
        vm.configured_worker_count = 0;
        assert_eq!(vm.run(), LanaError::Ok);
        vm.scheduler = Some(Scheduler::new());
        let task = vm.start_task(4, 1, 1).unwrap();
        let task_root = vm.retain_value(&Value::task(task.clone())).unwrap();
        drop(task);
        let ValueKind::Task(task) = &task_root.value.kind else { unreachable!() };
        let queued = vm.scheduler.as_ref().unwrap().state.lock().unwrap().queue.pop_front().unwrap();
        run_task(queued);
        let child = task_root.child(0).unwrap().unwrap();
        let ValueKind::ClassObject(original) = &child.value.kind else { unreachable!() };
        let original_storage = original.object.clone();
        let joined = vm.wait_task(task, -1.0).unwrap();
        let ValueKind::ClassObject(copy) = joined.kind else { unreachable!() };
        assert!(!Arc::ptr_eq(original, &copy));
        drop(copy);
        drop(vm);
        drop(task_root);
        assert!(original_storage.upgrade().is_some());
        assert_eq!(child.child(1).unwrap().unwrap().as_number(), 7.0);
        drop(child);
        assert!(original_storage.upgrade().is_none());
    }

    #[test]
    fn task_result_lease_keeps_child_class_storage_after_handle_drop() {
        let chunk = assembler::assemble(&source(&format!("{CREATE}RETURN R1"), INIT, "")).unwrap();
        let handle = Arc::new(Task::new(1, 0));
        let mut child = Vm::new(&chunk);
        child.configured_worker_count = 0;
        let heap = child.heap();
        run_task(QueuedTask { child, handle: handle.clone() });
        let root = handle.state.lock().unwrap().result_root.clone().unwrap();
        let ValueKind::ClassObject(reference) = &root.value.kind else { panic!("class"); };
        let storage = reference.object.clone();
        drop(handle);
        assert!(storage.upgrade().is_some());
        assert_eq!(root.child(1).unwrap().unwrap().as_number(), 7.0);
        drop(root);
        assert!(storage.upgrade().is_none());
        assert_eq!(heap.live_bytes(), 0);
    }

    #[test]
    fn class_collection_preserves_task_transfer_and_joined_results() {
        let noise = "OBJECT_NEW R3 R2 DESC 1\nOO_SET R3 R3 DESC 2\n".repeat(100);
        let main = format!("{CREATE}OO_SET R1 R1 DESC 2\nFORK echo R1 1 R5\nJOIN R5 R6\nSTATE_NEW R2 0.5 0.3 0.4\n{noise}OO_GET R4 R6 DESC 2\nCOMPARE R4 == R6 R7\nRETURN R7");
        let extra = format!(".function echo 1 10\nSTATE_NEW R2 0.5 0.3 0.4\n{noise}RETURN R0\n");
        let chunk = assembler::assemble(&source(&main, INIT, &extra)).unwrap();
        for workers in [0, 2] {
            let mut vm = Vm::new(&chunk);
            vm.configured_worker_count = workers;
            vm.set_memory_limit(128 * 1024).unwrap();
            assert_eq!(vm.run(), LanaError::Ok, "{:?}", vm.error());
            assert!(vm.result.as_bool());
        }
    }

    #[test]
    fn debugger_stops_in_defaults_and_init_without_publishing_candidate() {
        for (function, name) in [(1, "Sensor.init"), (2, "Sensor.<default:fixed>")] {
            let mut chunk = assembler::assemble(&source(&format!("{CREATE}RETURN R1"), INIT, "")).unwrap();
            let entry = chunk.functions[function].entry as usize;
            chunk.code[entry].line = 999;
            let mut vm = Vm::new(&chunk);
            vm.set_breakpoint_line(999);
            assert_eq!(vm.run(), LanaError::Ok);
            assert_eq!(vm.debug_location().unwrap().2, name);
            assert!(matches!(vm.frames[0].registers[1].kind, ValueKind::Null));
            assert_eq!(vm.debug_continue(), LanaError::Ok);
            assert!(matches!(vm.result.kind, ValueKind::ClassObject(_)));
        }
    }

    #[test]
    fn classes_construct_defaults_identity_mutation_and_cycles() {
        let chunk = assembler::assemble(&source(&format!("{CREATE}OO_SET R1 R1 DESC 2\nOO_GET R2 R1 DESC 2\nCOMPARE R1 == R2 R3\nRETURN R3"), INIT, "")).unwrap();
        let mut vm = Vm::new(&chunk);
        assert_eq!(vm.run(), LanaError::Ok);
        assert!(vm.result.as_bool());
        let ValueKind::ClassObject(reference) = &vm.frames[0].registers[1].kind else { panic!("class"); };
        let storage = vm.class_storage(reference).unwrap();
        assert_eq!(storage.fields.lock().unwrap()[1].as_ref().unwrap().as_number(), 7.0);
        assert_eq!(storage.fields.lock().unwrap()[0].as_ref().unwrap().as_state().state.d_im, 0.4);
        assert_eq!(vm.observation_count, 0);
        let weak = reference.object.clone();
        drop(storage);
        drop(vm);
        assert!(weak.upgrade().is_none(), "class cycle must not retain task storage");
    }

    #[test]
    fn classes_initialization_failure_does_not_publish_or_escape() {
        for (init, expected) in [
            ("RETURN R2", LanaError::Type),
            ("OO_SET R0 R1 DESC 0\nOO_SET R0 R1 DESC 1\nRETURN R2", LanaError::UnsupportedOperation),
            ("CALL escape R0 1 R3\nRETURN R2", LanaError::UnsupportedOperation),
            ("OO_SET R0 R1 DESC 0\nRETURN R0", LanaError::Type),
        ] {
            let program = source(&format!("{CREATE}RETURN R1"), init, ".function escape 1 1\nRETURN R0\n");
            let chunk = assembler::assemble(&program).unwrap();
            let mut vm = Vm::new(&chunk);
            vm.frames[0].registers[1] = Value::number(77.0);
            assert_eq!(vm.run(), expected, "{init}");
            assert_eq!(vm.frames[0].registers[1].as_number(), 77.0);
            assert!(matches!(vm.result.kind, ValueKind::Null));
            assert!(vm.class_objects.is_empty());
            assert!(vm.constructing.is_empty());
        }
    }

    #[test]
    fn classes_fork_join_clones_identity_and_preserves_cycle() {
        let program = source(&format!("{CREATE}OO_SET R1 R1 DESC 2\nFORK echo R1 1 R2\nJOIN R2 R3\nOO_GET R4 R3 DESC 2\nCOMPARE R3 == R4 R5\nCOMPARE R1 == R3 R6\nARRAY_NEW R7 R5 2\nRETURN R7"), INIT,
            ".function echo 1 1\nRETURN R0\n");
        let chunk = assembler::assemble(&program).unwrap();
        let mut vm = Vm::new(&chunk);
        assert_eq!(vm.run(), LanaError::Ok);
        assert_eq!(vm.result.print(), "[true, false]");
    }

    #[test]
    fn classes_transfer_captures_live_fields_and_keeps_aliases() {
        let chunk = assembler::assemble(&source(&format!("{CREATE}RETURN R1"), INIT, "")).unwrap();
        let mut parent = Vm::new(&chunk);
        assert_eq!(parent.run(), LanaError::Ok);
        let original = parent.result.clone();
        let ValueKind::ClassObject(reference) = &original.kind else { panic!("class"); };
        let storage = parent.class_storage(reference).unwrap();
        let root = parent.information_root(&Value::number(7.0)).unwrap();
        storage.fields.lock().unwrap()[2] = Some(root);
        let aliases = parent.array_value(vec![original.clone(), original]).unwrap();
        let mut child = Vm::new(&chunk);
        let transferred = child.deep_clone_value(&aliases, &mut DeepCloneMemo { transfer:true, ..DeepCloneMemo::default() }).unwrap();
        let ValueKind::Array(array) = transferred.kind else { panic!("array"); };
        let array = array.lock().unwrap();
        let (ValueKind::ClassObject(a), ValueKind::ClassObject(b)) = (&array.items[0].kind, &array.items[1].kind) else { panic!("class"); };
        assert!(std::sync::Weak::ptr_eq(&a.object, &b.object));
        let captured = child.class_storage(a).unwrap().fields.lock().unwrap()[2].clone().unwrap();
        assert!(captured.reactive.is_none());
        assert_eq!(captured.as_number(), 7.0);
        assert_eq!(captured.derivation.as_ref().unwrap().operation.as_ref(), "snapshot");
    }
}
