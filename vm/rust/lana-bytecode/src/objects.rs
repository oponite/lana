//! LABC v6 descriptor and object-instruction verification.
use std::collections::{BTreeMap, HashMap, HashSet};
use serde::{Deserialize, Serialize};
use crate::{Chunk, Value, OpCode, LanaError, LanaErrorInfo};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Descriptor {
    pub schema_version: u32,
    pub kind: String,
    pub qualified_name: String,
    pub fields: Vec<Field>,
    pub methods: Vec<Method>,
    pub implements: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Field {
    pub name: String,
    #[serde(rename = "type")]
    pub field_type: String,
    pub visibility: String,
    pub mutable: bool,
    pub default_function: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Method {
    pub name: String,
    pub visibility: String,
    #[serde(rename = "static")]
    pub is_static: bool,
    pub parameter_types: Vec<String>,
    pub result_type: Option<String>,
    pub effect_mask: u32,
    pub function_index: Option<u32>,
    pub is_init: bool,
}

fn invalid(message: impl Into<String>) -> LanaErrorInfo {
    LanaErrorInfo::new(LanaError::Format, 0, 0, 0, message)
}

fn identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn qualified(name: &str) -> bool {
    let Some((module, name)) = name.rsplit_once('/') else { return false; };
    if !identifier(name) || module.contains(['\\', '\0', ':']) { return false; }
    let Some((prefix, path)) = module.split_once('/') else { return false; };
    if !matches!(prefix, "file" | "project" | "std" | "pkg") || path.is_empty() { return false; }
    let mut path_started = false;
    for component in path.split('/') {
        if component.is_empty() || component == "." { return false; }
        if component == ".." {
            if !matches!(prefix, "file" | "project") || path_started { return false; }
        } else { path_started = true; }
    }
    path_started && (prefix != "pkg" || path.split('/').count() >= 3)
}

impl Descriptor {
    pub fn parse(text: &str) -> Result<Self, LanaErrorInfo> {
        let json: serde_json::Value = serde_json::from_str(text).map_err(|_| invalid("invalid object descriptor JSON"))?;
        // Re-encoding rejects duplicate keys, noncanonical escapes and whitespace.
        if serde_json::to_string(&json).map_err(|_| invalid("invalid descriptor"))? != text {
            return Err(invalid("object descriptor is not canonical JSON"));
        }
        if json.as_object().is_none_or(|v| v.len() != 6)
            || json.get("fields").and_then(|v| v.as_array()).is_none_or(|fields| fields.iter().any(|v| v.as_object().is_none_or(|v| v.len() != 5)))
            || json.get("methods").and_then(|v| v.as_array()).is_none_or(|methods| methods.iter().any(|v| v.as_object().is_none_or(|v| v.len() != 8))) {
            return Err(invalid("missing object descriptor fields"));
        }
        let descriptor: Self = serde_json::from_value(json).map_err(|_| invalid("invalid object descriptor schema"))?;
        if descriptor.schema_version != 1 || !matches!(descriptor.kind.as_str(), "value" | "class" | "interface") || !qualified(&descriptor.qualified_name) {
            return Err(invalid("invalid object descriptor identity"));
        }
        Ok(descriptor)
    }
}

/// Validated descriptors keyed by constant index. No runtime pointers enter the ABI.
pub fn descriptors(chunk: &Chunk) -> Result<BTreeMap<u32, Descriptor>, LanaErrorInfo> {
    let mut result = BTreeMap::new();
    for (index, constant) in chunk.constants.iter().enumerate() {
        let Value::String(text) = constant else { continue; };
        if serde_json::from_str::<serde_json::Value>(text).ok().is_some_and(|value|
            value.get("schema_version").is_some() && value.get("qualified_name").is_some()
                && value.get("kind").and_then(|kind| kind.as_str()).is_some_and(|kind| matches!(kind, "value" | "class" | "interface"))) {
            result.insert(index as u32, Descriptor::parse(text)?);
        }
    }
    Ok(result)
}

fn valid_type(name: &str, owner: &Descriptor, names: &HashMap<&str, u32>, depth: usize) -> bool {
    if depth > 32 { return false; }
    if let Some(inner) = name.strip_prefix("Information<").and_then(|text| text.strip_suffix('>')) {
        return valid_type(inner, owner, names, depth + 1);
    }
    matches!(name, "number" | "bool" | "string" | "null" | "STATE" | "STATE_DIST" | "array" | "map" | "Tensor" | "Shape" | "Dynamic" | "Self")
        || names.contains_key(name)
        || names.contains_key(format!("{}/{}", owner.qualified_name.rsplit_once('/').unwrap().0, name).as_str())
}

fn overload_type(name: &str) -> bool {
    matches!(name, "STATE" | "STATE_DIST") || name.starts_with("Information<")
}

pub fn verify(chunk: &Chunk) -> Result<(), LanaErrorInfo> {
    use OpCode::*;
    let descriptors = descriptors(chunk)?;
    let mut names = HashMap::new();
    for (index, descriptor) in &descriptors {
        if names.insert(descriptor.qualified_name.as_str(), *index).is_some() { return Err(invalid("duplicate object type identity")); }
    }
    let mut owners = vec![None; chunk.functions.len()];
    let mut own = |function: u32, owner: u32, arity: usize| -> Result<(), LanaErrorInfo> {
        let function_metadata = chunk.functions.get(function as usize).ok_or_else(|| invalid("missing object member function"))?;
        if function_metadata.arity as usize != arity || owners[function as usize].is_some() {
            return Err(invalid("object member has wrong arity or multiple owners"));
        }
        owners[function as usize] = Some(owner);
        Ok(())
    };
    for (index, descriptor) in &descriptors {
        if descriptor.kind == "interface" && (!descriptor.fields.is_empty() || !descriptor.implements.is_empty()) { return Err(invalid("interface has fields or inheritance")); }
        let mut fields = HashSet::new();
        for field in &descriptor.fields {
            if !identifier(&field.name) || !fields.insert(&field.name)
                || !matches!(field.visibility.as_str(), "public" | "private")
                || !valid_type(&field.field_type, descriptor, &names, 0)
                || (descriptor.kind == "value" && (field.mutable || field.default_function.is_some())) {
                return Err(invalid("invalid object field"));
            }
            if let Some(function) = field.default_function { own(function, *index, 0)?; }
        }
        let mut init = false;
        for (position, method) in descriptor.methods.iter().enumerate() {
            if !identifier(&method.name) || fields.contains(&method.name)
                || !matches!(method.visibility.as_str(), "public" | "private") || method.effect_mask > 63
                || method.parameter_types.iter().any(|name| !valid_type(name, descriptor, &names, 0))
                || (method.is_init != (method.name == "init"))
                || (method.is_init && (descriptor.kind != "class" || method.is_static || method.result_type.is_some() || init))
                || (!method.is_init && method.result_type.as_ref().is_none_or(|name| !valid_type(name, descriptor, &names, 0))) {
                return Err(invalid("invalid object method signature"));
            }
            init |= method.is_init;
            if descriptor.kind == "interface" {
                if method.visibility != "public" || method.is_static || method.function_index.is_some() { return Err(invalid("invalid interface promise")); }
            } else {
                own(method.function_index.ok_or_else(|| invalid("missing method body"))?, *index,
                    method.parameter_types.len() + usize::from(!method.is_static))?;
            }
            for prior in &descriptor.methods[..position] {
                if prior.name != method.name { continue; }
                if method.is_static || prior.is_static || method.parameter_types == prior.parameter_types
                    || method.parameter_types.len() != prior.parameter_types.len()
                    || method.parameter_types.iter().zip(&prior.parameter_types).any(|(a,b)| a != b && !(overload_type(a) && overload_type(b))) {
                    return Err(invalid("invalid object overload"));
                }
            }
        }
        if descriptor.kind == "class" && !init && descriptor.fields.iter().any(|field| field.default_function.is_none()) {
            return Err(invalid("class without init has an unset field"));
        }
        let mut implemented = HashSet::new();
        for name in &descriptor.implements {
            let interface = names.get(name.as_str()).and_then(|index| descriptors.get(index)).ok_or_else(|| invalid("missing implemented interface"))?;
            if !implemented.insert(name) || interface.kind != "interface" { return Err(invalid("invalid implemented interface")); }
            for promise in &interface.methods {
                let implementation = descriptor.methods.iter().find(|method| !method.is_static && method.visibility == "public"
                    && method.name == promise.name && method.parameter_types == promise.parameter_types
                    && method.result_type == promise.result_type).ok_or_else(|| invalid("missing interface implementation"))?;
                if implementation.effect_mask & !promise.effect_mask != 0 { return Err(invalid("interface effect promise exceeded")); }
            }
        }
    }
    let mut functions: Vec<_> = chunk.functions.iter().enumerate().collect();
    functions.sort_by_key(|(_, function)| function.entry);
    if functions.first().is_none_or(|(_, function)| function.entry != 0)
        || functions.windows(2).any(|pair| pair[0].1.entry == pair[1].1.entry) {
        return Err(invalid("v6 requires distinct function entries covering its instructions"));
    }
    if !functions.iter().any(|(index, function)| function.entry == chunk.entry && owners[*index].is_none()) { return Err(invalid("object body cannot be the entry point")); }
    for (position, (function_index, function)) in functions.iter().enumerate() {
        let start = function.entry as usize;
        let end = functions.get(position + 1).map_or(chunk.code.len(), |(_, function)| function.entry as usize);
        let owner = owners[*function_index];
        if !matches!(chunk.code[end - 1].opcode, Return | Halt | Jump) { return Err(invalid("v6 function may fall through into another body")); }
        for ip in start..end {
            let instruction = &chunk.code[ip];
            let check = || -> Result<(), LanaErrorInfo> {
                let direct = match instruction.opcode {
                    Call | Generator | Async | LoadFunction | Fork | Bootstrap | Lazy => Some(instruction.b),
                    _ => None,
                };
                if direct.is_some_and(|target| owners.get(target as usize).is_some_and(Option::is_some)) { return Err(invalid("direct call/reference to an object member")); }
                if matches!(instruction.opcode, Jump | JumpIfTrue | JumpIfFalse | PathSplit | AdtCase)
                    && !(start..end).contains(&(instruction.imm as usize)) { return Err(invalid("jump crosses v6 function ownership")); }
                if (instruction.opcode as u8) < ValueNew as u8 { return Ok(()); }
                let descriptor = descriptors.get(&instruction.c).ok_or_else(|| invalid("missing object descriptor constant"))?;
                let own = owner == Some(instruction.c);
                let register = |register| if register < function.register_count { Ok(()) } else { Err(invalid("object register outside function")) };
                register(instruction.a)?;
                match instruction.opcode {
                    ValueNew | ObjectNew => {
                        if instruction.imm == 0 { if instruction.b != u32::MAX { return Err(invalid("empty constructor requires unused argument register")); } }
                        else if instruction.b.checked_add(instruction.imm).is_none_or(|end| end > function.register_count) { return Err(invalid("constructor arguments outside function")); }
                        if instruction.opcode == ValueNew {
                            if descriptor.kind != "value" || instruction.imm as usize != descriptor.fields.len()
                                || (!own && descriptor.fields.iter().any(|field| field.visibility != "public")) { return Err(invalid("invalid or private value constructor")); }
                        } else {
                            let init = descriptor.methods.iter().find(|method| method.is_init);
                            if descriptor.kind != "class" || instruction.imm as usize != init.map_or(0, |method| method.parameter_types.len())
                                || (!own && init.is_some_and(|method| method.visibility != "public")) { return Err(invalid("invalid or private object initializer")); }
                        }
                    }
                    OoGet | OoSet => {
                        register(instruction.b)?;
                        let field = descriptor.fields.get(instruction.imm as usize).ok_or_else(|| invalid("invalid object field index"))?;
                        let initializing = own && descriptor.methods.iter().any(|method| method.is_init && method.function_index == Some(*function_index as u32));
                        if (!own && field.visibility != "public") || (instruction.opcode == OoSet && (descriptor.kind != "class" || (!field.mutable && !initializing))) { return Err(invalid("illegal object field access")); }
                    }
                    OoCall | OoStaticCall => {
                        register(instruction.b)?;
                        let method = descriptor.methods.get(instruction.imm as usize).ok_or_else(|| invalid("invalid object method index"))?;
                        if method.is_init || method.is_static != (instruction.opcode == OoStaticCall) || (!own && method.visibility != "public") { return Err(invalid("illegal object method access")); }
                    }
                    OoAsInterface => {
                        register(instruction.b)?;
                        if descriptor.kind != "interface" || instruction.imm != u32::MAX { return Err(invalid("invalid interface conversion")); }
                    }
                    _ => return Err(invalid("invalid object opcode")),
                }
                Ok(())
            };
            check().map_err(|mut error| { error.ip = ip; error.opcode = instruction.opcode as u8; error.line = instruction.line; error })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Function, Instruction, assembler, encoder, loader, verifier};
    use serde_json::json;

    fn encode(value: serde_json::Value) -> String { serde_json::to_string(&value).unwrap() }
    fn value() -> serde_json::Value {
        json!({"schema_version":1,"kind":"value","qualified_name":"file/main.lana/Reading",
            "fields":[{"name":"state","type":"STATE","visibility":"public","mutable":false,"default_function":null}],
            "methods":[],"implements":[]})
    }
    fn chunk(descriptor: serde_json::Value) -> Chunk {
        let mut chunk = Chunk::new(6, 0);
        chunk.constants.push(Value::String(encode(descriptor)));
        chunk.functions.push(Function { name:"main".into(), entry:0, register_count:3, arity:0 });
        chunk.code = vec![Instruction::new(OpCode::ValueNew, 1, 0, 0, 1, 8),
            Instruction::new(OpCode::OoGet, 2, 1, 0, 0, 9), Instruction::new(OpCode::Halt, 0, 0, 0, 0, 10)];
        chunk
    }
    fn method(function: Option<u32>) -> serde_json::Value {
        json!({"name":"read","visibility":"public","static":false,"parameter_types":[],
            "result_type":"STATE","effect_mask":0,"function_index":function,"is_init":false})
    }
    fn method_chunk() -> Chunk {
        let mut descriptor = value();
        descriptor["methods"] = json!([method(Some(1))]);
        let mut chunk = chunk(descriptor);
        chunk.functions.push(Function { name:"Reading.read".into(), entry:3, register_count:2, arity:1 });
        chunk.code.extend([Instruction::new(OpCode::OoGet, 1, 0, 0, 0, 12), Instruction::new(OpCode::Return, 1, 0, 0, 0, 13)]);
        chunk
    }

    #[test]
    fn v6_descriptors_roundtrip_and_old_versions_reject_object_opcodes() {
        let chunk = chunk(value());
        assert_eq!(OpCode::ValueNew as u8, 83);
        assert_eq!(OpCode::OoAsInterface as u8, 89);
        verifier::verify(&chunk).unwrap();
        assert_eq!(loader::load(&encoder::encode(&chunk)).unwrap(), chunk);
        for version in 1..=5 {
            let mut old = chunk.clone(); old.version = version;
            assert_eq!(loader::load(&encoder::encode(&old)).unwrap_err().code, LanaError::Opcode);
        }
        let descriptor = encode(value()).bytes().map(|b| format!("{b:02x}")).collect::<String>();
        let assembly = format!(".version 6\n.function main 0 3\nVALUE_NEW R1 R0 {descriptor} 1\nOO_GET R2 R1 {descriptor} 0\nHALT\n");
        let assembled = assembler::assemble(&assembly).unwrap();
        assert_eq!(assembled.constants.len(), 1);
        assert!(crate::disassembler::disassemble(&assembled).contains("descriptor=K0"));
        let mut bytes = encoder::encode(&chunk);
        bytes[29] = 0xff; // first byte of the descriptor string
        assert_eq!(loader::load(&bytes).unwrap_err().code, LanaError::Format);
    }

    #[test]
    fn v6_rejects_malformed_descriptors_and_illegal_field_access() {
        let valid = encode(value());
        for text in [format!(" {valid}"), valid.replace("\"schema_version\":1", "\"schema_version\":1,\"schema_version\":1"), valid.replace("\"default_function\":null,", "")] {
            assert!(Descriptor::parse(&text).is_err());
        }
        for (key, bad) in [("mutable", json!(true)), ("visibility", json!("protected")), ("type", json!("Missing")), ("default_function", json!(0))] {
            let mut descriptor = value(); descriptor["fields"][0][key] = bad;
            assert!(verifier::verify(&chunk(descriptor)).is_err());
        }
        let mut descriptor = value(); descriptor["fields"][0]["visibility"] = json!("private");
        assert!(verifier::verify(&chunk(descriptor)).is_err());
        for (operand, number) in [("a", 3), ("b", u32::MAX), ("c", 1), ("imm", 2)] {
            let mut bad = chunk(value());
            match operand { "a" => bad.code[0].a = number, "b" => bad.code[0].b = number,
                "c" => bad.code[0].c = number, _ => bad.code[0].imm = number };
            assert_eq!(verifier::verify(&bad).unwrap_err().line, 8);
        }
        let mut bad = chunk(value()); bad.code[1].opcode = OpCode::OoSet;
        assert!(verifier::verify(&bad).is_err());
        let mut bad = chunk(value()); bad.constants.push(bad.constants[0].clone());
        assert!(verifier::verify(&bad).is_err());
    }

    #[test]
    fn v6_checks_initializers_static_calls_and_interface_operands() {
        let mut descriptor = value();
        descriptor["kind"] = json!("class");
        let mut init = method(Some(1));
        init["name"] = json!("init"); init["is_init"] = json!(true);
        init["parameter_types"] = json!(["STATE"]); init["result_type"] = json!(null);
        descriptor["methods"] = json!([init]);
        let mut class = chunk(descriptor.clone());
        class.code[0].opcode = OpCode::ObjectNew;
        class.functions.push(Function { name:"Reading.init".into(), entry:3, register_count:3, arity:2 });
        class.constants.push(Value::Null);
        class.code.extend([Instruction::new(OpCode::OoSet, 0, 1, 0, 0, 12),
            Instruction::new(OpCode::LoadConst, 2, 0, 0, 1, 13), Instruction::new(OpCode::Return, 2, 0, 0, 0, 14)]);
        verifier::verify(&class).unwrap();
        descriptor["methods"][0]["visibility"] = json!("private");
        class.constants[0] = Value::String(encode(descriptor));
        assert!(verifier::verify(&class).is_err());
        let mut implicit = value(); implicit["kind"] = json!("class"); implicit["fields"] = json!([]);
        let mut class = chunk(implicit);
        class.code[0] = Instruction::new(OpCode::ObjectNew, 0, u32::MAX, 0, 0, 1);
        class.code[1] = Instruction::new(OpCode::Halt, 0, 0, 0, 0, 2);
        verifier::verify(&class).unwrap();
        let mut static_chunk = method_chunk();
        let mut descriptor = value(); let mut static_method = method(Some(1)); static_method["static"] = json!(true);
        descriptor["methods"] = json!([static_method]);
        static_chunk.constants[0] = Value::String(encode(descriptor));
        static_chunk.functions[1].arity = 0;
        static_chunk.code[0] = Instruction::new(OpCode::OoStaticCall, 1, 0, 0, 0, 1);
        verifier::verify(&static_chunk).unwrap();
        static_chunk.code[0].opcode = OpCode::OoCall;
        assert!(verifier::verify(&static_chunk).is_err());
        let interface = json!({"schema_version":1,"kind":"interface","qualified_name":"file/main.lana/Readable","fields":[],"methods":[],"implements":[]});
        let mut converted = chunk(interface);
        converted.code[0] = Instruction::new(OpCode::OoAsInterface, 1, 0, 0, u32::MAX, 1);
        converted.code[1] = Instruction::new(OpCode::Halt, 0, 0, 0, 0, 2);
        verifier::verify(&converted).unwrap();
        converted.code[0].imm = 0;
        assert!(verifier::verify(&converted).is_err());
    }

    #[test]
    fn v6_owns_methods_and_checks_interface_signatures_and_effects() {
        let valid = method_chunk();
        verifier::verify(&valid).unwrap();
        for opcode in [OpCode::Call, OpCode::LoadFunction, OpCode::Generator, OpCode::Async, OpCode::Fork, OpCode::Bootstrap, OpCode::Lazy] {
            let mut bad = valid.clone(); bad.code[0] = Instruction::new(opcode, 0, 1, 0, if matches!(opcode, OpCode::LoadFunction | OpCode::Lazy) { 0 } else { 1 }, 20);
            assert!(verifier::verify(&bad).is_err(), "{opcode:?}");
        }
        let mut bad = valid.clone(); bad.code[0] = Instruction::new(OpCode::Jump, 0, 0, 0, 3, 20);
        assert!(verifier::verify(&bad).is_err());
        let mut bad = valid.clone(); bad.constants.push(Value::Number(0.));
        bad.code[0] = Instruction::new(OpCode::AdtCase, 0, 1, 0, 3, 20);
        assert!(verifier::verify(&bad).is_err());
        let mut bad = valid.clone(); bad.entry = 3;
        assert!(verifier::verify(&bad).is_err());
        let mut bad = valid.clone(); bad.functions[1].arity = 0;
        assert!(verifier::verify(&bad).is_err());
        let mut implemented = valid.clone();
        let interface = json!({"schema_version":1,"kind":"interface","qualified_name":"file/main.lana/Readable","fields":[],"methods":[method(None)],"implements":[]});
        implemented.constants.push(Value::String(encode(interface)));
        let mut descriptor = value(); descriptor["methods"] = json!([method(Some(1))]);
        descriptor["implements"] = json!(["file/main.lana/Readable"]);
        implemented.constants[0] = Value::String(encode(descriptor.clone()));
        verifier::verify(&implemented).unwrap();
        descriptor["methods"][0]["effect_mask"] = json!(8);
        implemented.constants[0] = Value::String(encode(descriptor));
        assert!(verifier::verify(&implemented).is_err());
    }
}
