//! Serialized value types, mirroring `ValueType` in `vm/include/value.h`.
//!
//! Only the constant-serializable types (`Null`, `Number`, `Bool`, `String`)
//! appear in LABC constant pools. The remaining discriminants are reserved to
//! keep the numeric values aligned with the C11 enum.

/// Value types. The discriminants match `ValueType` in `vm/include/value.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ValueType {
    Null = 0,
    Number,
    Bool,
    String,
    State,
    Distribution,
    Sample,
    JointState,
    Array,
    Function,
    Task,
    StateDist,
    Map,
    Possibility,
    PathSet,
    SharedCapability,
    Adt,
    Lazy,
}

impl TryFrom<u8> for ValueType {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if value <= Self::Lazy as u8 {
            // SAFETY: every discriminant up to Lazy is valid.
            Ok(unsafe { core::mem::transmute::<u8, ValueType>(value) })
        } else {
            Err(())
        }
    }
}

/// A constant-pool value. Only the serializable variants are constructible.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Number(f64),
    Bool(bool),
    String(String),
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::Null => ValueType::Null,
            Value::Number(_) => ValueType::Number,
            Value::Bool(_) => ValueType::Bool,
            Value::String(_) => ValueType::String,
        }
    }
}
