//! Relationship claims and resolution, mirroring `runtime/c/claims.c` and
//! `runtime/include/claims.h`.
//!
//! The payload encoding, digest, and Ed25519 signature verification are
//! byte-identical to the C11 reference so that differential conformance can
//! compare the two byte-for-byte.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use ed25519_compact::{PublicKey, Signature};
use lana_bytecode::LanaError;
use lana_vm::value::RelationshipKind;

use crate::sha256::sha256;

pub const RELATIONSHIP_CLAIM_SCHEMA_VERSION: u32 = 1;
pub const RELATIONSHIP_RESOLUTION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ClaimLifecycle {
    Active = 0,
    Revoked,
}

#[derive(Debug, Clone)]
pub struct RelationshipClaim {
    pub claim_id: u64,
    pub version: u32,
    pub subject: Arc<str>,
    pub scope: Arc<str>,
    pub issuer: Arc<str>,
    pub issuer_key_id: Arc<str>,
    pub authority_policy_version: Arc<str>,
    pub origin: Arc<str>,
    pub schema_version: u32,
    pub relationship: RelationshipKind,
    pub parameters: Vec<u8>,
    pub validity_start: u64,
    pub validity_end: u64,
    pub lifecycle: ClaimLifecycle,
    pub payload_digest: [u8; 32],
    pub signature: [u8; 64],
}

#[derive(Debug, Clone)]
pub struct TrustedIssuer {
    pub issuer: Arc<str>,
    pub key_id: Arc<str>,
    pub scope: Arc<str>,
    pub public_key: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct ClaimTrustConfig {
    pub schema_version: u32,
    pub authority_policy_version: Arc<str>,
    pub issuers: Vec<TrustedIssuer>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ResolutionMode {
    Explicit = 0,
    TrustedClaim,
    Unresolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ResolutionReason {
    Explicit = 0,
    TrustedClaim,
    NoTrustedClaim,
}

#[derive(Debug, Clone)]
pub struct RelationshipRequest {
    pub subject: Arc<str>,
    pub scope: Arc<str>,
    pub has_explicit_relationship: bool,
    pub explicit_relationship: RelationshipKind,
    pub explicit_parameters: Vec<u8>,
    pub claims: Vec<RelationshipClaim>,
}

#[derive(Debug, Clone)]
pub struct RelationshipResolution {
    pub schema_version: u32,
    pub mode: ResolutionMode,
    pub reason: ResolutionReason,
    pub has_relationship: bool,
    pub relationship: RelationshipKind,
    pub claim_id: u64,
    pub claim_version: u32,
    pub evaluated_at: u64,
    pub subject: Arc<str>,
    pub scope: Arc<str>,
    pub authority_policy_version: Arc<str>,
    pub parameters: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Binary encoding helpers (big-endian, mirroring the C `LanaBuffer` writers)
// ---------------------------------------------------------------------------

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_blob(out: &mut Vec<u8>, data: &[u8]) -> Result<(), LanaError> {
    if data.len() > u32::MAX as usize {
        return Err(LanaError::Limit);
    }
    put_u32(out, data.len() as u32);
    out.extend_from_slice(data);
    Ok(())
}

fn put_text(out: &mut Vec<u8>, text: &str) -> Result<(), LanaError> {
    if text.is_empty() {
        return Err(LanaError::Schema);
    }
    put_blob(out, text.as_bytes())
}

fn read_u32(data: &[u8], cursor: &mut usize) -> Option<u32> {
    let p = *cursor;
    if data.len() - p < 4 {
        return None;
    }
    let value = ((data[p] as u32) << 24)
        | ((data[p + 1] as u32) << 16)
        | ((data[p + 2] as u32) << 8)
        | (data[p + 3] as u32);
    *cursor = p + 4;
    Some(value)
}

fn read_u64(data: &[u8], cursor: &mut usize) -> Option<u64> {
    let p = *cursor;
    if data.len() - p < 8 {
        return None;
    }
    let mut value = 0u64;
    for index in 0..8 {
        value = (value << 8) | data[p + index] as u64;
    }
    *cursor = p + 8;
    Some(value)
}

fn read_blob(data: &[u8], cursor: &mut usize) -> Result<Vec<u8>, LanaError> {
    let length = read_u32(data, cursor).ok_or(LanaError::Corruption)?;
    let p = *cursor;
    if data.len() - p < length as usize {
        return Err(LanaError::Corruption);
    }
    if length == u32::MAX {
        return Err(LanaError::Limit);
    }
    let bytes = data[p..p + length as usize].to_vec();
    *cursor = p + length as usize;
    Ok(bytes)
}

fn read_text(data: &[u8], cursor: &mut usize) -> Result<Arc<str>, LanaError> {
    let bytes = read_blob(data, cursor)?;
    let text = std::str::from_utf8(&bytes).map_err(|_| LanaError::Corruption)?;
    Ok(Arc::from(text))
}

fn validate_parameters(relationship: RelationshipKind, parameters: &[u8]) -> Result<(), LanaError> {
    if (relationship != RelationshipKind::Exact && relationship != RelationshipKind::SameDependency)
        || parameters != b"{}"
    {
        return Err(LanaError::InvalidParameters);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Claim payload / digest / signature
// ---------------------------------------------------------------------------

pub fn claim_encode_payload(claim: &RelationshipClaim) -> Result<Vec<u8>, LanaError> {
    if claim.schema_version != RELATIONSHIP_CLAIM_SCHEMA_VERSION
        || claim.validity_start > claim.validity_end
        || (claim.lifecycle != ClaimLifecycle::Active && claim.lifecycle != ClaimLifecycle::Revoked)
    {
        return Err(LanaError::Schema);
    }
    validate_parameters(claim.relationship, &claim.parameters)?;

    let mut out = Vec::new();
    out.extend_from_slice(b"LCP1");
    put_u32(&mut out, claim.schema_version);
    put_u64(&mut out, claim.claim_id);
    put_u32(&mut out, claim.version);
    put_text(&mut out, &claim.subject)?;
    put_text(&mut out, &claim.scope)?;
    put_text(&mut out, &claim.issuer)?;
    put_text(&mut out, &claim.issuer_key_id)?;
    put_text(&mut out, &claim.authority_policy_version)?;
    put_text(&mut out, &claim.origin)?;
    put_u32(&mut out, claim.relationship as u32);
    put_blob(&mut out, &claim.parameters)?;
    put_u64(&mut out, claim.validity_start);
    put_u64(&mut out, claim.validity_end);
    put_u32(&mut out, claim.lifecycle as u32);
    Ok(out)
}

pub fn claim_compute_payload_digest(claim: &RelationshipClaim) -> Result<[u8; 32], LanaError> {
    let payload = claim_encode_payload(claim)?;
    Ok(sha256(&payload))
}

pub fn claim_verify(claim: &RelationshipClaim, public_key: &[u8; 32]) -> Result<bool, LanaError> {
    let payload = claim_encode_payload(claim)?;
    let digest = sha256(&payload);
    if digest != claim.payload_digest {
        return Ok(false);
    }
    let public_key = PublicKey::new(*public_key);
    let signature = Signature::new(claim.signature);
    Ok(public_key.verify(&payload, &signature).is_ok())
}

// ---------------------------------------------------------------------------
// Trust policy and validation
// ---------------------------------------------------------------------------

fn now_seconds() -> Result<u64, LanaError> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| LanaError::Io)?;
    Ok(now.as_secs())
}

fn trust_config_valid(trust: &ClaimTrustConfig) -> bool {
    trust.schema_version == 1 && !trust.authority_policy_version.is_empty()
}

fn find_issuer<'a>(claim: &RelationshipClaim, trust: &'a ClaimTrustConfig) -> Option<&'a TrustedIssuer> {
    trust.issuers.iter().find(|issuer| {
        issuer.issuer == claim.issuer
            && issuer.key_id == claim.issuer_key_id
            && issuer.scope == claim.scope
    })
}

fn validate_at(
    claim: &RelationshipClaim,
    trust: &ClaimTrustConfig,
    subject: &str,
    scope: &str,
    now: u64,
) -> Result<(), LanaError> {
    if !trust_config_valid(trust) {
        return Err(LanaError::InvalidState);
    }
    if claim.schema_version != RELATIONSHIP_CLAIM_SCHEMA_VERSION {
        return Err(LanaError::Schema);
    }
    if claim.subject.as_ref() != subject || claim.scope.as_ref() != scope {
        return Err(LanaError::ClaimMismatch);
    }
    validate_parameters(claim.relationship, &claim.parameters)?;
    if claim.lifecycle == ClaimLifecycle::Revoked {
        return Err(LanaError::ClaimRevoked);
    }
    if claim.lifecycle != ClaimLifecycle::Active {
        return Err(LanaError::Schema);
    }
    if now < claim.validity_start || now > claim.validity_end {
        return Err(LanaError::ClaimExpired);
    }
    if claim.authority_policy_version.as_ref() != trust.authority_policy_version.as_ref() {
        return Err(LanaError::UnauthorizedIssuer);
    }
    let issuer = find_issuer(claim, trust).ok_or(LanaError::UnauthorizedIssuer)?;
    let valid = claim_verify(claim, &issuer.public_key)?;
    if valid {
        Ok(())
    } else {
        Err(LanaError::Integrity)
    }
}

pub fn claim_validate(
    claim: &RelationshipClaim,
    trust: &ClaimTrustConfig,
    expected_subject: &str,
    expected_scope: &str,
) -> Result<u64, LanaError> {
    let now = now_seconds()?;
    validate_at(claim, trust, expected_subject, expected_scope, now)?;
    Ok(now)
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

fn set_resolution(
    request: &RelationshipRequest,
    trust: &ClaimTrustConfig,
    now: u64,
    mode: ResolutionMode,
    reason: ResolutionReason,
    claim: Option<&RelationshipClaim>,
    relationship: RelationshipKind,
    parameters: &[u8],
) -> RelationshipResolution {
    RelationshipResolution {
        schema_version: RELATIONSHIP_RESOLUTION_SCHEMA_VERSION,
        mode,
        reason,
        has_relationship: mode != ResolutionMode::Unresolved,
        relationship,
        claim_id: claim.map_or(0, |c| c.claim_id),
        claim_version: claim.map_or(0, |c| c.version),
        evaluated_at: now,
        subject: request.subject.clone(),
        scope: request.scope.clone(),
        authority_policy_version: trust.authority_policy_version.clone(),
        parameters: parameters.to_vec(),
    }
}

pub fn relationship_resolve(
    request: &RelationshipRequest,
    trust: &ClaimTrustConfig,
) -> Result<RelationshipResolution, LanaError> {
    if !trust_config_valid(trust) {
        return Err(LanaError::InvalidState);
    }
    let now = now_seconds()?;

    if request.has_explicit_relationship {
        validate_parameters(request.explicit_relationship, &request.explicit_parameters)?;
        return Ok(set_resolution(
            request,
            trust,
            now,
            ResolutionMode::Explicit,
            ResolutionReason::Explicit,
            None,
            request.explicit_relationship,
            &request.explicit_parameters,
        ));
    }

    let mut selected: Option<&RelationshipClaim> = None;
    for candidate in &request.claims {
        match validate_at(candidate, trust, &request.subject, &request.scope, now) {
            Err(LanaError::ClaimMismatch)
            | Err(LanaError::ClaimRevoked)
            | Err(LanaError::ClaimExpired)
            | Err(LanaError::UnauthorizedIssuer)
            | Err(LanaError::Integrity) => continue,
            Err(error) => return Err(error),
            Ok(()) => {}
        }
        if let Some(sel) = selected {
            if sel.relationship != candidate.relationship {
                return Err(LanaError::Conflict);
            }
        }
        if selected.is_none()
            || candidate.version > selected.unwrap().version
            || (candidate.version == selected.unwrap().version
                && candidate.claim_id < selected.unwrap().claim_id)
        {
            selected = Some(candidate);
        }
    }

    match selected {
        Some(sel) => Ok(set_resolution(
            request,
            trust,
            now,
            ResolutionMode::TrustedClaim,
            ResolutionReason::TrustedClaim,
            Some(sel),
            sel.relationship,
            &sel.parameters,
        )),
        None => Ok(set_resolution(
            request,
            trust,
            now,
            ResolutionMode::Unresolved,
            ResolutionReason::NoTrustedClaim,
            None,
            RelationshipKind::Exact,
            &[],
        )),
    }
}

pub fn relationship_resolution_encode(resolution: &RelationshipResolution) -> Result<Vec<u8>, LanaError> {
    if resolution.schema_version != RELATIONSHIP_RESOLUTION_SCHEMA_VERSION
        || (!resolution.has_relationship && !resolution.parameters.is_empty())
    {
        return Err(LanaError::Schema);
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"LRR1");
    put_u32(&mut out, resolution.schema_version);
    put_u32(&mut out, resolution.mode as u32);
    put_u32(&mut out, resolution.reason as u32);
    put_u32(&mut out, if resolution.has_relationship { 1 } else { 0 });
    put_u32(&mut out, resolution.relationship as u32);
    put_u64(&mut out, resolution.claim_id);
    put_u32(&mut out, resolution.claim_version);
    put_u64(&mut out, resolution.evaluated_at);
    put_text(&mut out, &resolution.subject)?;
    put_text(&mut out, &resolution.scope)?;
    put_text(&mut out, &resolution.authority_policy_version)?;
    put_blob(&mut out, &resolution.parameters)?;
    Ok(out)
}

pub fn relationship_resolution_decode(data: &[u8]) -> Result<RelationshipResolution, LanaError> {
    if data.len() < 4 || &data[..4] != b"LRR1" {
        return Err(LanaError::Schema);
    }
    let mut cursor = 4usize;
    let end = data.len();

    let schema_version = read_u32(data, &mut cursor).ok_or(LanaError::Corruption)?;
    if schema_version != RELATIONSHIP_RESOLUTION_SCHEMA_VERSION {
        return Err(LanaError::Corruption);
    }
    let mode_raw = read_u32(data, &mut cursor).ok_or(LanaError::Corruption)?;
    if mode_raw > ResolutionMode::Unresolved as u32 {
        return Err(LanaError::Corruption);
    }
    let mode = match mode_raw {
        0 => ResolutionMode::Explicit,
        1 => ResolutionMode::TrustedClaim,
        _ => ResolutionMode::Unresolved,
    };
    let reason_raw = read_u32(data, &mut cursor).ok_or(LanaError::Corruption)?;
    if reason_raw > ResolutionReason::NoTrustedClaim as u32 {
        return Err(LanaError::Corruption);
    }
    let reason = match reason_raw {
        0 => ResolutionReason::Explicit,
        1 => ResolutionReason::TrustedClaim,
        _ => ResolutionReason::NoTrustedClaim,
    };
    let has_rel_raw = read_u32(data, &mut cursor).ok_or(LanaError::Corruption)?;
    if has_rel_raw > 1 {
        return Err(LanaError::Corruption);
    }
    let has_relationship = has_rel_raw != 0;
    let relationship_raw = read_u32(data, &mut cursor).ok_or(LanaError::Corruption)?;
    let relationship = match relationship_raw {
        0 => RelationshipKind::Exact,
        1 => RelationshipKind::SameDependency,
        2 => RelationshipKind::ExplicitJoint,
        _ => return Err(LanaError::Corruption),
    };
    let claim_id = read_u64(data, &mut cursor).ok_or(LanaError::Corruption)?;
    let claim_version = read_u32(data, &mut cursor).ok_or(LanaError::Corruption)?;
    let evaluated_at = read_u64(data, &mut cursor).ok_or(LanaError::Corruption)?;

    let subject = read_text(data, &mut cursor)?;
    let scope = read_text(data, &mut cursor)?;
    let authority_policy_version = read_text(data, &mut cursor)?;
    let parameters = read_blob(data, &mut cursor)?;

    let resolution = RelationshipResolution {
        schema_version,
        mode,
        reason,
        has_relationship,
        relationship,
        claim_id,
        claim_version,
        evaluated_at,
        subject,
        scope,
        authority_policy_version,
        parameters,
    };

    // The C reference returns Ok (ignoring trailing bytes) when the parameters
    // blob is followed by extra data, skipping the consistency check below.
    if cursor != end {
        return Ok(resolution);
    }

    if (has_relationship && validate_parameters(relationship, &resolution.parameters).is_err())
        || (!has_relationship && !resolution.parameters.is_empty())
    {
        return Err(LanaError::Corruption);
    }

    Ok(resolution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_compact::{KeyPair, Seed};

    fn hex32(hex: &str) -> [u8; 32] {
        let mut out = [0u8; 32];
        for index in 0..32 {
            out[index] = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap();
        }
        out
    }

    fn test_claim() -> RelationshipClaim {
        RelationshipClaim {
            claim_id: 1,
            version: 1,
            subject: Arc::from("alice"),
            scope: Arc::from("scope-a"),
            issuer: Arc::from("issuer-1"),
            issuer_key_id: Arc::from("key-1"),
            authority_policy_version: Arc::from("v1"),
            origin: Arc::from("origin-1"),
            schema_version: RELATIONSHIP_CLAIM_SCHEMA_VERSION,
            relationship: RelationshipKind::Exact,
            parameters: b"{}".to_vec(),
            validity_start: 0,
            validity_end: u64::MAX,
            lifecycle: ClaimLifecycle::Active,
            payload_digest: [0u8; 32],
            signature: [0u8; 64],
        }
    }

    #[test]
    fn verify_rfc8032_signature() {
        let seed_bytes = hex32("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
        let public_key = hex32("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");

        let mut claim = test_claim();
        let payload = claim_encode_payload(&claim).unwrap();
        claim.payload_digest = claim_compute_payload_digest(&claim).unwrap();

        let key_pair = KeyPair::from_seed(Seed::new(seed_bytes));
        let signature = key_pair.sk.sign(&payload, None);
        claim.signature = *signature;

        assert_eq!(claim_verify(&claim, &public_key), Ok(true));

        claim.signature[0] ^= 0xff;
        assert_eq!(claim_verify(&claim, &public_key), Ok(false));
    }
}
