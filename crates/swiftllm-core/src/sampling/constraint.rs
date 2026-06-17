// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      constraint.rs
// PATH:      /crates/swiftllm-core/src/sampling/constraint.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

//! Schema-constrained (structured) generation.
//!
//! This module implements zero-penalty JSON-Schema constraints for sampling. A
//! [`SchemaConstraint`] is compiled from a JSON Schema document and can:
//!
//! * answer, incrementally, whether a partially-generated string is a valid
//!   *prefix* of some document matching the schema ([`SchemaConstraint::is_valid_prefix`]);
//! * answer whether a string is already a *complete* matching document
//!   ([`SchemaConstraint::is_complete`]);
//! * mask a logits vector so that only tokens which keep the output on a valid
//!   path survive ([`SchemaConstraint::mask_logits`]); and
//! * fully validate a finished document ([`SchemaConstraint::validate_json`]).
//!
//! The incremental matcher is a recursive-descent *prefix* parser. Rather than
//! materialising a token-level finite automaton up front (the FlashInfer/xgrammar
//! approach used on GPU), it answers "could this prefix still be completed?" in
//! one pass over the generated string. That keeps it allocation-light and makes
//! it directly unit-testable on CPU without a tokenizer or model.
//!
//! ## Supported schema subset
//!
//! `object` (closed, all declared properties required, emitted in the order
//! `serde_json` yields them — i.e. lexicographic), `array` (`items`, `minItems`,
//! `maxItems`), `string` (`enum`, `maxLength`), `integer` (`minimum`/`maximum`),
//! `number`, `boolean`, `null`, top-level `enum`/`const`, and `true`/`false`
//! schemas. Unconstrained nodes fall back to "any well-formed JSON value".
//!
//! Optional/unordered object keys are intentionally *not* permitted by the
//! incremental constraint — generation is pinned to a canonical key order so the
//! prefix parser stays deterministic. [`SchemaConstraint::validate_json`] is the
//! lenient, order-independent checker (it honours `required`) and should be used
//! to validate inputs rather than to guide generation.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Result of attempting to scan a value off the front of an input string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scan {
    /// A complete value was consumed using exactly `n` bytes.
    Done(usize),
    /// The input is a valid prefix of a value but is not yet complete.
    Partial,
    /// The input cannot be the prefix of any value matching the schema.
    Fail,
}

/// A compiled subset of JSON Schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JsonSchema {
    /// Matches any well-formed JSON value.
    Any,
    /// Matches nothing (the `false` schema).
    Never,
    /// JSON `null`.
    Null,
    /// JSON boolean.
    Boolean,
    /// JSON integer, with optional inclusive bounds.
    Integer {
        /// Inclusive lower bound.
        minimum: Option<i64>,
        /// Inclusive upper bound.
        maximum: Option<i64>,
    },
    /// JSON number (integer or floating point).
    Number,
    /// JSON string, optionally restricted to an enumeration of literal values.
    StringT {
        /// Allowed literal values (string `enum`); `None` means any string.
        enum_values: Option<Vec<String>>,
        /// Maximum number of characters (after decoding).
        max_length: Option<usize>,
    },
    /// JSON array of homogeneously-typed items.
    Array {
        /// Schema each element must satisfy.
        items: Box<JsonSchema>,
        /// Minimum element count.
        min_items: usize,
        /// Maximum element count (`None` = unbounded).
        max_items: Option<usize>,
    },
    /// JSON object with a fixed, ordered set of properties.
    Object {
        /// `(name, schema)` pairs in canonical generation order.
        properties: Vec<(String, JsonSchema)>,
        /// Property names that must be present (used by `validate_json`).
        required: Vec<String>,
    },
    /// A fixed enumeration of allowed JSON values (covers `enum`/`const`).
    Enum(Vec<Value>),
}

impl JsonSchema {
    /// Compile a JSON Schema document into a [`JsonSchema`].
    ///
    /// Unknown or unsupported constructs degrade gracefully to [`JsonSchema::Any`]
    /// rather than failing, so a partially-understood schema still produces a
    /// usable (if looser) constraint.
    pub fn compile(doc: &Value) -> Self {
        match doc {
            // Boolean schemas: `true` => anything, `false` => nothing.
            Value::Bool(true) => JsonSchema::Any,
            Value::Bool(false) => JsonSchema::Never,
            Value::Object(map) => Self::compile_object_schema(map),
            // A bare value is treated as a `const`.
            other => JsonSchema::Enum(vec![other.clone()]),
        }
    }

    fn compile_object_schema(map: &serde_json::Map<String, Value>) -> Self {
        // `const` takes precedence: exactly one allowed value.
        if let Some(c) = map.get("const") {
            return JsonSchema::Enum(vec![c.clone()]);
        }
        // A typeless `enum` is a set of allowed values.
        if let Some(Value::Array(values)) = map.get("enum") {
            // If it's a string-typed enum, model it as a string with literals so
            // the incremental matcher can constrain character-by-character.
            let all_strings = values.iter().all(|v| v.is_string());
            if all_strings && map.get("type").and_then(|t| t.as_str()) != Some("object") {
                let literals = values
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                return JsonSchema::StringT {
                    enum_values: Some(literals),
                    max_length: None,
                };
            }
            return JsonSchema::Enum(values.clone());
        }

        let ty = map.get("type").and_then(|t| t.as_str());
        match ty {
            Some("null") => JsonSchema::Null,
            Some("boolean") => JsonSchema::Boolean,
            Some("integer") => JsonSchema::Integer {
                minimum: map.get("minimum").and_then(Value::as_i64),
                maximum: map.get("maximum").and_then(Value::as_i64),
            },
            Some("number") => JsonSchema::Number,
            Some("string") => JsonSchema::StringT {
                enum_values: None,
                max_length: map
                    .get("maxLength")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize),
            },
            Some("array") => {
                let items = map
                    .get("items")
                    .map(Self::compile)
                    .unwrap_or(JsonSchema::Any);
                JsonSchema::Array {
                    items: Box::new(items),
                    min_items: map
                        .get("minItems")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize,
                    max_items: map
                        .get("maxItems")
                        .and_then(Value::as_u64)
                        .map(|n| n as usize),
                }
            }
            Some("object") | None if map.contains_key("properties") => {
                Self::compile_object(map)
            }
            Some("object") => Self::compile_object(map),
            _ => JsonSchema::Any,
        }
    }

    fn compile_object(map: &serde_json::Map<String, Value>) -> Self {
        let mut properties = Vec::new();
        if let Some(Value::Object(props)) = map.get("properties") {
            // `serde_json::Map` iterates in lexicographic order by default; that
            // fixed order is the canonical generation order for the constraint.
            for (name, sub) in props {
                properties.push((name.clone(), Self::compile(sub)));
            }
        }
        let required = map
            .get("required")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        JsonSchema::Object {
            properties,
            required,
        }
    }
}

/// Count leading JSON whitespace bytes.
fn lead_ws(s: &str) -> usize {
    s.bytes()
        .take_while(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
        .count()
}

/// Scan a fixed literal (e.g. `true`, `null`) as a prefix.
fn scan_literal(s: &str, lit: &str) -> Scan {
    let sb = s.as_bytes();
    let lb = lit.as_bytes();
    let n = sb.len().min(lb.len());
    if sb[..n] != lb[..n] {
        return Scan::Fail;
    }
    if sb.len() >= lb.len() {
        Scan::Done(lb.len())
    } else {
        Scan::Partial
    }
}

/// Scan a JSON number prefix. When `integer_only`, a fractional or exponent part
/// terminates the number (the caller's delimiter check rejects misuse).
fn scan_number(s: &str, integer_only: bool) -> Scan {
    let b = s.as_bytes();
    let n = b.len();
    let mut i = 0;
    if n == 0 {
        return Scan::Partial;
    }
    if b[i] == b'-' {
        i += 1;
        if i == n {
            return Scan::Partial;
        }
    }
    // Integer part.
    if b[i] == b'0' {
        i += 1;
    } else if b[i].is_ascii_digit() {
        while i < n && b[i].is_ascii_digit() {
            i += 1;
        }
    } else {
        return Scan::Fail;
    }
    let mut complete = i;
    // Fractional part.
    if i < n && b[i] == b'.' {
        if integer_only {
            return Scan::Done(complete);
        }
        i += 1;
        if i == n {
            return Scan::Partial;
        }
        if !b[i].is_ascii_digit() {
            return Scan::Fail;
        }
        while i < n && b[i].is_ascii_digit() {
            i += 1;
        }
        complete = i;
    }
    // Exponent part.
    if i < n && (b[i] == b'e' || b[i] == b'E') {
        if integer_only {
            return Scan::Done(complete);
        }
        i += 1;
        if i == n {
            return Scan::Partial;
        }
        if b[i] == b'+' || b[i] == b'-' {
            i += 1;
            if i == n {
                return Scan::Partial;
            }
        }
        if !b[i].is_ascii_digit() {
            return Scan::Fail;
        }
        while i < n && b[i].is_ascii_digit() {
            i += 1;
        }
        complete = i;
    }
    Scan::Done(complete)
}

/// Scan a JSON string prefix. If `allowed` is set, the decoded content must be a
/// prefix of (and, when closed, equal to) one of the allowed literals. Escapes in
/// enum literals are not supported (enum values are assumed to be plain text).
fn scan_string(s: &str, allowed: Option<&[String]>, max_length: Option<usize>) -> Scan {
    let b = s.as_bytes();
    let n = b.len();
    if n == 0 {
        return Scan::Partial;
    }
    if b[0] != b'"' {
        return Scan::Fail;
    }
    let mut i = 1;
    // Decoded character count for max_length enforcement.
    let mut decoded_len = 0usize;
    while i < n {
        let c = b[i];
        match c {
            b'"' => {
                // Closing quote: the inner slice is the raw (still-escaped) body.
                let body = &s[1..i];
                if let Some(opts) = allowed {
                    if !opts.iter().any(|o| o == body) {
                        return Scan::Fail;
                    }
                }
                return Scan::Done(i + 1);
            }
            b'\\' => {
                if i + 1 >= n {
                    return Scan::Partial; // escape started, need its second char
                }
                match b[i + 1] {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                        i += 2;
                        decoded_len += 1;
                    }
                    b'u' => {
                        // Need exactly four hex digits.
                        let have = n - (i + 2);
                        let take = have.min(4);
                        for k in 0..take {
                            if !b[i + 2 + k].is_ascii_hexdigit() {
                                return Scan::Fail;
                            }
                        }
                        if take < 4 {
                            return Scan::Partial;
                        }
                        i += 6;
                        decoded_len += 1;
                    }
                    _ => return Scan::Fail,
                }
            }
            // Raw control characters are not allowed unescaped in JSON strings.
            0x00..=0x1F => return Scan::Fail,
            _ => {
                i += 1;
                decoded_len += 1;
            }
        }
        if let Some(max) = max_length {
            if decoded_len > max {
                return Scan::Fail;
            }
        }
        // Enum prefix check on the raw body decoded so far.
        if let Some(opts) = allowed {
            let body = &s[1..i];
            if !opts.iter().any(|o| o.starts_with(body)) {
                return Scan::Fail;
            }
        }
    }
    Scan::Partial // string not yet closed
}

/// Scan an arbitrary well-formed JSON value prefix (used for `Any`).
fn scan_any(s: &str) -> Scan {
    let ws = lead_ws(s);
    let rest = &s[ws..];
    let b = rest.as_bytes();
    if b.is_empty() {
        return Scan::Partial;
    }
    let inner = match b[0] {
        b'{' => scan_any_object(rest),
        b'[' => scan_any_array(rest),
        b'"' => scan_string(rest, None, None),
        b't' => scan_literal(rest, "true"),
        b'f' => scan_literal(rest, "false"),
        b'n' => scan_literal(rest, "null"),
        b'-' | b'0'..=b'9' => scan_number(rest, false),
        _ => Scan::Fail,
    };
    match inner {
        Scan::Done(k) => Scan::Done(ws + k),
        other => other,
    }
}

fn scan_any_array(s: &str) -> Scan {
    scan_array(&JsonSchema::Any, 0, None, s)
}

fn scan_any_object(s: &str) -> Scan {
    let b = s.as_bytes();
    let mut i = 1; // past '{'
    let mut count = 0;
    loop {
        i += lead_ws(&s[i..]);
        if i >= s.len() {
            return Scan::Partial;
        }
        if b[i] == b'}' {
            return Scan::Done(i + 1);
        }
        if count > 0 {
            if b[i] != b',' {
                return Scan::Fail;
            }
            i += 1;
            i += lead_ws(&s[i..]);
            if i >= s.len() {
                return Scan::Partial;
            }
        }
        // key
        match scan_string(&s[i..], None, None) {
            Scan::Done(k) => i += k,
            Scan::Partial => return Scan::Partial,
            Scan::Fail => return Scan::Fail,
        }
        i += lead_ws(&s[i..]);
        if i >= s.len() {
            return Scan::Partial;
        }
        if b[i] != b':' {
            return Scan::Fail;
        }
        i += 1;
        match scan_value(&JsonSchema::Any, &s[i..]) {
            Scan::Done(k) => i += k,
            Scan::Partial => return Scan::Partial,
            Scan::Fail => return Scan::Fail,
        }
        count += 1;
    }
}

/// Scan an array against an item schema and cardinality bounds.
fn scan_array(item: &JsonSchema, min_items: usize, max_items: Option<usize>, s: &str) -> Scan {
    let pre = lead_ws(s);
    let s = &s[pre..];
    let b = s.as_bytes();
    if b.is_empty() {
        return Scan::Partial;
    }
    if b[0] != b'[' {
        return Scan::Fail;
    }
    let mut i = 1;
    let mut count = 0usize;
    loop {
        i += lead_ws(&s[i..]);
        if i >= s.len() {
            return Scan::Partial;
        }
        if s.as_bytes()[i] == b']' {
            return if count >= min_items {
                Scan::Done(pre + i + 1)
            } else {
                Scan::Fail
            };
        }
        if count > 0 {
            if s.as_bytes()[i] != b',' {
                return Scan::Fail;
            }
            // A comma commits to another element; reject it if that element
            // would exceed maxItems (so e.g. "[1,2,3," with maxItems=3 fails).
            if let Some(mx) = max_items {
                if count >= mx {
                    return Scan::Fail;
                }
            }
            i += 1;
            i += lead_ws(&s[i..]);
            if i >= s.len() {
                return Scan::Partial;
            }
            // A ']' right after ',' is a trailing comma — scan_value rejects it.
        }
        if let Some(mx) = max_items {
            if count >= mx {
                return Scan::Fail;
            }
        }
        match scan_value(item, &s[i..]) {
            Scan::Done(k) => {
                i += k;
                count += 1;
            }
            Scan::Partial => return Scan::Partial,
            Scan::Fail => return Scan::Fail,
        }
    }
}

/// Scan an object against an ordered, all-required property list.
fn scan_object(properties: &[(String, JsonSchema)], s: &str) -> Scan {
    let pre = lead_ws(s);
    let s = &s[pre..];
    let b = s.as_bytes();
    if b.is_empty() {
        return Scan::Partial;
    }
    if b[0] != b'{' {
        return Scan::Fail;
    }
    let mut i = 1;
    if properties.is_empty() {
        i += lead_ws(&s[i..]);
        if i >= s.len() {
            return Scan::Partial;
        }
        return if s.as_bytes()[i] == b'}' {
            Scan::Done(pre + i + 1)
        } else {
            Scan::Fail
        };
    }
    for (idx, (key, vschema)) in properties.iter().enumerate() {
        i += lead_ws(&s[i..]);
        if i >= s.len() {
            return Scan::Partial;
        }
        if idx == 0 {
            if s.as_bytes()[i] == b'}' {
                return Scan::Fail; // closed before required properties were emitted
            }
        } else {
            if s.as_bytes()[i] != b',' {
                return Scan::Fail;
            }
            i += 1;
            i += lead_ws(&s[i..]);
            if i >= s.len() {
                return Scan::Partial;
            }
        }
        // Key must be exactly the (quoted) property name.
        let quoted = format!("\"{}\"", key);
        match scan_literal(&s[i..], &quoted) {
            Scan::Done(k) => i += k,
            Scan::Partial => return Scan::Partial,
            Scan::Fail => return Scan::Fail,
        }
        i += lead_ws(&s[i..]);
        if i >= s.len() {
            return Scan::Partial;
        }
        if s.as_bytes()[i] != b':' {
            return Scan::Fail;
        }
        i += 1;
        i += lead_ws(&s[i..]);
        if i >= s.len() {
            return Scan::Partial;
        }
        match scan_value(vschema, &s[i..]) {
            Scan::Done(k) => i += k,
            Scan::Partial => return Scan::Partial,
            Scan::Fail => return Scan::Fail,
        }
    }
    i += lead_ws(&s[i..]);
    if i >= s.len() {
        return Scan::Partial;
    }
    if s.as_bytes()[i] != b'}' {
        return Scan::Fail;
    }
    Scan::Done(pre + i + 1)
}

/// Scan a value of `schema` off the front of `s`.
fn scan_value(schema: &JsonSchema, s: &str) -> Scan {
    let pre = lead_ws(s);
    let body = &s[pre..];
    if body.is_empty() {
        return Scan::Partial;
    }
    let inner = match schema {
        JsonSchema::Any => return scan_any(s),
        JsonSchema::Never => Scan::Fail,
        JsonSchema::Null => scan_literal(body, "null"),
        JsonSchema::Boolean => match body.as_bytes()[0] {
            b't' => scan_literal(body, "true"),
            b'f' => scan_literal(body, "false"),
            _ => Scan::Fail,
        },
        JsonSchema::Integer { .. } => scan_number(body, true),
        JsonSchema::Number => scan_number(body, false),
        JsonSchema::StringT {
            enum_values,
            max_length,
        } => scan_string(body, enum_values.as_deref(), *max_length),
        JsonSchema::Array {
            items,
            min_items,
            max_items,
        } => scan_array(items, *min_items, *max_items, body),
        JsonSchema::Object { properties, .. } => scan_object(properties, body),
        JsonSchema::Enum(values) => {
            // Match against each allowed value's canonical JSON serialisation.
            let mut any_partial = false;
            let mut best: Option<usize> = None;
            for v in values {
                let lit = serde_json::to_string(v).unwrap_or_default();
                match scan_literal(body, &lit) {
                    Scan::Done(k) => best = Some(best.map_or(k, |b| b.max(k))),
                    Scan::Partial => any_partial = true,
                    Scan::Fail => {}
                }
            }
            match best {
                Some(k) => Scan::Done(k),
                None if any_partial => Scan::Partial,
                None => Scan::Fail,
            }
        }
    };
    match inner {
        Scan::Done(k) => Scan::Done(pre + k),
        other => other,
    }
}

/// A compiled, reusable schema constraint for guided generation.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaConstraint {
    schema: JsonSchema,
}

impl SchemaConstraint {
    /// Compile a constraint from a JSON Schema document.
    pub fn from_schema(doc: &Value) -> Self {
        Self {
            schema: JsonSchema::compile(doc),
        }
    }

    /// Build a constraint directly from a compiled [`JsonSchema`].
    pub fn new(schema: JsonSchema) -> Self {
        Self { schema }
    }

    /// Borrow the compiled schema.
    pub fn schema(&self) -> &JsonSchema {
        &self.schema
    }

    fn match_root(&self, s: &str) -> Scan {
        match scan_value(&self.schema, s) {
            Scan::Done(n) => {
                // Only trailing whitespace may follow a complete document.
                if s[n..].trim().is_empty() {
                    Scan::Done(n)
                } else {
                    Scan::Fail
                }
            }
            other => other,
        }
    }

    /// Whether `s` is a valid prefix of some document matching the schema.
    pub fn is_valid_prefix(&self, s: &str) -> bool {
        !matches!(self.match_root(s), Scan::Fail)
    }

    /// Whether `s` is already a complete document matching the schema.
    pub fn is_complete(&self, s: &str) -> bool {
        matches!(self.match_root(s), Scan::Done(_))
    }

    /// Mask `logits` in place so that any candidate token which would push the
    /// running output `generated` off a valid path is set to negative infinity.
    ///
    /// `token_strings[i]` is the decoded text that token `i` would append.
    /// Returns the number of tokens masked. Tokens beyond `logits.len()` are
    /// ignored. This is the sampling-time hook that makes schema enforcement
    /// "zero-penalty": invalid tokens never enter the distribution.
    pub fn mask_logits(&self, generated: &str, token_strings: &[&str], logits: &mut [f32]) -> usize {
        let n = token_strings.len().min(logits.len());
        let mut masked = 0;
        // Reuse one buffer instead of allocating per token.
        let mut candidate = String::with_capacity(generated.len() + 16);
        for i in 0..n {
            candidate.clear();
            candidate.push_str(generated);
            candidate.push_str(token_strings[i]);
            if !self.is_valid_prefix(&candidate) {
                logits[i] = f32::NEG_INFINITY;
                masked += 1;
            }
        }
        masked
    }

    /// Fully validate a finished JSON document against the schema. Unlike the
    /// incremental matcher this is order-independent and honours `required`.
    pub fn validate_json(&self, s: &str) -> bool {
        match serde_json::from_str::<Value>(s) {
            Ok(v) => validate_value(&self.schema, &v),
            Err(_) => false,
        }
    }
}

/// Order-independent validation of a parsed value against a schema.
fn validate_value(schema: &JsonSchema, value: &Value) -> bool {
    match schema {
        JsonSchema::Any => true,
        JsonSchema::Never => false,
        JsonSchema::Null => value.is_null(),
        JsonSchema::Boolean => value.is_boolean(),
        JsonSchema::Integer { minimum, maximum } => match value.as_i64() {
            Some(n) => {
                minimum.map_or(true, |m| n >= m) && maximum.map_or(true, |m| n <= m)
            }
            None => false,
        },
        JsonSchema::Number => value.is_number(),
        JsonSchema::StringT {
            enum_values,
            max_length,
        } => match value.as_str() {
            Some(text) => {
                let len_ok = max_length.map_or(true, |m| text.chars().count() <= m);
                let enum_ok = enum_values
                    .as_ref()
                    .map_or(true, |opts| opts.iter().any(|o| o == text));
                len_ok && enum_ok
            }
            None => false,
        },
        JsonSchema::Array {
            items,
            min_items,
            max_items,
        } => match value.as_array() {
            Some(arr) => {
                arr.len() >= *min_items
                    && max_items.map_or(true, |m| arr.len() <= m)
                    && arr.iter().all(|el| validate_value(items, el))
            }
            None => false,
        },
        JsonSchema::Object {
            properties,
            required,
        } => match value.as_object() {
            Some(obj) => {
                // Every required key present.
                if !required.iter().all(|k| obj.contains_key(k)) {
                    return false;
                }
                // Every declared key that is present validates.
                properties.iter().all(|(name, sub)| {
                    obj.get(name).map_or(true, |v| validate_value(sub, v))
                })
            }
            None => false,
        },
        JsonSchema::Enum(values) => values.iter().any(|v| v == value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn constraint(doc: Value) -> SchemaConstraint {
        SchemaConstraint::from_schema(&doc)
    }

    #[test]
    fn boolean_schema_prefixes() {
        let c = constraint(json!({"type": "boolean"}));
        assert!(c.is_valid_prefix(""));
        assert!(c.is_valid_prefix("t"));
        assert!(c.is_valid_prefix("tru"));
        assert!(c.is_complete("true"));
        assert!(c.is_complete("false"));
        assert!(!c.is_valid_prefix("x"));
        assert!(!c.is_valid_prefix("truex"));
    }

    #[test]
    fn integer_bounds_and_prefixes() {
        let c = constraint(json!({"type": "integer", "minimum": 0, "maximum": 100}));
        assert!(c.is_valid_prefix("-")); // prefix-valid structurally
        assert!(c.is_complete("42"));
        assert!(c.is_complete("0"));
        assert!(!c.is_complete("1.5")); // not an integer
        // Bounds are enforced by validate_json, not the structural prefix matcher.
        assert!(c.validate_json("42"));
        assert!(!c.validate_json("250"));
        assert!(!c.validate_json("-5"));
    }

    #[test]
    fn number_partial_states() {
        let c = constraint(json!({"type": "number"}));
        assert!(c.is_valid_prefix("1."));
        assert!(c.is_valid_prefix("1e"));
        assert!(c.is_valid_prefix("-1.5e+"));
        assert!(c.is_complete("1.5"));
        assert!(c.is_complete("-3.2e10"));
        assert!(!c.is_valid_prefix("1.2.3"));
    }

    #[test]
    fn string_enum_constrains_characters() {
        let c = constraint(json!({"type": "string", "enum": ["red", "green", "blue"]}));
        assert!(c.is_valid_prefix("\"r"));
        assert!(c.is_valid_prefix("\"gre"));
        assert!(c.is_complete("\"red\""));
        assert!(c.is_complete("\"blue\""));
        assert!(!c.is_valid_prefix("\"x")); // no enum value starts with x
        assert!(!c.is_valid_prefix("\"redd")); // overshoots "red"
    }

    #[test]
    fn array_cardinality() {
        let c = constraint(json!({
            "type": "array", "items": {"type": "integer"},
            "minItems": 1, "maxItems": 3
        }));
        assert!(c.is_valid_prefix("["));
        assert!(c.is_valid_prefix("[1,"));
        assert!(c.is_complete("[1]"));
        assert!(c.is_complete("[1, 2, 3]"));
        assert!(!c.is_valid_prefix("[]")); // violates minItems
        assert!(!c.is_valid_prefix("[1,2,3,")); // would exceed maxItems
        assert!(!c.is_valid_prefix("[1,,")); // trailing/double comma
    }

    #[test]
    fn object_ordered_required_keys() {
        // Keys iterate lexicographically: "age" then "name".
        let c = constraint(json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"]
        }));
        assert!(c.is_valid_prefix("{"));
        assert!(c.is_valid_prefix("{\"age\": 3"));
        assert!(c.is_valid_prefix("{\"age\": 30, \"name\": \"a"));
        assert!(c.is_complete("{\"age\": 30, \"name\": \"alice\"}"));
        assert!(!c.is_valid_prefix("{\"name\"")); // wrong key order
        assert!(!c.is_valid_prefix("{}")); // required keys missing
    }

    #[test]
    fn nested_object_and_array() {
        let c = constraint(json!({
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
                "ok": {"type": "boolean"}
            }
        }));
        assert!(c.is_complete("{\"items\": [\"a\", \"b\"], \"ok\": true}"));
        assert!(c.is_valid_prefix("{\"items\": [\"a\""));
        assert!(!c.is_valid_prefix("{\"items\": [1")); // item must be string
    }

    #[test]
    fn mask_logits_blocks_invalid_tokens() {
        let c = constraint(json!({"type": "boolean"}));
        // Vocabulary of candidate token strings.
        let vocab = ["tr", "fa", "xy", "ue"];
        let mut logits = vec![1.0_f32; vocab.len()];
        let masked = c.mask_logits("", &vocab, &mut logits);
        // "tr" and "fa" are valid starts; "xy" and "ue" are not.
        assert_eq!(masked, 2);
        assert!(logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
        // After "tr", only "ue" completes "true".
        let mut logits2 = vec![1.0_f32; vocab.len()];
        c.mask_logits("tr", &vocab, &mut logits2);
        assert_eq!(logits2[3], 1.0); // "ue" -> "true"
        assert_eq!(logits2[0], f32::NEG_INFINITY); // "tr" -> "trtr"
    }

    #[test]
    fn validate_json_respects_optionality() {
        let c = constraint(json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
            "required": ["a"]
        }));
        assert!(c.validate_json("{\"a\": 1}")); // b optional
        assert!(c.validate_json("{\"a\": 1, \"b\": \"x\"}"));
        assert!(!c.validate_json("{\"b\": \"x\"}")); // missing required a
        assert!(!c.validate_json("{\"a\": \"not-int\"}"));
    }

    #[test]
    fn const_and_enum_values() {
        let c = constraint(json!({"const": 7}));
        assert!(c.is_complete("7"));
        assert!(!c.is_valid_prefix("8"));

        let e = constraint(json!({"enum": [1, 2, 42]}));
        assert!(e.is_valid_prefix("4"));
        assert!(e.is_complete("42"));
        assert!(!e.is_valid_prefix("3"));
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: constraint.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/sampling/constraint.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
