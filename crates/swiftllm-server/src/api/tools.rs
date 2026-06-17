// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      tools.rs
// PATH:      /crates/swiftllm-server/src/api/tools.rs
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

//! Tool / function calling support shared by the OpenAI and Anthropic APIs.
//!
//! This module provides:
//!
//! * the OpenAI-shaped wire types ([`ToolDefinition`], [`ToolChoice`],
//!   [`ToolCall`]);
//! * [`render_tool_system_prompt`], which serialises tool schemas into a system
//!   prompt instruction (how a server actually exposes tools to a model); and
//! * [`parse_tool_calls`], a zero-overhead native parser that extracts tool
//!   calls from a model's raw output. It recognises the three conventions models
//!   are commonly trained to emit: `<tool_call>{…}</tool_call>` tags (Hermes /
//!   Qwen style), fenced ```` ```tool_call ```` / ```` ```json ```` blocks, and a
//!   bare top-level JSON object.
//!
//! Because the streaming engine is format-agnostic, the same parser serves both
//! the OpenAI (`tool_calls`) and Anthropic (`tool_use` content block) responses.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

fn default_function_type() -> String {
    "function".to_string()
}

/// A tool the model is allowed to call (OpenAI shape).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    /// Tool type — currently always `"function"`.
    #[serde(rename = "type", default = "default_function_type")]
    pub tool_type: String,

    /// The function definition.
    pub function: FunctionDef,
}

/// A callable function's schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionDef {
    /// Function name.
    pub name: String,

    /// Human-readable description (helps the model decide when to call it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON-Schema document describing the arguments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

/// How the model should choose among tools.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ToolChoice {
    /// `"auto"`, `"none"`, or `"required"`.
    Mode(String),
    /// Force a specific named function.
    Named {
        /// Always `"function"`.
        #[serde(rename = "type")]
        tool_type: String,
        /// The forced function.
        function: NamedFunction,
    },
}

/// The function selected by a [`ToolChoice::Named`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NamedFunction {
    /// Function name.
    pub name: String,
}

impl ToolChoice {
    /// Whether tool calling is disabled (`"none"`).
    pub fn is_none(&self) -> bool {
        matches!(self, ToolChoice::Mode(m) if m == "none")
    }

    /// Whether the model is required to emit a tool call (`"required"` or a
    /// forced named function).
    pub fn is_required(&self) -> bool {
        match self {
            ToolChoice::Mode(m) => m == "required",
            ToolChoice::Named { .. } => true,
        }
    }

    /// The forced function name, if a specific tool was requested.
    pub fn forced_name(&self) -> Option<&str> {
        match self {
            ToolChoice::Named { function, .. } => Some(function.name.as_str()),
            _ => None,
        }
    }
}

/// A tool call emitted by the model (OpenAI shape).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Unique call id (`call_…`).
    pub id: String,

    /// Always `"function"`.
    #[serde(rename = "type", default = "default_function_type")]
    pub call_type: String,

    /// The function invocation.
    pub function: FunctionCall,
}

/// The function name + arguments of a [`ToolCall`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    /// Function name.
    pub name: String,

    /// Arguments encoded as a JSON string (per the OpenAI wire format).
    pub arguments: String,
}

impl ToolCall {
    /// Construct a tool call from a name and arguments value, generating an id.
    pub fn new(name: impl Into<String>, arguments: &Value) -> Self {
        Self {
            id: format!("call_{}", &Uuid::new_v4().to_string().replace('-', "")[..24]),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.to_string(),
            },
        }
    }
}

/// Render a system-prompt instruction describing the available tools.
///
/// This is the mechanism by which the server actually exposes tools to a model
/// that has no native tool grammar: the schemas are injected as text and the
/// model is told to reply with `<tool_call>` blocks, which [`parse_tool_calls`]
/// then recovers.
pub fn render_tool_system_prompt(tools: &[ToolDefinition], choice: Option<&ToolChoice>) -> String {
    let mut out = String::new();
    out.push_str(
        "You have access to the following tools. To call one, emit a line of the form \
         <tool_call>{\"name\": <tool-name>, \"arguments\": <args-object>}</tool_call>. \
         You may emit multiple such blocks.\n\nTools:\n",
    );
    for tool in tools {
        let f = &tool.function;
        out.push_str("- ");
        out.push_str(&f.name);
        if let Some(desc) = &f.description {
            out.push_str(": ");
            out.push_str(desc);
        }
        if let Some(params) = &f.parameters {
            out.push_str("\n  parameters: ");
            out.push_str(&params.to_string());
        }
        out.push('\n');
    }
    match choice {
        Some(c) if c.is_required() => {
            out.push_str("\nYou MUST call a tool to answer this request.");
            if let Some(name) = c.forced_name() {
                out.push_str(" Call the tool named '");
                out.push_str(name);
                out.push_str("'.");
            }
        }
        _ => {}
    }
    out
}

/// Extract tool calls from a model's raw output.
///
/// Returns `None` if the output contains no recognisable tool call (the caller
/// then treats the output as ordinary text). When `tools` is non-empty, calls
/// naming an unknown function are dropped.
pub fn parse_tool_calls(output: &str, tools: &[ToolDefinition]) -> Option<Vec<ToolCall>> {
    let known: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
    let mut calls = Vec::new();

    // 1) <tool_call>…</tool_call> tagged blocks (Hermes / Qwen).
    for block in extract_delimited(output, "<tool_call>", "</tool_call>") {
        if let Some(call) = tool_call_from_json_str(block.trim(), &known) {
            calls.push(call);
        }
    }

    // 2) Fenced ```tool_call / ```json blocks.
    if calls.is_empty() {
        for block in extract_fenced(output) {
            if let Some(call) = tool_call_from_json_str(block.trim(), &known) {
                calls.push(call);
            }
        }
    }

    // 3) A bare top-level JSON object/array.
    if calls.is_empty() {
        let trimmed = output.trim();
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
                collect_calls_from_value(&value, &known, &mut calls);
            }
        }
    }

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

/// Parse one `{"name":…,"arguments":…}` (or `{"tool_calls":[…]}`) JSON fragment.
fn tool_call_from_json_str(s: &str, known: &[&str]) -> Option<ToolCall> {
    let value: Value = serde_json::from_str(s).ok()?;
    single_call_from_value(&value, known)
}

fn single_call_from_value(value: &Value, known: &[&str]) -> Option<ToolCall> {
    let obj = value.as_object()?;
    // Anthropic-ish `{"name":…, "input":…}` or OpenAI-ish `{"name":…, "arguments":…}`.
    let name = obj.get("name")?.as_str()?.to_string();
    if !known.is_empty() && !known.contains(&name.as_str()) {
        return None;
    }
    let args = obj
        .get("arguments")
        .or_else(|| obj.get("input"))
        .or_else(|| obj.get("parameters"))
        .cloned()
        .unwrap_or(Value::Object(Default::default()));
    // Arguments may already be a JSON-encoded string.
    let args = match args {
        Value::String(s) => serde_json::from_str(&s).unwrap_or(Value::String(s)),
        other => other,
    };
    Some(ToolCall::new(name, &args))
}

fn collect_calls_from_value(value: &Value, known: &[&str], out: &mut Vec<ToolCall>) {
    match value {
        Value::Array(items) => {
            for item in items {
                if let Some(call) = single_call_from_value(item, known) {
                    out.push(call);
                }
            }
        }
        Value::Object(obj) => {
            if let Some(Value::Array(items)) = obj.get("tool_calls") {
                for item in items {
                    if let Some(call) = single_call_from_value(item, known) {
                        out.push(call);
                    }
                }
            } else if let Some(call) = single_call_from_value(value, known) {
                out.push(call);
            }
        }
        _ => {}
    }
}

/// Collect every substring strictly between matching `open`/`close` delimiters.
fn extract_delimited(s: &str, open: &str, close: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(start) = rest.find(open) {
        let after = &rest[start + open.len()..];
        if let Some(end) = after.find(close) {
            out.push(after[..end].to_string());
            rest = &after[end + close.len()..];
        } else {
            break;
        }
    }
    out
}

/// Collect the bodies of fenced code blocks whose info string is empty,
/// `json`, or `tool_call`.
fn extract_fenced(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(start) = rest.find("```") {
        let after = &rest[start + 3..];
        // Info string up to the first newline.
        let (info, body_start) = match after.find('\n') {
            Some(nl) => (after[..nl].trim(), nl + 1),
            None => break,
        };
        let body_region = &after[body_start..];
        let end = match body_region.find("```") {
            Some(e) => e,
            None => break,
        };
        if info.is_empty() || info.eq_ignore_ascii_case("json") || info.eq_ignore_ascii_case("tool_call") {
            out.push(body_region[..end].to_string());
        }
        rest = &body_region[end + 3..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather_tool() -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "get_weather".to_string(),
                description: Some("Get the weather".to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                })),
            },
        }
    }

    #[test]
    fn parses_tagged_tool_call() {
        let out = "Let me check.\n<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>";
        let calls = parse_tool_calls(out, &[weather_tool()]).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "Paris");
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn parses_multiple_tagged_calls() {
        let out = "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"A\"}}</tool_call>\
                   <tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"B\"}}</tool_call>";
        let calls = parse_tool_calls(out, &[weather_tool()]).unwrap();
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn parses_fenced_json_block() {
        let out = "Sure:\n```json\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}\n```";
        let calls = parse_tool_calls(out, &[weather_tool()]).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn parses_bare_json_and_input_alias() {
        // Anthropic-style "input" key, bare object.
        let out = "{\"name\": \"get_weather\", \"input\": {\"city\": \"Oslo\"}}";
        let calls = parse_tool_calls(out, &[weather_tool()]).unwrap();
        assert_eq!(calls.len(), 1);
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "Oslo");
    }

    #[test]
    fn arguments_encoded_as_string_are_reparsed() {
        let out = "<tool_call>{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"NYC\\\"}\"}</tool_call>";
        let calls = parse_tool_calls(out, &[weather_tool()]).unwrap();
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn unknown_tool_is_dropped() {
        let out = "<tool_call>{\"name\": \"launch_rockets\", \"arguments\": {}}</tool_call>";
        assert!(parse_tool_calls(out, &[weather_tool()]).is_none());
    }

    #[test]
    fn plain_text_yields_no_calls() {
        assert!(parse_tool_calls("The weather is sunny.", &[weather_tool()]).is_none());
    }

    #[test]
    fn tool_choice_helpers() {
        assert!(ToolChoice::Mode("none".into()).is_none());
        assert!(ToolChoice::Mode("required".into()).is_required());
        let named = ToolChoice::Named {
            tool_type: "function".into(),
            function: NamedFunction { name: "get_weather".into() },
        };
        assert!(named.is_required());
        assert_eq!(named.forced_name(), Some("get_weather"));
    }

    #[test]
    fn system_prompt_lists_tools() {
        let prompt = render_tool_system_prompt(&[weather_tool()], Some(&ToolChoice::Mode("required".into())));
        assert!(prompt.contains("get_weather"));
        assert!(prompt.contains("<tool_call>"));
        assert!(prompt.contains("MUST call a tool"));
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: tools.rs
// REPO PATH:   /swiftllm/crates/swiftllm-server/src/api/tools.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
