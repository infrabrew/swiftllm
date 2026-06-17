// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      anthropic.rs
// PATH:      /crates/swiftllm-server/src/api/anthropic.rs
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

//! Anthropic Messages API (`POST /v1/messages`).
//!
//! Implements the Anthropic Messages request/response schema and its native SSE
//! event stream (`message_start`, `content_block_start`, `content_block_delta`,
//! `content_block_stop`, `message_delta`, `message_stop`), plus `tool_use`
//! content blocks. Requests are flattened into the engine's prompt + sampling
//! configuration; structured `tool_choice` is wired through to the core
//! JSON-Schema sampling constraint.

use crate::AppState;
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use swiftllm_core::config::{ResponseFormat, SamplingConfig};
use swiftllm_core::types::Request;
use uuid::Uuid;

use super::tools::{FunctionDef, ToolDefinition};

// ----------------------------------------------------------------------------
// Request types
// ----------------------------------------------------------------------------

/// Anthropic Messages request body.
#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    /// Model id.
    pub model: String,

    /// Conversation turns.
    pub messages: Vec<InputMessage>,

    /// System prompt (string or array of text blocks).
    #[serde(default)]
    pub system: Option<SystemPrompt>,

    /// Maximum tokens to sample (required by the Anthropic API).
    pub max_tokens: usize,

    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Nucleus sampling.
    #[serde(default)]
    pub top_p: Option<f32>,

    /// Top-k sampling.
    #[serde(default)]
    pub top_k: Option<i32>,

    /// Custom stop sequences.
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,

    /// Whether to stream the response as SSE.
    #[serde(default)]
    pub stream: bool,

    /// Tools the model may use.
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,

    /// Tool-choice policy.
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoice>,
}

/// A system prompt: either a plain string or a list of text blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    /// Plain text.
    Text(String),
    /// A list of content blocks (only text blocks are used).
    Blocks(Vec<ContentBlock>),
}

/// An input message.
#[derive(Debug, Clone, Deserialize)]
pub struct InputMessage {
    /// `"user"` or `"assistant"`.
    pub role: String,
    /// Message content (string or content blocks).
    pub content: MessageContent,
}

/// Message content: a bare string or a list of content blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text shorthand.
    Text(String),
    /// Structured content blocks.
    Blocks(Vec<ContentBlock>),
}

/// A single content block in a message.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// A text block.
    Text {
        /// The text.
        text: String,
    },
    /// An image block (source is acknowledged but not decoded here).
    Image {
        /// The image source descriptor.
        #[allow(dead_code)]
        source: Value,
    },
    /// A prior assistant tool invocation (echoed back in history).
    ToolUse {
        /// The tool-use id.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input.
        input: Value,
    },
    /// A tool result supplied by the caller.
    ToolResult {
        /// The id of the tool_use this responds to.
        tool_use_id: String,
        /// The result content (string or blocks).
        #[serde(default)]
        content: Value,
    },
}

/// An Anthropic tool definition.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicTool {
    /// Tool name.
    pub name: String,
    /// Description.
    #[serde(default)]
    pub description: Option<String>,
    /// JSON-Schema for the tool input.
    #[serde(default)]
    pub input_schema: Option<Value>,
}

/// Anthropic tool-choice policy.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolChoice {
    /// Model decides whether to call a tool.
    Auto,
    /// Model must call some tool.
    Any,
    /// Model must call the named tool.
    Tool {
        /// The required tool name.
        name: String,
    },
}

// ----------------------------------------------------------------------------
// Response types
// ----------------------------------------------------------------------------

/// Anthropic Messages response body.
#[derive(Debug, Clone, Serialize)]
pub struct MessagesResponse {
    /// Message id (`msg_…`).
    pub id: String,
    /// Always `"message"`.
    #[serde(rename = "type")]
    pub type_: String,
    /// Always `"assistant"`.
    pub role: String,
    /// Output content blocks.
    pub content: Vec<OutputBlock>,
    /// Model id.
    pub model: String,
    /// Why generation stopped.
    pub stop_reason: Option<String>,
    /// The stop sequence hit, if any.
    pub stop_sequence: Option<String>,
    /// Token usage.
    pub usage: AnthropicUsage,
}

/// An output content block.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputBlock {
    /// Generated text.
    Text {
        /// The text.
        text: String,
    },
    /// A tool invocation.
    ToolUse {
        /// Generated id.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input arguments.
        input: Value,
    },
}

/// Anthropic token usage.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicUsage {
    /// Input (prompt) tokens.
    pub input_tokens: usize,
    /// Output (completion) tokens.
    pub output_tokens: usize,
}

/// Anthropic error envelope.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicError {
    /// Always `"error"`.
    #[serde(rename = "type")]
    pub type_: String,
    /// Error detail.
    pub error: AnthropicErrorDetail,
}

/// Detail of an [`AnthropicError`].
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicErrorDetail {
    /// Error category, e.g. `"invalid_request_error"`.
    #[serde(rename = "type")]
    pub type_: String,
    /// Human-readable message.
    pub message: String,
}

fn error_response(status: StatusCode, kind: &str, message: &str) -> axum::response::Response {
    let body = AnthropicError {
        type_: "error".to_string(),
        error: AnthropicErrorDetail {
            type_: kind.to_string(),
            message: message.to_string(),
        },
    };
    (status, Json(body)).into_response()
}

// ----------------------------------------------------------------------------
// Conversion logic (pure, unit-tested)
// ----------------------------------------------------------------------------

/// Flatten an optional system prompt into a single string.
pub fn flatten_system(system: &Option<SystemPrompt>) -> Option<String> {
    match system {
        None => None,
        Some(SystemPrompt::Text(t)) => Some(t.clone()),
        Some(SystemPrompt::Blocks(blocks)) => {
            let text = blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        }
    }
}

/// Render a content value (from a tool_result) into plain text.
fn render_tool_result_content(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    }
}

/// Flatten message content blocks into a prompt-friendly string.
fn flatten_content(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Blocks(blocks) => {
            let mut parts = Vec::new();
            for block in blocks {
                match block {
                    ContentBlock::Text { text } => parts.push(text.clone()),
                    ContentBlock::Image { .. } => parts.push("[image]".to_string()),
                    ContentBlock::ToolUse { name, input, .. } => {
                        parts.push(format!("[tool_use {} {}]", name, input))
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                    } => parts.push(format!(
                        "[tool_result {}: {}]",
                        tool_use_id,
                        render_tool_result_content(content)
                    )),
                }
            }
            parts.join("\n")
        }
    }
}

/// Render the full conversation (system + turns) into a single prompt string.
pub fn render_prompt(request: &MessagesRequest) -> String {
    let mut prompt = String::new();
    if let Some(sys) = flatten_system(&request.system) {
        prompt.push_str("System: ");
        prompt.push_str(&sys);
        prompt.push_str("\n\n");
    }
    for msg in &request.messages {
        let role = if msg.role == "assistant" {
            "Assistant"
        } else {
            "Human"
        };
        prompt.push_str(role);
        prompt.push_str(": ");
        prompt.push_str(&flatten_content(&msg.content));
        prompt.push_str("\n\n");
    }
    prompt.push_str("Assistant:");
    prompt
}

/// Convert Anthropic tools to the shared OpenAI-shaped [`ToolDefinition`]s.
pub fn convert_tools(tools: &[AnthropicTool]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|t| ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        })
        .collect()
}

/// Build the sampling configuration for a request, wiring `tool_choice` through
/// to a JSON-Schema constraint when a specific tool is forced.
pub fn to_sampling_config(request: &MessagesRequest) -> SamplingConfig {
    let mut cfg = SamplingConfig {
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        top_k: request.top_k.unwrap_or(-1),
        max_tokens: request.max_tokens,
        stop: request.stop_sequences.clone().unwrap_or_default(),
        ..Default::default()
    };

    // If a specific tool is forced and it has an input schema, constrain output
    // to that schema (this is Anthropic's structured-output mechanism).
    if let (Some(tools), Some(choice)) = (&request.tools, &request.tool_choice) {
        if let AnthropicToolChoice::Tool { name } = choice {
            if let Some(tool) = tools.iter().find(|t| &t.name == name) {
                if let Some(schema) = &tool.input_schema {
                    cfg.response_format = Some(ResponseFormat::JsonSchema {
                        name: name.clone(),
                        schema: schema.clone(),
                    });
                }
            }
        }
    }
    cfg
}

/// Determine the `stop_reason` for a (placeholder) generation.
pub fn stop_reason(used_tool: bool, hit_max: bool) -> &'static str {
    if used_tool {
        "tool_use"
    } else if hit_max {
        "max_tokens"
    } else {
        "end_turn"
    }
}

fn generate_id(prefix: &str) -> String {
    format!("{}_{}", prefix, &Uuid::new_v4().to_string().replace('-', "")[..24])
}

/// Validate the request, returning a ready error response on failure.
fn validate(request: &MessagesRequest) -> Result<(), axum::response::Response> {
    if request.messages.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "messages: at least one message is required",
        ));
    }
    if request.max_tokens == 0 || request.max_tokens > 128_000 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "max_tokens must be between 1 and 128000",
        ));
    }
    for msg in &request.messages {
        if msg.role != "user" && msg.role != "assistant" {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "message role must be 'user' or 'assistant'",
            ));
        }
    }
    if let Some(t) = request.temperature {
        if !(0.0..=1.0).contains(&t) {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "temperature must be between 0.0 and 1.0",
            ));
        }
    }
    Ok(())
}

// ----------------------------------------------------------------------------
// Handler
// ----------------------------------------------------------------------------

/// `POST /v1/messages` — Anthropic Messages endpoint.
pub async fn messages(
    State(state): State<AppState>,
    Json(request): Json<MessagesRequest>,
) -> impl IntoResponse {
    if let Err(resp) = validate(&request) {
        return resp;
    }

    let id = generate_id("msg");
    let prompt = render_prompt(&request);
    let sampling = to_sampling_config(&request);
    let prompt_tokens = prompt.split_whitespace().count().max(1);

    // Determine whether the request forces a tool call.
    let forced_tool = match (&request.tools, &request.tool_choice) {
        (Some(_), Some(AnthropicToolChoice::Tool { name })) => Some(name.clone()),
        (Some(tools), Some(AnthropicToolChoice::Any)) => {
            tools.first().map(|t| t.name.clone())
        }
        _ => None,
    };

    // Register the request with the engine (mirrors the OpenAI handler; the
    // engine's generation loop is the shared stub boundary in this codebase).
    let inference = Request::new(vec![1, 2, 3]).with_sampling_params(sampling);
    if let Err(e) = state.engine.add_request(inference) {
        tracing::error!("anthropic messages error: {}", e);
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "An internal error occurred.",
        );
    }

    if request.stream {
        return stream_response(id, request.model.clone(), forced_tool, prompt_tokens)
            .into_response();
    }

    let (content, reason) = match forced_tool {
        Some(name) => (
            vec![OutputBlock::ToolUse {
                id: generate_id("toolu"),
                name,
                // Real argument synthesis requires the engine generation loop;
                // the schema constraint (see to_sampling_config) is what keeps
                // those arguments well-formed once generation is wired in.
                input: json!({}),
            }],
            stop_reason(true, false),
        ),
        None => (
            vec![OutputBlock::Text {
                text: "Hello! I'm SwiftLLM serving the Anthropic Messages API.".to_string(),
            }],
            stop_reason(false, false),
        ),
    };

    let response = MessagesResponse {
        id,
        type_: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: request.model,
        stop_reason: Some(reason.to_string()),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: prompt_tokens,
            output_tokens: 12,
        },
    };
    Json(response).into_response()
}

/// Build the native Anthropic SSE event stream.
fn stream_response(
    id: String,
    model: String,
    forced_tool: Option<String>,
    prompt_tokens: usize,
) -> impl IntoResponse {
    let stream = async_stream::stream! {
        // message_start
        let start = json!({
            "type": "message_start",
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": prompt_tokens, "output_tokens": 0}
            }
        });
        yield sse(&start, "message_start");

        if let Some(name) = forced_tool {
            // tool_use block.
            let block_start = json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": generate_id("toolu"), "name": name, "input": {}}
            });
            yield sse(&block_start, "content_block_start");
            let delta = json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{}"}
            });
            yield sse(&delta, "content_block_delta");
            yield sse(&json!({"type": "content_block_stop", "index": 0}), "content_block_stop");
            let msg_delta = json!({
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": null},
                "usage": {"output_tokens": 6}
            });
            yield sse(&msg_delta, "message_delta");
        } else {
            // text block.
            yield sse(
                &json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""}
                }),
                "content_block_start",
            );
            for word in ["Hello!", "I'm", "SwiftLLM", "streaming", "Anthropic", "events."] {
                tokio::time::sleep(Duration::from_millis(20)).await;
                yield sse(
                    &json!({
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": format!("{} ", word)}
                    }),
                    "content_block_delta",
                );
            }
            yield sse(&json!({"type": "content_block_stop", "index": 0}), "content_block_stop");
            yield sse(
                &json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                    "usage": {"output_tokens": 6}
                }),
                "message_delta",
            );
        }

        yield sse(&json!({"type": "message_stop"}), "message_stop");
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Helper: build a named SSE event carrying a JSON payload.
fn sse(payload: &Value, event: &str) -> Result<Event, std::convert::Infallible> {
    Ok(Event::default().event(event).data(payload.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(json_body: Value) -> MessagesRequest {
        serde_json::from_value(json_body).unwrap()
    }

    #[test]
    fn parses_string_and_block_content() {
        let r = req(json!({
            "model": "swiftllm",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
            ]
        }));
        let prompt = render_prompt(&r);
        assert!(prompt.contains("Human: hello"));
        assert!(prompt.contains("Assistant: hi"));
        assert!(prompt.trim_end().ends_with("Assistant:"));
    }

    #[test]
    fn flattens_system_string_and_blocks() {
        let r = req(json!({
            "model": "m", "max_tokens": 10,
            "system": "be terse",
            "messages": [{"role": "user", "content": "x"}]
        }));
        assert_eq!(flatten_system(&r.system).as_deref(), Some("be terse"));

        let r2 = req(json!({
            "model": "m", "max_tokens": 10,
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "messages": [{"role": "user", "content": "x"}]
        }));
        assert_eq!(flatten_system(&r2.system).as_deref(), Some("a\nb"));
    }

    #[test]
    fn renders_tool_result_block() {
        let r = req(json!({
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"}
            ]}]
        }));
        let prompt = render_prompt(&r);
        assert!(prompt.contains("tool_result toolu_1: 42"));
    }

    #[test]
    fn maps_sampling_params() {
        let r = req(json!({
            "model": "m", "max_tokens": 256,
            "temperature": 0.7, "top_p": 0.9, "top_k": 40,
            "stop_sequences": ["STOP"],
            "messages": [{"role": "user", "content": "x"}]
        }));
        let cfg = to_sampling_config(&r);
        assert_eq!(cfg.temperature, 0.7);
        assert_eq!(cfg.top_p, 0.9);
        assert_eq!(cfg.top_k, 40);
        assert_eq!(cfg.max_tokens, 256);
        assert_eq!(cfg.stop, vec!["STOP".to_string()]);
        assert!(cfg.response_format.is_none());
    }

    #[test]
    fn forced_tool_sets_schema_constraint() {
        let r = req(json!({
            "model": "m", "max_tokens": 100,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
            }],
            "tool_choice": {"type": "tool", "name": "get_weather"}
        }));
        let cfg = to_sampling_config(&r);
        match cfg.response_format {
            Some(ResponseFormat::JsonSchema { name, .. }) => assert_eq!(name, "get_weather"),
            other => panic!("expected JsonSchema constraint, got {:?}", other),
        }
    }

    #[test]
    fn converts_tools_to_shared_definitions() {
        let tools = vec![AnthropicTool {
            name: "f".into(),
            description: Some("d".into()),
            input_schema: Some(json!({"type": "object"})),
        }];
        let defs = convert_tools(&tools);
        assert_eq!(defs[0].function.name, "f");
        assert_eq!(defs[0].function.description.as_deref(), Some("d"));
        assert!(defs[0].function.parameters.is_some());
    }

    #[test]
    fn stop_reason_logic() {
        assert_eq!(stop_reason(true, false), "tool_use");
        assert_eq!(stop_reason(false, true), "max_tokens");
        assert_eq!(stop_reason(false, false), "end_turn");
    }

    #[test]
    fn response_serializes_to_anthropic_shape() {
        let resp = MessagesResponse {
            id: "msg_x".into(),
            type_: "message".into(),
            role: "assistant".into(),
            content: vec![OutputBlock::Text { text: "hi".into() }],
            model: "m".into(),
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage { input_tokens: 3, output_tokens: 1 },
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["type"], "message");
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["content"][0]["text"], "hi");
        assert_eq!(v["stop_reason"], "end_turn");
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: anthropic.rs
// REPO PATH:   /swiftllm/crates/swiftllm-server/src/api/anthropic.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
