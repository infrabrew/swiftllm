// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      api_integration.rs
// PATH:      /crates/swiftllm-server/tests/api_integration.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// End-to-end HTTP tests for the Phase 1 API features. These drive the real axum
// router (routing, extractors, validation, serialization) via `tower::oneshot`
// — no network socket and no model required, since the engine queues requests
// without running a generation loop.
// ==============================================================================

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use std::sync::Arc;
use swiftllm_core::config::{EngineConfig, ServerConfig};
use swiftllm_core::engine::Engine;
use swiftllm_server::{create_router, AppState};
use tower::ServiceExt; // for `oneshot`

/// Build a router backed by a real (model-less) engine.
fn test_app() -> axum::Router {
    let engine = Arc::new(Engine::new(EngineConfig::default()).expect("engine"));
    let state = AppState {
        engine,
        config: ServerConfig::default(),
        api_key: None,
    };
    create_router(state)
}

/// POST `body` as JSON to `uri` and return (status, parsed-json-body).
async fn post_json(uri: &str, body: Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let response = test_app().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

async fn get(uri: &str) -> (StatusCode, Value) {
    let request = Request::builder().uri(uri).body(Body::empty()).unwrap();
    let response = test_app().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

#[tokio::test]
async fn health_endpoint_ok() {
    let (status, body) = get("/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn models_endpoint_lists_default() {
    let (status, body) = get("/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "swiftllm-default");
}

#[tokio::test]
async fn anthropic_messages_text_response() {
    let (status, body) = post_json(
        "/v1/messages",
        json!({
            "model": "swiftllm",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["stop_reason"], "end_turn");
    assert!(body["usage"]["input_tokens"].as_u64().is_some());
}

#[tokio::test]
async fn anthropic_forced_tool_use() {
    let (status, body) = post_json(
        "/v1/messages",
        json!({
            "model": "swiftllm",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }],
            "tool_choice": {"type": "tool", "name": "get_weather"}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["content"][0]["type"], "tool_use");
    assert_eq!(body["content"][0]["name"], "get_weather");
    assert_eq!(body["stop_reason"], "tool_use");
}

#[tokio::test]
async fn anthropic_validation_rejects_empty_messages() {
    let (status, body) = post_json(
        "/v1/messages",
        json!({"model": "m", "max_tokens": 16, "messages": []}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["type"], "error");
}

#[tokio::test]
async fn openai_chat_forced_tool_call() {
    let (status, body) = post_json(
        "/v1/chat/completions",
        json!({
            "model": "swiftllm",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            }],
            "tool_choice": "required"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let choice = &body["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    assert_eq!(choice["message"]["tool_calls"][0]["function"]["name"], "get_weather");
    // content is null on a tool-call turn.
    assert!(choice["message"]["content"].is_null());
}

#[tokio::test]
async fn openai_chat_with_json_schema_response_format() {
    // The structured-output constraint is accepted and plumbed through to the
    // sampling config; the endpoint responds successfully.
    let (status, body) = post_json(
        "/v1/chat/completions",
        json!({
            "model": "swiftllm",
            "messages": [{"role": "user", "content": "give me json"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
                }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn openai_chat_plain_text_response() {
    let (status, body) = post_json(
        "/v1/chat/completions",
        json!({
            "model": "swiftllm",
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert!(body["choices"][0]["message"]["content"].is_string());
}

// ------------------------------------------------------------------------------
// END OF FILE: api_integration.rs
// REPO PATH:   /swiftllm/crates/swiftllm-server/tests/api_integration.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
