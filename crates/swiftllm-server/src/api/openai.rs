//! OpenAI-compatible API
//!
//! Implements the OpenAI API specification for chat completions and completions.

use crate::streaming::SseStream;
use crate::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    Json,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use swiftllm_core::config::SamplingConfig;
use swiftllm_core::types::Request;
use uuid::Uuid;

/// Chat completion request
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model ID
    pub model: String,

    /// Chat messages
    pub messages: Vec<ChatMessage>,

    /// Temperature (0.0 - 2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p sampling
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Number of completions
    #[serde(default = "default_n")]
    pub n: usize,

    /// Stream response
    #[serde(default)]
    pub stream: bool,

    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,

    /// Maximum tokens to generate
    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// Presence penalty
    #[serde(default)]
    pub presence_penalty: f32,

    /// Frequency penalty
    #[serde(default)]
    pub frequency_penalty: f32,

    /// Logit bias
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>,

    /// User ID
    #[serde(default)]
    pub user: Option<String>,

    /// Random seed
    #[serde(default)]
    pub seed: Option<u64>,

    /// Return log probabilities
    #[serde(default)]
    pub logprobs: Option<bool>,

    /// Top logprobs to return
    #[serde(default)]
    pub top_logprobs: Option<usize>,
}

fn default_temperature() -> f32 { 1.0 }
fn default_top_p() -> f32 { 1.0 }
fn default_n() -> usize { 1 }

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role (system, user, assistant)
    pub role: String,

    /// Message content
    pub content: String,

    /// Optional name
    #[serde(default)]
    pub name: Option<String>,
}

/// Chat completion response
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    /// Response ID
    pub id: String,

    /// Object type
    pub object: String,

    /// Creation timestamp
    pub created: i64,

    /// Model ID
    pub model: String,

    /// System fingerprint
    pub system_fingerprint: String,

    /// Choices
    pub choices: Vec<ChatChoice>,

    /// Usage statistics
    pub usage: Usage,
}

/// Chat choice
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    /// Choice index
    pub index: usize,

    /// Message
    pub message: ChatMessage,

    /// Finish reason
    pub finish_reason: Option<String>,

    /// Log probabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

/// Streaming chat choice delta
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoiceDelta {
    /// Choice index
    pub index: usize,

    /// Delta
    pub delta: ChatMessageDelta,

    /// Finish reason
    pub finish_reason: Option<String>,
}

/// Chat message delta for streaming
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessageDelta {
    /// Role (only in first chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// Content delta
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Streaming response chunk
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    /// Response ID
    pub id: String,

    /// Object type
    pub object: String,

    /// Creation timestamp
    pub created: i64,

    /// Model ID
    pub model: String,

    /// System fingerprint
    pub system_fingerprint: String,

    /// Choices
    pub choices: Vec<ChatChoiceDelta>,
}

/// Token usage
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    /// Prompt tokens
    pub prompt_tokens: usize,

    /// Completion tokens
    pub completion_tokens: usize,

    /// Total tokens
    pub total_tokens: usize,
}

/// Completion request (legacy endpoint)
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    /// Model ID
    pub model: String,

    /// Prompt text
    pub prompt: String,

    /// Maximum tokens
    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// Temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Number of completions
    #[serde(default = "default_n")]
    pub n: usize,

    /// Stream response
    #[serde(default)]
    pub stream: bool,

    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

/// Completion response
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    /// Response ID
    pub id: String,

    /// Object type
    pub object: String,

    /// Creation timestamp
    pub created: i64,

    /// Model ID
    pub model: String,

    /// Choices
    pub choices: Vec<CompletionChoice>,

    /// Usage
    pub usage: Usage,
}

/// Completion choice
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    /// Text
    pub text: String,

    /// Choice index
    pub index: usize,

    /// Finish reason
    pub finish_reason: Option<String>,
}

/// Model info
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    /// Model ID
    pub id: String,

    /// Object type
    pub object: String,

    /// Created timestamp
    pub created: i64,

    /// Owner
    pub owned_by: String,
}

/// Models list response
#[derive(Debug, Clone, Serialize)]
pub struct ModelsResponse {
    /// Object type
    pub object: String,

    /// Models
    pub data: Vec<ModelInfo>,
}

/// Error response
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    /// Error details
    pub error: ErrorDetail,
}

/// Error detail
#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    /// Error message
    pub message: String,

    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,

    /// Error code
    pub code: Option<String>,
}

/// OpenAI API implementation
pub struct OpenAIApi;

impl OpenAIApi {
    /// Generate a response ID
    fn generate_id(prefix: &str) -> String {
        format!("{}-{}", prefix, Uuid::new_v4().to_string().replace("-", "")[..24].to_string())
    }

    /// Get system fingerprint
    fn system_fingerprint() -> String {
        format!("fp_{}", &Uuid::new_v4().to_string().replace("-", "")[..12])
    }

    /// Convert sampling parameters
    fn to_sampling_config(
        temperature: f32,
        top_p: f32,
        max_tokens: Option<usize>,
        stop: Option<Vec<String>>,
        presence_penalty: f32,
        frequency_penalty: f32,
        seed: Option<u64>,
    ) -> SamplingConfig {
        SamplingConfig {
            temperature,
            top_p,
            max_tokens: max_tokens.unwrap_or(256),
            stop: stop.unwrap_or_default(),
            presence_penalty,
            frequency_penalty,
            seed,
            ..Default::default()
        }
    }
}

/// Chat completions endpoint
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let id = OpenAIApi::generate_id("chatcmpl");
    let created = Utc::now().timestamp();
    let system_fingerprint = OpenAIApi::system_fingerprint();

    // Convert messages to prompt tokens
    // In a real implementation, we would use a tokenizer
    let prompt_tokens: Vec<u32> = vec![1, 2, 3]; // Placeholder

    // Create sampling config
    let sampling_config = OpenAIApi::to_sampling_config(
        request.temperature,
        request.top_p,
        request.max_tokens,
        request.stop,
        request.presence_penalty,
        request.frequency_penalty,
        request.seed,
    );

    // Create inference request
    let inference_request = Request::new(prompt_tokens.clone())
        .with_sampling_params(sampling_config);

    // Add to engine
    match state.engine.add_request(inference_request) {
        Ok(request_id) => {
            if request.stream {
                // Streaming response
                // In a real implementation, we would stream tokens as they're generated
                let stream = async_stream::stream! {
                    // First chunk with role
                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: request.model.clone(),
                        system_fingerprint: system_fingerprint.clone(),
                        choices: vec![ChatChoiceDelta {
                            index: 0,
                            delta: ChatMessageDelta {
                                role: Some("assistant".to_string()),
                                content: None,
                            },
                            finish_reason: None,
                        }],
                    };
                    yield Ok::<_, std::convert::Infallible>(
                        Event::default().data(serde_json::to_string(&chunk).unwrap())
                    );

                    // Simulate streaming content
                    let response_text = "Hello! I'm SwiftLLM, ready to help you.";
                    for word in response_text.split_whitespace() {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        let chunk = ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: request.model.clone(),
                            system_fingerprint: system_fingerprint.clone(),
                            choices: vec![ChatChoiceDelta {
                                index: 0,
                                delta: ChatMessageDelta {
                                    role: None,
                                    content: Some(format!("{} ", word)),
                                },
                                finish_reason: None,
                            }],
                        };
                        yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
                    }

                    // Final chunk
                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: request.model.clone(),
                        system_fingerprint: system_fingerprint.clone(),
                        choices: vec![ChatChoiceDelta {
                            index: 0,
                            delta: ChatMessageDelta {
                                role: None,
                                content: None,
                            },
                            finish_reason: Some("stop".to_string()),
                        }],
                    };
                    yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));

                    // Done marker
                    yield Ok(Event::default().data("[DONE]"));
                };

                Sse::new(stream)
                    .keep_alive(KeepAlive::default())
                    .into_response()
            } else {
                // Non-streaming response
                let response = ChatCompletionResponse {
                    id,
                    object: "chat.completion".to_string(),
                    created,
                    model: request.model,
                    system_fingerprint,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: "Hello! I'm SwiftLLM, a high-performance inference engine. How can I help you today?".to_string(),
                            name: None,
                        },
                        finish_reason: Some("stop".to_string()),
                        logprobs: None,
                    }],
                    usage: Usage {
                        prompt_tokens: prompt_tokens.len(),
                        completion_tokens: 15,
                        total_tokens: prompt_tokens.len() + 15,
                    },
                };

                Json(response).into_response()
            }
        }
        Err(e) => {
            let error = ErrorResponse {
                error: ErrorDetail {
                    message: e.to_string(),
                    error_type: "server_error".to_string(),
                    code: Some("internal_error".to_string()),
                },
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response()
        }
    }
}

/// Completions endpoint (legacy)
pub async fn completions(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> impl IntoResponse {
    let id = OpenAIApi::generate_id("cmpl");
    let created = Utc::now().timestamp();

    let response = CompletionResponse {
        id,
        object: "text_completion".to_string(),
        created,
        model: request.model,
        choices: vec![CompletionChoice {
            text: " SwiftLLM completion response.".to_string(),
            index: 0,
            finish_reason: Some("stop".to_string()),
        }],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        },
    };

    Json(response)
}

/// List models endpoint
pub async fn list_models(State(_state): State<AppState>) -> impl IntoResponse {
    let response = ModelsResponse {
        object: "list".to_string(),
        data: vec![
            ModelInfo {
                id: "swiftllm-default".to_string(),
                object: "model".to_string(),
                created: Utc::now().timestamp(),
                owned_by: "swiftllm".to_string(),
            },
        ],
    };

    Json(response)
}

/// Get model endpoint
pub async fn get_model(
    State(_state): State<AppState>,
    Path(model_id): Path<String>,
) -> impl IntoResponse {
    let model = ModelInfo {
        id: model_id,
        object: "model".to_string(),
        created: Utc::now().timestamp(),
        owned_by: "swiftllm".to_string(),
    };

    Json(model)
}
