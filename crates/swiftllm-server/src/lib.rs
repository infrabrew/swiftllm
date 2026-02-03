//! SwiftLLM Server
//!
//! HTTP server providing an OpenAI-compatible API for LLM inference.

#![warn(clippy::all)]

pub mod api;
pub mod streaming;

use api::openai::OpenAIApi;
use axum::{
    extract::State,
    http::{header, Method, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::net::SocketAddr;
use std::sync::Arc;
use swiftllm_core::config::{EngineConfig, ServerConfig};
use swiftllm_core::engine::Engine;
use swiftllm_core::error::Result;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

/// Server state shared across handlers
#[derive(Clone)]
pub struct AppState {
    /// Inference engine
    pub engine: Arc<Engine>,

    /// Server configuration
    pub config: ServerConfig,

    /// API key (optional)
    pub api_key: Option<String>,
}

/// Create the API router
pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    Router::new()
        // Health check
        .route("/health", get(health_check))
        .route("/v1/health", get(health_check))
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(api::openai::chat_completions))
        .route("/v1/completions", post(api::openai::completions))
        .route("/v1/models", get(api::openai::list_models))
        .route("/v1/models/:model_id", get(api::openai::get_model))
        // Metrics
        .route("/metrics", get(metrics))
        // State
        .with_state(state)
        // Middleware
        .layer(cors)
        .layer(TraceLayer::new_for_http())
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Metrics endpoint
async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    let stats = state.engine.stats();
    Json(serde_json::json!({
        "scheduler": {
            "running_requests": stats.scheduler.running_requests,
            "waiting_requests": stats.scheduler.waiting_requests,
            "completed_requests": stats.scheduler.completed_requests,
            "throughput_tps": stats.execution.tokens_per_second
        },
        "memory": {
            "gpu_utilization": stats.block_manager.gpu_utilization,
            "cpu_utilization": stats.block_manager.cpu_utilization,
            "free_gpu_blocks": stats.block_manager.free_gpu_blocks,
            "free_cpu_blocks": stats.block_manager.free_cpu_blocks
        },
        "execution": {
            "prefill_tokens": stats.execution.prefill_tokens,
            "decode_tokens": stats.execution.decode_tokens,
            "total_forward_passes": stats.execution.total_forward_passes
        }
    }))
}

/// Start the server
pub async fn start_server(
    engine: Arc<Engine>,
    config: ServerConfig,
) -> Result<()> {
    let state = AppState {
        engine: engine.clone(),
        api_key: config.api_key.clone(),
        config: config.clone(),
    };

    let app = create_router(state);

    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|e| swiftllm_core::error::Error::Internal(format!("Invalid address: {}", e)))?;

    tracing::info!("Starting server on {}", addr);

    // Create listener
    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        swiftllm_core::error::Error::Internal(format!("Failed to bind: {}", e))
    })?;

    // Serve
    axum::serve(listener, app).await.map_err(|e| {
        swiftllm_core::error::Error::Internal(format!("Server error: {}", e))
    })?;

    Ok(())
}
