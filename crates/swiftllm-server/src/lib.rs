//! SwiftLLM Server
//!
//! HTTP server providing an OpenAI-compatible API for LLM inference.

#![warn(clippy::all)]

pub mod api;
pub mod streaming;

use api::openai::OpenAIApi;
use axum::{
    body::Body,
    extract::State,
    http::{header, Method, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
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

/// Maximum request body size (10 MB)
const MAX_REQUEST_BODY_SIZE: usize = 10 * 1024 * 1024;

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

/// API key authentication middleware
async fn auth_middleware(
    State(state): State<AppState>,
    req: Request<Body>,
    next: Next,
) -> std::result::Result<Response, StatusCode> {
    // Skip auth for health check and metrics
    let path = req.uri().path();
    if path == "/health" || path == "/v1/health" || path == "/metrics" {
        return Ok(next.run(req).await);
    }

    // If no API key configured, skip auth
    let expected_key = match &state.api_key {
        Some(key) => key,
        None => return Ok(next.run(req).await),
    };

    // Check Authorization header
    let auth_header = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(header_val) => {
            let token = header_val.strip_prefix("Bearer ").unwrap_or(header_val);
            if token == expected_key {
                Ok(next.run(req).await)
            } else {
                tracing::warn!("Invalid API key from {:?}", req.headers().get("x-forwarded-for"));
                Err(StatusCode::UNAUTHORIZED)
            }
        }
        None => {
            tracing::warn!("Missing Authorization header");
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

/// Security headers middleware
async fn security_headers(req: Request<Body>, next: Next) -> Response {
    let mut response = next.run(req).await;
    let headers = response.headers_mut();

    headers.insert("x-content-type-options", "nosniff".parse().unwrap());
    headers.insert("x-frame-options", "DENY".parse().unwrap());
    headers.insert(
        "strict-transport-security",
        "max-age=31536000; includeSubDomains".parse().unwrap(),
    );
    headers.insert(
        "cache-control",
        "no-store, no-cache, must-revalidate".parse().unwrap(),
    );
    headers.insert("x-request-id", uuid::Uuid::new_v4().to_string().parse().unwrap());

    response
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
        // State and middleware
        .with_state(state.clone())
        .layer(middleware::from_fn_with_state(state, auth_middleware))
        .layer(middleware::from_fn(security_headers))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .layer(
            tower_http::limit::RequestBodyLimitLayer::new(MAX_REQUEST_BODY_SIZE),
        )
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
    if config.api_key.is_some() {
        tracing::info!("API key authentication enabled");
    }

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
