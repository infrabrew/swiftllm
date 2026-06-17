// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      lib.rs
// PATH:      /crates/swiftllm-server/src/lib.rs
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

//! SwiftLLM Server
//!
//! HTTP server providing an OpenAI-compatible API for LLM inference.

#![warn(clippy::all)]

pub mod api;
pub mod streaming;

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
use subtle::ConstantTimeEq;
use swiftllm_core::config::ServerConfig;
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
    // Skip auth for health check only — metrics requires auth
    let path = req.uri().path();
    if path == "/health" || path == "/v1/health" {
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
            // Constant-time comparison to prevent timing side-channel attacks.
            // We first check lengths (which leaks only the length, not content),
            // then compare bytes in constant time.
            let token_bytes = token.as_bytes();
            let expected_bytes = expected_key.as_bytes();
            let length_match = token_bytes.len() == expected_bytes.len();
            let content_match = token_bytes.ct_eq(expected_bytes).into();
            if length_match && content_match {
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
    // Build CORS layer from config — default to localhost-only, not Any.
    // This prevents cross-origin credential theft from arbitrary websites.
    let cors = {
        let origins = &state.config.cors_origins;
        let is_wildcard = origins.iter().any(|o| o == "*");
        if is_wildcard {
            tracing::warn!(
                "CORS configured with wildcard origin ('*'). \
                 This allows ANY website to make cross-origin requests to this server. \
                 Consider restricting cors_origins in production."
            );
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
        } else {
            let allowed: Vec<_> = origins
                .iter()
                .filter_map(|o| o.parse::<axum::http::HeaderValue>().ok())
                .collect();
            CorsLayer::new()
                .allow_origin(allowed)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
        }
    };

    Router::new()
        // Health check
        .route("/health", get(health_check))
        .route("/v1/health", get(health_check))
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(api::openai::chat_completions))
        .route("/v1/completions", post(api::openai::completions))
        .route("/v1/models", get(api::openai::list_models))
        .route("/v1/models/:model_id", get(api::openai::get_model))
        // Anthropic-compatible Messages endpoint
        .route("/v1/messages", post(api::anthropic::messages))
        .route("/v1/messages/count_tokens", post(api::anthropic::count_tokens))
        // Realtime bidirectional WebSocket endpoint
        .route("/v1/realtime", get(api::realtime::realtime_ws))
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

/// Metrics endpoint — returns JSON by default, Prometheus text format if Accept header requests it
async fn metrics(State(state): State<AppState>, req: Request<Body>) -> Response {
    let stats = state.engine.stats();

    // Check if Prometheus format is requested
    let wants_prometheus = req
        .headers()
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.contains("text/plain") || v.contains("openmetrics"))
        .unwrap_or(false);

    if wants_prometheus {
        let body = format!(
            "# HELP swiftllm_requests_running Number of running requests\n\
             # TYPE swiftllm_requests_running gauge\n\
             swiftllm_requests_running {}\n\
             # HELP swiftllm_requests_waiting Number of waiting requests\n\
             # TYPE swiftllm_requests_waiting gauge\n\
             swiftllm_requests_waiting {}\n\
             # HELP swiftllm_requests_completed_total Total completed requests\n\
             # TYPE swiftllm_requests_completed_total counter\n\
             swiftllm_requests_completed_total {}\n\
             # HELP swiftllm_throughput_tokens_per_second Token throughput\n\
             # TYPE swiftllm_throughput_tokens_per_second gauge\n\
             swiftllm_throughput_tokens_per_second {:.2}\n\
             # HELP swiftllm_gpu_memory_utilization GPU memory utilization\n\
             # TYPE swiftllm_gpu_memory_utilization gauge\n\
             swiftllm_gpu_memory_utilization {:.4}\n\
             # HELP swiftllm_gpu_blocks_free Free GPU blocks\n\
             # TYPE swiftllm_gpu_blocks_free gauge\n\
             swiftllm_gpu_blocks_free {}\n\
             # HELP swiftllm_steps_total Total engine steps\n\
             # TYPE swiftllm_steps_total counter\n\
             swiftllm_steps_total {}\n",
            stats.scheduler.running_requests,
            stats.scheduler.waiting_requests,
            stats.scheduler.completed_requests,
            stats.execution.tokens_per_second,
            stats.block_manager.gpu_utilization,
            stats.block_manager.free_gpu_blocks,
            stats.step_count,
        );
        Response::builder()
            .header(header::CONTENT_TYPE, "text/plain; version=0.0.4")
            .body(Body::from(body))
            .unwrap()
    } else {
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
                "total_forward_passes": stats.execution.total_forward_passes,
                "step_count": stats.step_count
            }
        }))
        .into_response()
    }
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
    } else {
        tracing::warn!(
            "No API key configured — all endpoints are unauthenticated. \
             Set --api-key or SWIFTLLM_API_KEY for production use."
        );
    }

    // Warn about network exposure
    if config.host == "0.0.0.0" {
        tracing::warn!(
            "Server is binding to 0.0.0.0 (all interfaces). \
             This exposes the server on all network interfaces including public ones. \
             Without TLS, API keys and inference data are transmitted in plaintext. \
             Use --host 127.0.0.1 for local-only access."
        );
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

// ------------------------------------------------------------------------------
// END OF FILE: lib.rs
// REPO PATH:   /swiftllm/crates/swiftllm-server/src/lib.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
