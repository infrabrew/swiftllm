//! SwiftLLM Server - Main Entry Point
//!
//! High-performance LLM inference server with OpenAI-compatible API.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use swiftllm_core::config::{EngineConfig, ServerConfig};
use swiftllm_core::engine::Engine;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// SwiftLLM - High-performance LLM inference engine
#[derive(Parser)]
#[command(name = "swiftllm")]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference server
    Serve(ServeArgs),

    /// Run benchmarks
    Benchmark(BenchmarkArgs),

    /// Convert model format
    Convert(ConvertArgs),

    /// Show model information
    Info(InfoArgs),
}

/// Arguments for the serve command
#[derive(Parser)]
struct ServeArgs {
    /// Path to the model
    #[arg(short, long)]
    model: PathBuf,

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind to
    #[arg(short, long, default_value = "8000")]
    port: u16,

    /// Tensor parallel size
    #[arg(long, default_value = "1")]
    tensor_parallel_size: usize,

    /// Maximum sequence length
    #[arg(long, default_value = "4096")]
    max_seq_len: usize,

    /// GPU memory utilization (0.0 - 1.0)
    #[arg(long, default_value = "0.90")]
    gpu_memory_utilization: f32,

    /// Block size for PagedAttention
    #[arg(long, default_value = "16")]
    block_size: usize,

    /// Swap space in GiB
    #[arg(long, default_value = "4")]
    swap_space: f32,

    /// Enable prefix caching
    #[arg(long)]
    enable_prefix_caching: bool,

    /// Draft model for speculative decoding
    #[arg(long)]
    speculative_model: Option<PathBuf>,

    /// Number of speculative tokens
    #[arg(long, default_value = "5")]
    speculative_tokens: usize,

    /// API key (optional)
    #[arg(long)]
    api_key: Option<String>,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Trust remote code
    #[arg(long)]
    trust_remote_code: bool,

    /// Data type
    #[arg(long, default_value = "float16")]
    dtype: String,
}

/// Arguments for the benchmark command
#[derive(Parser)]
struct BenchmarkArgs {
    /// Path to the model
    #[arg(short, long)]
    model: PathBuf,

    /// Input length
    #[arg(long, default_value = "128")]
    input_len: usize,

    /// Output length
    #[arg(long, default_value = "128")]
    output_len: usize,

    /// Number of requests
    #[arg(long, default_value = "100")]
    num_requests: usize,

    /// Concurrent requests
    #[arg(long, default_value = "10")]
    concurrency: usize,
}

/// Arguments for the convert command
#[derive(Parser)]
struct ConvertArgs {
    /// Input model path
    #[arg(short, long)]
    input: PathBuf,

    /// Output path
    #[arg(short, long)]
    output: PathBuf,

    /// Output format
    #[arg(long, default_value = "safetensors")]
    format: String,

    /// Quantization
    #[arg(long)]
    quantize: Option<String>,
}

/// Arguments for the info command
#[derive(Parser)]
struct InfoArgs {
    /// Path to the model
    #[arg(short, long)]
    model: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve(args) => serve(args).await,
        Commands::Benchmark(args) => benchmark(args).await,
        Commands::Convert(args) => convert(args).await,
        Commands::Info(args) => info(args).await,
    }
}

async fn serve(args: ServeArgs) -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| args.log_level.clone().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("SwiftLLM v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Loading model from: {}", args.model.display());

    // Create engine configuration
    let engine_config = EngineConfig {
        model: swiftllm_core::config::ModelConfig {
            path: args.model,
            max_seq_len: args.max_seq_len,
            trust_remote_code: args.trust_remote_code,
            ..Default::default()
        },
        device: swiftllm_core::config::DeviceConfig {
            tensor_parallel_size: args.tensor_parallel_size,
            ..Default::default()
        },
        memory: swiftllm_core::config::MemoryConfig {
            block_size: args.block_size,
            gpu_memory_utilization: args.gpu_memory_utilization,
            swap_space_gib: args.swap_space,
            enable_prefix_caching: args.enable_prefix_caching,
            ..Default::default()
        },
        speculative: args.speculative_model.map(|path| {
            swiftllm_core::config::SpeculativeConfig {
                draft_model_path: path,
                num_speculative_tokens: args.speculative_tokens,
                ..Default::default()
            }
        }),
        ..Default::default()
    };

    // Create engine
    let engine = Arc::new(Engine::new(engine_config)?);

    // Create server configuration
    let server_config = ServerConfig {
        host: args.host,
        port: args.port,
        api_key: args.api_key,
        ..Default::default()
    };

    tracing::info!(
        "Server starting on {}:{}",
        server_config.host,
        server_config.port
    );

    // Start server
    swiftllm_server::start_server(engine, server_config).await?;

    Ok(())
}

async fn benchmark(args: BenchmarkArgs) -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    tracing::info!("Running benchmark with model: {}", args.model.display());
    tracing::info!(
        "Input length: {}, Output length: {}, Requests: {}",
        args.input_len,
        args.output_len,
        args.num_requests
    );

    // TODO: Implement benchmark
    tracing::warn!("Benchmark not yet implemented");

    Ok(())
}

async fn convert(args: ConvertArgs) -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    tracing::info!("Converting model from {} to {}", args.input.display(), args.output.display());
    tracing::info!("Output format: {}", args.format);

    if let Some(ref quant) = args.quantize {
        tracing::info!("Quantization: {}", quant);
    }

    // TODO: Implement conversion
    tracing::warn!("Model conversion not yet implemented");

    Ok(())
}

async fn info(args: InfoArgs) -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    tracing::info!("Model information for: {}", args.model.display());

    // Try to load config
    match swiftllm_models::load_config(&args.model) {
        Ok(config) => {
            println!("Architecture: {:?}", config.architecture);
            println!("Hidden size: {}", config.hidden_size);
            println!("Intermediate size: {}", config.intermediate_size);
            println!("Num attention heads: {}", config.num_attention_heads);
            println!("Num KV heads: {}", config.num_key_value_heads);
            println!("Num layers: {}", config.num_hidden_layers);
            println!("Vocab size: {}", config.vocab_size);
            println!("Max position embeddings: {}", config.max_position_embeddings);
            println!("RoPE theta: {}", config.rope_theta);

            if let Some(window) = config.sliding_window {
                println!("Sliding window: {}", window);
            }

            // Calculate model size
            let params = estimate_params(&config);
            println!("\nEstimated parameters: {:.2}B", params as f64 / 1e9);

            // Estimate memory requirements
            let fp16_size = params * 2;
            let int8_size = params;
            let int4_size = params / 2;

            println!("\nEstimated memory:");
            println!("  FP16: {:.2} GB", fp16_size as f64 / 1e9);
            println!("  INT8: {:.2} GB", int8_size as f64 / 1e9);
            println!("  INT4: {:.2} GB", int4_size as f64 / 1e9);
        }
        Err(e) => {
            tracing::error!("Failed to load model config: {}", e);
        }
    }

    Ok(())
}

fn estimate_params(config: &swiftllm_models::ModelConfig) -> usize {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let layers = config.num_hidden_layers;
    let vocab = config.vocab_size;
    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;

    // Embedding
    let embed = vocab * hidden;

    // Attention per layer
    let q_proj = hidden * heads * head_dim;
    let k_proj = hidden * kv_heads * head_dim;
    let v_proj = hidden * kv_heads * head_dim;
    let o_proj = heads * head_dim * hidden;
    let attn = q_proj + k_proj + v_proj + o_proj;

    // MLP per layer
    let gate = hidden * intermediate;
    let up = hidden * intermediate;
    let down = intermediate * hidden;
    let mlp = gate + up + down;

    // Norms per layer
    let norms = 2 * hidden;

    // Total per layer
    let per_layer = attn + mlp + norms;

    // Final norm + LM head
    let final_norm = hidden;
    let lm_head = hidden * vocab;

    embed + layers * per_layer + final_norm + lm_head
}
