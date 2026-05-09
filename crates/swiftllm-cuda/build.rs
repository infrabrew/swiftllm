// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      build.rs
// PATH:      /crates/swiftllm-cuda/build.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// Build script for SwiftLLM CUDA kernels.
//
// Compiles all .cu files in kernels/ into a single static library
// (swiftllm_kernels) using the `cc` crate with CUDA support.
//
// Conditional behaviour:
//   • CUDA found (CUDA_PATH / CUDA_HOME / /usr/local/cuda):
//       - Sets cfg(has_cuda)
//       - Compiles all kernel .cu files to swiftllm_kernels static lib
//       - Links cuda + cudart + cublas
//   • CUDA not found:
//       - Emits a warning; cfg(has_cuda) is NOT set
//       - No kernels compiled; stub-only build proceeds
// Licensed under the Apache License, Version 2.0
// ==============================================================================

fn main() {
    // Declare has_cuda as a valid cfg key so Rust doesn't warn about it.
    println!("cargo::rustc-check-cfg=cfg(has_cuda)");

    // ── 1. Locate CUDA installation ──────────────────────────────────────────
    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_include = format!("{}/include", cuda_path);
    let cuda_lib     = format!("{}/lib64",   cuda_path);

    if !std::path::Path::new(&cuda_include).exists() {
        println!("cargo:warning=CUDA not found at '{}'; building without GPU kernel support.", cuda_path);
        println!("cargo:warning=Set CUDA_PATH or CUDA_HOME to enable native CUDA kernels.");
        return;
    }

    // ── 2. Announce CUDA availability ────────────────────────────────────────
    println!("cargo:rustc-cfg=has_cuda");
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=build.rs");

    // ── 3. Link CUDA runtime libraries ───────────────────────────────────────
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");

    // ── 4. Compile .cu kernels via `cc` crate ─────────────────────────────────
    //
    // cc 1.x gained .cuda(true) which invokes nvcc under the hood and produces
    // a static archive that Cargo links automatically.
    //
    // sm_80 = Ampere (A100); sm_86 = GA10x (RTX 30xx); sm_89 = Ada (RTX 40xx)
    // sm_90 = Hopper (H100); sm_120 = Blackwell (RTX PRO 4000 / B200).
    let kernels = [
        "kernels/mamba3_scan.cu",
        "kernels/moe_dispatch.cu",
        "kernels/dense_verif_attn.cu",
        "kernels/rlm_ops.cu",
        "kernels/linear_f16.cu",
        "kernels/paged_attention.cu",
    ];

    // ── Detect CUDA toolkit version for conditional gencode flags ──────────
    let cuda_version_major = detect_cuda_major_version(&cuda_path);

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .flag("-O3")
        .flag("-use_fast_math")
        .flag("--generate-code=arch=compute_80,code=sm_80")
        .flag("--generate-code=arch=compute_86,code=sm_86")
        .flag("--generate-code=arch=compute_89,code=sm_89")
        .flag("--generate-code=arch=compute_90,code=sm_90");

    // sm_120 (Blackwell) requires CUDA 13.0+
    if cuda_version_major >= 13 {
        build.flag("--generate-code=arch=compute_120,code=sm_120");
        println!("cargo:warning=Blackwell (sm_120) target enabled (CUDA {}).", cuda_version_major);
    } else {
        println!("cargo:warning=CUDA {} detected; skipping Blackwell (sm_120). Needs CUDA >= 13.", cuda_version_major);
    }

    build
        .include(&cuda_include)
        .define("__CUDA_NO_HALF_OPERATORS__", None)   // let us use __half2float etc
        .define("CUDA_HAS_FP16", Some("1"));

    for kernel in &kernels {
        let path = std::path::Path::new(kernel);
        if path.exists() {
            build.file(kernel);
        } else {
            println!("cargo:warning=Kernel file '{}' not found, skipping.", kernel);
        }
    }

    build.compile("swiftllm_kernels");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Try to detect the CUDA toolkit major version from `version.json` or
/// `version.txt` in the CUDA installation directory.  Falls back to 12
/// (a safe conservative default that won't emit unsupported gencode flags).
fn detect_cuda_major_version(cuda_path: &str) -> u32 {
    // Method 1: version.json (CUDA 11.6+)
    let version_json = format!("{}/version.json", cuda_path);
    if let Ok(contents) = std::fs::read_to_string(&version_json) {
        // Look for "version" : "13.0.88"  (crude parse — avoid serde dep)
        if let Some(pos) = contents.find("\"version\"") {
            let rest = &contents[pos..];
            // Find first digit sequence after the colon
            if let Some(colon) = rest.find(':') {
                let after_colon = &rest[colon + 1..];
                let trimmed = after_colon.trim().trim_matches('"');
                if let Some(dot) = trimmed.find('.') {
                    if let Ok(major) = trimmed[..dot].parse::<u32>() {
                        return major;
                    }
                }
            }
        }
    }

    // Method 2: version.txt (older toolkits)
    let version_txt = format!("{}/version.txt", cuda_path);
    if let Ok(contents) = std::fs::read_to_string(&version_txt) {
        // e.g. "CUDA Version 12.4.1"
        for word in contents.split_whitespace() {
            if let Some(dot) = word.find('.') {
                if let Ok(major) = word[..dot].parse::<u32>() {
                    return major;
                }
            }
        }
    }

    // Fallback: conservative default
    println!("cargo:warning=Could not detect CUDA version; assuming 12.x");
    12
}

// ------------------------------------------------------------------------------
// END OF FILE: build.rs
// REPO PATH:   /swiftllm/crates/swiftllm-cuda/build.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
