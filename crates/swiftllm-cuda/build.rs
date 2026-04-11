// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      build.rs
// PATH:      /crates/swiftllm-cuda/build.rs
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

//! Build script for CUDA kernels

fn main() {
    // Check for CUDA installation
    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_include = format!("{}/include", cuda_path);

    if std::path::Path::new(&cuda_include).exists() {
        println!("cargo:rustc-cfg=has_cuda");
        println!("cargo:rerun-if-changed=kernels/");

        // In a real build, we would compile CUDA kernels here using nvcc
        // For now, we just set up the include paths

        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
    } else {
        println!("cargo:warning=CUDA not found, building without GPU support");
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: build.rs
// REPO PATH:   /swiftllm/crates/swiftllm-cuda/build.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
