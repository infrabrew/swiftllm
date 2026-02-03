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
