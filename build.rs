use std::fs;
use std::path::Path;
use std::{env, io};
fn copy_recursive(from: &Path, to: &Path) -> io::Result<()> {
    if !to.exists() {
        fs::create_dir_all(&to)?;
    }
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let to_path = to.join(entry.file_name());

        if file_type.is_dir() {
            copy_recursive(&entry.path(), &to_path)?;
        } else {
            fs::copy(&entry.path(), &to_path)?;
        }
    }
    Ok(())
}

fn main() {
    copy_dir("training_data", "training_data");
    copy_dir("tests", "tests");

    let openblas_path = env::var("OPENBLAS_DIR").unwrap_or_else(|_| r"C:\\Users\\jeti8\\OneDrive\\Desktop\\Rodion Projects\\OpenBLAS-0.3.29_x64".into());
    let openblas_lib = format!("{}\\lib", openblas_path);
    println!("cargo:rustc-link-search=native={}", openblas_lib);
    println!("cargo:rustc-link-lib=dylib=openblas");

    // Add CUDA library path
    // Common CUDA installation paths on Windows
    // IMPORTANT: Use CUDA 12.9 to match driver version (nvidia-smi shows CUDA 12.9)
    // FORCE CUDA 12.9 - ignore CUDA_PATH environment variable that may point to v13.0
    let cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9";
    if !Path::new(cuda_path).exists() {
        panic!("CUDA 12.9 not found at: {}", cuda_path);
    }
    
    let cuda_lib_path = format!("{}\\lib\\x64", cuda_path);
    let cuda_bin_path = format!("{}\\bin", cuda_path);
    
    println!("cargo:warning=CUDA bin path: {}", cuda_bin_path);
    
    println!("cargo:rustc-link-search=native={}", cuda_lib_path);
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    
    // Copy CUDA DLLs to output directory to ensure correct version is loaded
    let out_dir = env::var("OUT_DIR").unwrap();
    let target_dir = Path::new(&out_dir).join("../../..").canonicalize().unwrap();
    
    // Include all essential CUDA DLLs for deep learning
    let dlls = [
        "cudart64_12.dll", 
        "cublas64_12.dll", 
        "cublasLt64_12.dll",
        "nvrtc64_120_0.dll",
        "nvrtc-builtins64_129.dll",
        "curand64_10.dll",
        "cusparse64_12.dll",
        "cusolver64_11.dll",
        "cufft64_11.dll"
    ];
    for dll in &dlls {
        let src = Path::new(&cuda_bin_path).join(dll);
        let dst = target_dir.join(dll);
        if src.exists() {
            let _ = fs::copy(&src, &dst);
            println!("cargo:warning=Copied {} to output directory", dll);
        } else {
            println!("cargo:warning=DLL not found: {}", dll);
        }
    }
    
    // Also copy OpenBLAS DLL
    let openblas_dll_path = r"C:\Users\rkovalenko\Desktop\OpenBLAS-0.3.29_x64_64\bin\libopenblas.dll";
    if Path::new(openblas_dll_path).exists() {
        let dst = target_dir.join("openblas.dll");
        let _ = fs::copy(openblas_dll_path, &dst);
        println!("cargo:warning=Copied OpenBLAS to output directory");
    }
}

pub fn copy_dir(file_dir_path: &str, target_dir_path: &str) {
    let out_dir = env::var("OUT_DIR").unwrap();
    let target_dir = Path::new(&out_dir).join("../../..").canonicalize().unwrap();

    let source_dir = Path::new(file_dir_path);

    let destination = target_dir.join(target_dir_path);
    if destination.exists() {
        fs::remove_dir_all(&destination).unwrap();
    }

    copy_recursive(&source_dir, &destination).unwrap();
}
