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

    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9".into());

    let cuda_lib_path = format!("{}\\lib\\x64", cuda_path);
    if !Path::new(&cuda_lib_path).exists() {
        panic!("CUDA library path not found: {}", cuda_lib_path);
    }

    println!("cargo:rustc-link-search=native={}", cuda_lib_path);
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");

    // Optional: emit path warnings to debug build.rs output
    println!("cargo:warning=Linked against CUDA from: {}", cuda_lib_path);
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
