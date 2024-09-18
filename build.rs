use std::{env, io};
use std::fs;
use std::path::Path;
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

// trait CopyRecursive {
//     fn copy_recursive(&self, destination: &Path) -> std::io::Result<()>;
// }

// impl CopyRecursive for Path {
//     fn copy_recursive(&self, destination: &Path) -> std::io::Result<()> {
//         if !self.exists() {
//             return Err(std::io::Error::new(
//                 std::io::ErrorKind::NotFound,
//                 format!("Source path does not exist: {:?}", self),
//             ));
//         }
//         if self.is_dir() {
//             fs::create_dir_all(destination)?;
//             for entry in fs::read_dir(self)? {
//                 let entry = entry?;
//                 let file_type = entry.file_type()?;
//                 let entry_path = entry.path();
//                 let destination_path = destination.join(entry.file_name());
//                 if file_type.is_dir() {
//                     entry_path.copy_recursive(&destination_path)?;
//                 } else {
//                     fs::copy(&entry_path, &destination_path)?;
//                 }
//             }
//         } else {
//             if let Some(parent) = destination.parent() {
//                 fs::create_dir_all(parent)?;
//             }
//             fs::copy(self, destination)?;
//         }
//         Ok(())
//     }
// }
