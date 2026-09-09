//! MNIST loading and the 28x28 -> 32x32 zero padding. Blueprint section A1.
//!
//! Files are read from `datasets/mnist/`. Both the raw `idx-ubyte` files and
//! their gzipped form are accepted; if neither is present the loader downloads
//! them once from a public mirror.

use crate::neural_networks::spectral_model::bands::N;
use crate::neural_networks::utils::dtype::Real;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const MNIST_DIR: &str = "datasets/mnist";
pub const IMG_SIDE: usize = 28;

const TRAIN_IMAGES: &str = "train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte";
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte";
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte";

/// Mirrors that still serve the original `idx-ubyte.gz` files.
const MIRRORS: [&str; 2] = [
    "https://ossci-datasets.s3.amazonaws.com/mnist",
    "https://storage.googleapis.com/cvdf-datasets/mnist",
];

/// A padded MNIST split: images already converted to `f32` in `0..1` and
/// zero-padded to `N x N`, stored row-major.
#[derive(Debug, Clone)]
pub struct MnistSplit {
    pub images: Vec<Vec<Real>>,
    pub labels: Vec<u8>,
}

impl MnistSplit {
    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }
}

/// Load the training split, downloading it first if necessary.
pub fn load_train() -> std::io::Result<MnistSplit> {
    load_split(TRAIN_IMAGES, TRAIN_LABELS)
}

/// Load the 10k test split.
pub fn load_test() -> std::io::Result<MnistSplit> {
    load_split(TEST_IMAGES, TEST_LABELS)
}

fn load_split(images_name: &str, labels_name: &str) -> std::io::Result<MnistSplit> {
    let images_raw = read_idx(&ensure_file(images_name)?)?;
    let labels_raw = read_idx(&ensure_file(labels_name)?)?;

    let (count, rows, cols) = parse_image_header(&images_raw)?;
    if rows != IMG_SIDE || cols != IMG_SIDE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("expected {IMG_SIDE}x{IMG_SIDE} images, found {rows}x{cols}"),
        ));
    }

    let label_count = u32::from_be_bytes([labels_raw[4], labels_raw[5], labels_raw[6], labels_raw[7]]) as usize;
    if label_count != count {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("image/label count mismatch: {count} vs {label_count}"),
        ));
    }

    let pixels = &images_raw[16..];
    let images = (0..count).map(|i| pad_to_grid(&pixels[i * rows * cols..(i + 1) * rows * cols])).collect();
    let labels = labels_raw[8..8 + count].to_vec();

    Ok(MnistSplit { images, labels })
}

fn parse_image_header(raw: &[u8]) -> std::io::Result<(usize, usize, usize)> {
    if raw.len() < 16 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "idx file too short"));
    }
    let count = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
    let rows = u32::from_be_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
    let cols = u32::from_be_bytes([raw[12], raw[13], raw[14], raw[15]]) as usize;
    Ok((count, rows, cols))
}

/// Convert one raw 28x28 `u8` image to `0..1` and centre it in an `N x N` grid.
pub fn pad_to_grid(raw: &[u8]) -> Vec<Real> {
    let pad = (N - IMG_SIDE) / 2;
    let mut out = vec![0.0 as Real; N * N];
    for r in 0..IMG_SIDE {
        for c in 0..IMG_SIDE {
            out[(r + pad) * N + (c + pad)] = raw[r * IMG_SIDE + c] as Real / 255.0;
        }
    }
    out
}

/// Undo [`pad_to_grid`], for writing generated samples back out at 28x28.
pub fn crop_from_grid(grid: &[Real]) -> Vec<Real> {
    let pad = (N - IMG_SIDE) / 2;
    let mut out = vec![0.0 as Real; IMG_SIDE * IMG_SIDE];
    for r in 0..IMG_SIDE {
        for c in 0..IMG_SIDE {
            out[r * IMG_SIDE + c] = grid[(r + pad) * N + (c + pad)];
        }
    }
    out
}

/// Return a readable path for `name`, downloading the gzipped file if needed.
fn ensure_file(name: &str) -> std::io::Result<PathBuf> {
    let dir = Path::new(MNIST_DIR);
    let plain = dir.join(name);
    if plain.exists() {
        return Ok(plain);
    }
    let gz = dir.join(format!("{name}.gz"));
    if gz.exists() {
        return Ok(gz);
    }

    fs::create_dir_all(dir)?;
    download(name, &gz)?;
    Ok(gz)
}

fn download(name: &str, dest: &Path) -> std::io::Result<()> {
    let mut last_err = None;
    for base in MIRRORS {
        let url = format!("{base}/{name}.gz");
        println!("[spectral] downloading {url}");
        match reqwest::blocking::get(&url).and_then(|r| r.error_for_status()).and_then(|r| r.bytes()) {
            Ok(bytes) => {
                fs::write(dest, &bytes)?;
                return Ok(());
            }
            Err(e) => last_err = Some(e.to_string()),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!(
            "could not download {name}: {}. Place the file in {MNIST_DIR}/ manually.",
            last_err.unwrap_or_default()
        ),
    ))
}

/// Read an idx file, transparently gunzipping when the path ends in `.gz`.
fn read_idx(path: &Path) -> std::io::Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if path.extension().and_then(|e| e.to_str()) == Some("gz") {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(&bytes[..]).read_to_end(&mut out)?;
        Ok(out)
    } else {
        Ok(bytes)
    }
}
