//! Training images. Blueprint section A1.
//!
//! Two sources ([`DataSource`]):
//! * MNIST, read from `datasets/mnist/`. Both the raw `idx-ubyte` files and
//!   their gzipped form are accepted; if neither is present the loader downloads
//!   them once from a public mirror. The 28x28 digits are centred in the grid
//!   (28 -> 32 zero padding at the original size, scaled up for larger grids).
//! * A folder of images (PNG, JPEG, BMP, WebP, ...): each image is centre-cropped
//!   to a square and resized to the model's size.
//!
//! Images are stored as `channels * size * size` floats in `0..1`, channel-planar.

use crate::neural_networks::spectral_model::bands::N;
use crate::neural_networks::utils::dtype::Real;
use image::imageops::FilterType;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
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

// ---------------------------------------------------------------------------
// Datasets at any size and colour depth
// ---------------------------------------------------------------------------

/// Where the training images come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "path", rename_all = "lowercase")]
pub enum DataSource {
    Mnist,
    /// A folder with image files (searched recursively).
    Folder(String),
}

impl DataSource {
    pub fn describe(&self) -> String {
        match self {
            DataSource::Mnist => "MNIST".to_string(),
            DataSource::Folder(path) => format!("folder {path}"),
        }
    }
}

/// Training images at the model's geometry.
#[derive(Debug, Clone)]
pub struct ImageSet {
    pub size: usize,
    pub channels: usize,
    /// Side of the centred region that holds the image content (MNIST: the 28/32 digit area); the rest is padding.
    pub content_side: usize,
    /// `channels * size * size` floats in `0..1`, channel-planar.
    pub images: Vec<Vec<Real>>,
}

/// File extensions read from an image folder.
const IMAGE_EXTENSIONS: [&str; 7] = ["png", "jpg", "jpeg", "bmp", "webp", "gif", "tif"];

/// Loads the images of `source` at `size x size` with `channels` (1 or 3) colour channels.
/// `max_images` limits how many are loaded (memory: `4 * channels * size^2` bytes per image).
pub fn load_images(source: &DataSource, size: usize, channels: usize, max_images: Option<usize>) -> std::io::Result<ImageSet> {
    assert!(channels == 1 || channels == 3, "channels must be 1 (grayscale) or 3 (RGB)");
    match source {
        DataSource::Mnist => {
            let data = load_train()?;
            let content_side = size * IMG_SIDE / N;
            let count = max_images.unwrap_or(usize::MAX).min(data.len());
            let images = data.images[..count]
                .par_iter()
                .map(|grid| {
                    let digit = crop_from_grid(grid);
                    let digit = if content_side == IMG_SIDE { digit } else { resize_gray(&digit, IMG_SIDE, content_side) };
                    let mut out = vec![0.0 as Real; channels * size * size];
                    let pad = (size - content_side) / 2;
                    for ch in 0..channels {
                        for r in 0..content_side {
                            for c in 0..content_side {
                                out[ch * size * size + (r + pad) * size + (c + pad)] = digit[r * content_side + c];
                            }
                        }
                    }
                    out
                })
                .collect();
            Ok(ImageSet { size, channels, content_side, images })
        }
        DataSource::Folder(path) => {
            let mut files = Vec::new();
            collect_image_files(Path::new(path), &mut files)?;
            files.sort();
            if let Some(max) = max_images {
                files.truncate(max);
            }
            if files.is_empty() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("no images ({}) found in {path:?}", IMAGE_EXTENSIONS.join(", ")),
                ));
            }
            let images: Vec<Vec<Real>> = files
                .par_iter()
                .filter_map(|file| match image::open(file) {
                    Ok(img) => Some(prepare_image(&img, size, channels)),
                    Err(e) => {
                        eprintln!("[spectral] skipping {}: {e}", file.display());
                        None
                    }
                })
                .collect();
            if images.is_empty() {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("none of the images in {path:?} could be read")));
            }
            Ok(ImageSet { size, channels, content_side: size, images })
        }
    }
}

fn collect_image_files(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let entries = fs::read_dir(dir).map_err(|e| std::io::Error::new(e.kind(), format!("cannot read image folder {}: {e}", dir.display())))?;
    for entry in entries {
        let path = entry?.path();
        if path.is_dir() {
            collect_image_files(&path, files)?;
        } else if path.extension().and_then(|e| e.to_str()).is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str())) {
            files.push(path);
        }
    }
    Ok(())
}

/// Centre-crops `img` to a square, resizes it to `size x size` and converts it to `channels` planes in `0..1`.
pub fn prepare_image(img: &image::DynamicImage, size: usize, channels: usize) -> Vec<Real> {
    let side = img.width().min(img.height());
    let (x, y) = ((img.width() - side) / 2, (img.height() - side) / 2);
    let square = img.crop_imm(x, y, side, side).resize_exact(size as u32, size as u32, FilterType::Lanczos3);
    let pixels = size * size;
    let mut out = vec![0.0 as Real; channels * pixels];
    if channels == 1 {
        for (i, p) in square.to_luma8().pixels().enumerate() {
            out[i] = p.0[0] as Real / 255.0;
        }
    } else {
        for (i, p) in square.to_rgb8().pixels().enumerate() {
            for ch in 0..3 {
                out[ch * pixels + i] = p.0[ch] as Real / 255.0;
            }
        }
    }
    out
}

fn resize_gray(img: &[Real], side: usize, new_side: usize) -> Vec<Real> {
    let buffer = image::GrayImage::from_fn(side as u32, side as u32, |x, y| image::Luma([(img[y as usize * side + x as usize].clamp(0.0, 1.0) * 255.0).round() as u8]));
    let resized = image::imageops::resize(&buffer, new_side as u32, new_side as u32, FilterType::Lanczos3);
    resized.pixels().map(|p| p.0[0] as Real / 255.0).collect()
}

/// Cuts the centred `content_side` region out of every channel of a `size x size` image.
pub fn crop_content(img: &[Real], size: usize, channels: usize, content_side: usize) -> Vec<Real> {
    let pad = (size - content_side) / 2;
    let mut out = vec![0.0 as Real; channels * content_side * content_side];
    for ch in 0..channels {
        for r in 0..content_side {
            for c in 0..content_side {
                out[(ch * content_side + r) * content_side + c] = img[ch * size * size + (r + pad) * size + (c + pad)];
            }
        }
    }
    out
}
