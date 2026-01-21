use core::fmt;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// A shareable 2D `f32` matrix with `Arc<RwLock<..>>` storage.
///
/// This is primarily used to tie (share) a single embedding table between multiple layers.
#[derive(Clone, Default)]
pub struct SharedF32Matrix {
    inner: Arc<RwLock<Vec<Vec<f32>>>>,
}

impl SharedF32Matrix {
    pub fn new(matrix: Vec<Vec<f32>>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(matrix)),
        }
    }

    pub fn read(&self) -> RwLockReadGuard<'_, Vec<Vec<f32>>> {
        self.inner.read().expect("SharedF32Matrix poisoned")
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, Vec<Vec<f32>>> {
        self.inner.write().expect("SharedF32Matrix poisoned")
    }

    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }

    pub fn dims(&self) -> (usize, usize) {
        let m = self.read();
        let rows = m.len();
        let cols = m.first().map(|r| r.len()).unwrap_or(0);
        (rows, cols)
    }

    pub fn to_vec(&self) -> Vec<Vec<f32>> {
        self.read().clone()
    }
}

impl fmt::Debug for SharedF32Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (r, c) = self.dims();
        f.debug_struct("SharedF32Matrix")
            .field("rows", &r)
            .field("cols", &c)
            .finish()
    }
}

impl Serialize for SharedF32Matrix {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.read().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SharedF32Matrix {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let matrix = Vec::<Vec<f32>>::deserialize(deserializer)?;
        Ok(Self::new(matrix))
    }
}
