//! The RM transformer rebuilt with candle tensors: batched, differentiated by candle's autograd and runnable on the
//! GPU (feature `candle-cuda`). Complex values are pairs of real tensors (`complex::CTensor`).

pub mod block;
pub mod complex;

#[cfg(test)]
mod tests;
