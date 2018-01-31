//! This crate provides an extension method for [`ndarray`]'s `ArrayBase` type to
//! write an array in [`.npy` format].
//!
//! [`ndarray`]: https://github.com/bluss/rust-ndarray
//! [`.npy` format]: https://docs.scipy.org/doc/numpy/neps/npy-format.html
//!
//! # Example
//!
//! ```ignore
//! #[macro_use]
//! extern crate ndarray;
//! extern crate ndarray_npy;
//!
//! use ndarray_npy::NpyExt;
//! use std::fs::File;
//!
//! # fn example() -> std::io::Result<()> {
//! let arr = array![[1, 2, 3], [4, 5, 6]];
//! let mut file = File::create("array.npy")?;
//! arr.write_npy(&mut file)?;
//! # Ok(())
//! # }
//! # fn main () {}
//! ```

extern crate byteorder;
extern crate ndarray;

mod npy;

pub use npy::{Element, NpyExt};
