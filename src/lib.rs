//! This crate provides an extension method for [`ndarray`]'s `ArrayBase` type to
//! write [`.npy`] and [`.npz`] files..
//!
//! [`ndarray`]: https://github.com/bluss/rust-ndarray
//! [`.npy`]: https://docs.scipy.org/doc/numpy/neps/npy-format.html
//! [`.npz`]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
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
#[cfg(feature = "npz")]
extern crate zip;

mod npy;
#[cfg(feature = "npz")]
mod npz;

pub use npy::{Element, NpyExt};
#[cfg(feature = "npz")]
pub use npz::{NpzError, NpzWriter};
