//! This crate provides methods to read/write [`ndarray`]'s `ArrayBase` type
//! from/to [`.npy`] and [`.npz`] files.
//!
//! [`ndarray`]: https://github.com/bluss/rust-ndarray
//! [`.npy`]: https://docs.scipy.org/doc/numpy/neps/npy-format.html
//! [`.npz`]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
//!
//! See [`ReadNpyExt`] and [`WriteNpyExt`] for reading/writing `.npy` files.
//!
//! [`ReadNpyExt`]: trait.ReadNpyExt.html
//! [`WriteNpyExt`]: trait.WriteNpyExt.html
//!
//! See [`NpzReader`] and [`NpzWriter`] for reading/writing `.npz` files.
//!
//! [`NpzReader`]: struct.NpzReader.html
//! [`NpzWriter`]: struct.NpzWriter.html

extern crate byteorder;
extern crate ndarray;
extern crate pest;
#[macro_use]
extern crate pest_derive;
#[macro_use]
extern crate quick_error;
#[cfg(feature = "npz")]
extern crate zip;

mod npy;
#[cfg(feature = "npz")]
mod npz;

pub use npy::{ReadNpyError, ReadNpyExt, ReadableElement, WritableElement, WriteNpyExt};
#[cfg(feature = "npz")]
pub use npz::{NpzReader, NpzWriter, ReadNpzError, WriteNpzError};
