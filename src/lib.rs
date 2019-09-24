//! This crate provides methods to read/write [`ndarray`]'s `ArrayBase` type
//! from/to [`.npy`] and [`.npz`] files.
//!
//! [`ndarray`]: https://github.com/bluss/ndarray
//! [`.npy`]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html
//! [`.npz`]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
//!
//! See the [`ReadNpyExt`] and [`WriteNpyExt`] extension traits or the
//! [`read_npy`] and [`write_npy`] convenience functions for reading/writing
//! `.npy` files.
//!
//! [`ReadNpyExt`]: trait.ReadNpyExt.html
//! [`WriteNpyExt`]: trait.WriteNpyExt.html
//! [`read_npy`]: fn.read_npy.html
//! [`write_npy`]: fn.write_npy.html
//!
//! See [`NpzReader`] and [`NpzWriter`] for reading/writing `.npz` files.
//!
//! [`NpzReader`]: struct.NpzReader.html
//! [`NpzWriter`]: struct.NpzWriter.html
//!
//! See the [repository] for information about the default features and how to
//! use this crate with Cargo.
//!
//! [repository]: https://github.com/jturner314/ndarray-npy
//!
//! # Limitations
//!
//! * Parsing of `.npy` files is currently limited to files where the `descr`
//!   field of the [header dictionary] is a Python string literal of the form
//!   `'string'`, `"string"`, `'''string'''`, or `"""string"""`.
//!
//! * `WritableElement` and `ReadableElement` are currently implemented only
//!   for fixed-size integers, floating point numbers, and `bool`.
//!
//! The plan is to add support for more element types (including custom
//! user-defined structs) in the future.
//!
//! [header dictionary]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html#format-version-1-0

mod npy;
#[cfg(feature = "npz")]
mod npz;

pub use crate::npy::{
    read_npy, write_npy, ReadDataError, ReadNpyError, ReadNpyExt, ReadableElement, WritableElement,
    WriteDataError, WriteNpyError, WriteNpyExt,
};
#[cfg(feature = "npz")]
pub use crate::npz::{NpzReader, NpzWriter, ReadNpzError, WriteNpzError};
