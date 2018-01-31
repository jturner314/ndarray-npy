use {Element, NpyExt};
use ndarray::Data;
use ndarray::prelude::*;
use std::error;
use std::fmt;
use std::io::{self, Seek, Write};
use zip::{CompressionMethod, ZipWriter};
use zip::result::ZipError;
use zip::write::FileOptions;

/// An error reading/writing a `.npz` file.
#[derive(Debug)]
pub enum NpzError {
    /// An error caused by I/O.
    Io(io::Error),
    /// An error caused by the zip file.
    Zip(ZipError),
}

impl fmt::Display for NpzError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            NpzError::Io(ref err) => write!(f, "I/O error: {}", err),
            NpzError::Zip(ref err) => write!(f, "Zip file error: {}", err),
        }
    }
}

impl error::Error for NpzError {
    fn description(&self) -> &str {
        match *self {
            NpzError::Io(ref err) => err.description(),
            NpzError::Zip(ref err) => err.description(),
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            NpzError::Io(ref err) => Some(err),
            NpzError::Zip(ref err) => Some(err),
        }
    }
}

impl From<io::Error> for NpzError {
    fn from(err: io::Error) -> NpzError {
        NpzError::Io(err)
    }
}

impl From<ZipError> for NpzError {
    fn from(err: ZipError) -> NpzError {
        NpzError::Zip(err)
    }
}

/// Writer for `.npz` files.
pub struct NpzWriter<W: Write + Seek> {
    zip: ZipWriter<W>,
    options: FileOptions,
}

impl<W: Write + Seek> NpzWriter<W> {
    /// Create a new `.npz` file without compression. See [`numpy.savez`].
    ///
    /// [`numpy.savez`]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
    pub fn new(writer: W) -> NpzWriter<W> {
        NpzWriter {
            zip: ZipWriter::new(writer),
            options: FileOptions::default().compression_method(CompressionMethod::Stored),
        }
    }

    /// Creates a new `.npz` file with compression. See [`numpy.savez_compressed`].
    ///
    /// [`numpy.savez_compressed`]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez_compressed.html
    #[cfg(feature = "compressed_npz")]
    pub fn new_compressed(writer: W) -> NpzWriter<W> {
        NpzWriter {
            zip: ZipWriter::new(writer),
            options: FileOptions::default().compression_method(CompressionMethod::Deflated),
        }
    }

    /// Adds an array with the specified `name` to the `.npz` file.
    pub fn add_array<N, S, D>(&mut self, name: N, array: ArrayBase<S, D>) -> Result<(), NpzError>
    where
        N: Into<String>,
        S::Elem: Element,
        S: Data,
        D: Dimension,
    {
        self.zip.start_file(name, self.options)?;
        array.write_npy(&mut self.zip)?;
        Ok(())
    }
}
