use crate::{
    ReadNpyError, ReadNpyExt, ReadableElement, WritableElement, WriteNpyError, WriteNpyExt
};
use ndarray::{Data, DataOwned};
use ndarray::prelude::*;
use std::error::Error;
use std::io::{self, Read, Seek, Write};
use zip::{CompressionMethod, ZipArchive, ZipWriter};
use zip::result::ZipError;
use zip::write::FileOptions;

quick_error! {
    /// An error writing a `.npz` file.
    #[derive(Debug)]
    pub enum WriteNpzError {
        /// An error caused by I/O.
        Io(err: io::Error) {
            description("I/O error")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
        /// An error caused by the zip file.
        Zip(err: ZipError) {
            description("zip file error")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
        /// An error caused by writing an inner `.npy` file.
        Npy(err: WriteNpyError) {
            description("error writing npy file to npz archive")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
    }
}

/// Writer for `.npz` files.
///
/// # Example
///
/// ```
/// #[macro_use]
/// extern crate ndarray;
/// extern crate ndarray_npy;
///
/// use ndarray::prelude::*;
/// use ndarray_npy::NpzWriter;
/// use std::fs::File;
/// # use ndarray_npy::WriteNpzError;
///
/// # fn write_example() -> Result<(), WriteNpzError> {
/// let mut npz = NpzWriter::new(File::create("arrays.npz")?);
/// let a: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];
/// let b: Array1<i32> = array![7, 8, 9];
/// npz.add_array("a", &a)?;
/// npz.add_array("b", &b)?;
/// # Ok(())
/// # }
/// # fn main () {}
/// ```
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
    pub fn add_array<N, S, D>(
        &mut self,
        name: N,
        array: &ArrayBase<S, D>,
    ) -> Result<(), WriteNpzError>
    where
        N: Into<String>,
        S::Elem: WritableElement,
        S: Data,
        D: Dimension,
    {
        self.zip.start_file(name, self.options)?;
        array.write_npy(&mut self.zip)?;
        Ok(())
    }
}

quick_error! {
    /// An error reading a `.npz` file.
    #[derive(Debug)]
    pub enum ReadNpzError {
        /// An error caused by I/O.
        Io(err: io::Error) {
            description("I/O error")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
        /// An error caused by the zip archive.
        Zip(err: ZipError) {
            description("zip file error")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
        /// An error caused by reading an inner `.npy` file.
        Npy(err: ReadNpyError) {
            description("error reading npy file in npz archive")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
    }
}

/// Reader for `.npz` files.
///
/// # Example
///
/// ```
/// #[macro_use]
/// extern crate ndarray;
/// extern crate ndarray_npy;
///
/// use ndarray::prelude::*;
/// use ndarray_npy::NpzReader;
/// use std::fs::File;
/// # use ndarray_npy::ReadNpzError;
///
/// # fn read_example() -> Result<(), ReadNpzError> {
/// let mut npz = NpzReader::new(File::open("arrays.npz")?)?;
/// let a: Array2<i32> = npz.by_name("a")?;
/// let b: Array1<i32> = npz.by_name("b")?;
/// # Ok(())
/// # }
/// # fn main() {}
/// ```
pub struct NpzReader<R: Read + Seek> {
    zip: ZipArchive<R>,
}

impl<R: Read + Seek> NpzReader<R> {
    /// Creates a new `.npz` file reader.
    pub fn new(reader: R) -> Result<NpzReader<R>, ReadNpzError> {
        Ok(NpzReader {
            zip: ZipArchive::new(reader)?,
        })
    }

    /// Returns the number of arrays in the `.npz` file.
    pub fn len(&self) -> usize {
        self.zip.len()
    }

    /// Returns the names of all of the arrays in the file.
    pub fn names(&mut self) -> Result<Vec<String>, ReadNpzError> {
        Ok((0..self.zip.len())
            .map(|i| Ok(self.zip.by_index(i)?.name().to_owned()))
            .collect::<Result<_, ZipError>>()?)
    }

    /// Reads an array by name.
    pub fn by_name<S, D>(&mut self, name: &str) -> Result<ArrayBase<S, D>, ReadNpzError>
    where
        S::Elem: ReadableElement,
        S: DataOwned,
        D: Dimension,
    {
        Ok(ArrayBase::<S, D>::read_npy(self.zip.by_name(name)?)?)
    }

    /// Reads an array by index in the `.npz` file.
    pub fn by_index<S, D>(&mut self, index: usize) -> Result<ArrayBase<S, D>, ReadNpzError>
    where
        S::Elem: ReadableElement,
        S: DataOwned,
        D: Dimension,
    {
        Ok(ArrayBase::<S, D>::read_npy(self.zip.by_index(index)?)?)
    }
}
