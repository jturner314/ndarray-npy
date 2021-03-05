use crate::{
    ReadNpyError, ReadNpyExt, ReadableElement, WritableElement, WriteNpyError, WriteNpyExt,
};
use ndarray::prelude::*;
use ndarray::{Data, DataOwned};
use std::error::Error;
use std::fmt;
use std::io::{Read, Seek, Write};
use zip::result::ZipError;
use zip::write::FileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

/// An error writing a `.npz` file.
#[derive(Debug)]
pub enum WriteNpzError {
    /// An error caused by the zip file.
    Zip(ZipError),
    /// An error caused by writing an inner `.npy` file.
    Npy(WriteNpyError),
}

impl Error for WriteNpzError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            WriteNpzError::Zip(err) => Some(err),
            WriteNpzError::Npy(err) => Some(err),
        }
    }
}

impl fmt::Display for WriteNpzError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            WriteNpzError::Zip(err) => write!(f, "zip file error: {}", err),
            WriteNpzError::Npy(err) => write!(f, "error writing npy file to npz archive: {}", err),
        }
    }
}

impl From<ZipError> for WriteNpzError {
    fn from(err: ZipError) -> WriteNpzError {
        WriteNpzError::Zip(err)
    }
}

impl From<WriteNpyError> for WriteNpzError {
    fn from(err: WriteNpyError) -> WriteNpzError {
        WriteNpzError::Npy(err)
    }
}

/// Writer for `.npz` files.
///
/// # Example
///
/// ```no_run
/// use ndarray::{array, Array1, Array2};
/// use ndarray_npy::NpzWriter;
/// use std::fs::File;
///
/// let mut npz = NpzWriter::new(File::create("arrays.npz")?);
/// let a: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];
/// let b: Array1<i32> = array![7, 8, 9];
/// npz.add_array("a", &a)?;
/// npz.add_array("b", &b)?;
/// npz.finish()?;
/// # Ok::<_, Box<dyn std::error::Error>>(())
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

    /// Calls [`.finish()`](ZipWriter::finish) on the zip file and
    /// [`.flush()`](Write::flush) on the writer, and then returns the writer.
    ///
    /// This finishes writing the remaining zip structures and flushes the
    /// writer. While dropping will automatically attempt to finish the zip
    /// file and (for writers that flush on drop, such as
    /// [`BufWriter`](std::io::BufWriter)) flush the writer, any errors that
    /// occur during drop will be silently ignored. So, it's necessary to call
    /// `.finish()` to properly handle errors.
    pub fn finish(mut self) -> Result<W, WriteNpzError> {
        let mut writer = self.zip.finish()?;
        writer.flush().map_err(ZipError::from)?;
        Ok(writer)
    }
}

/// An error reading a `.npz` file.
#[derive(Debug)]
pub enum ReadNpzError {
    /// An error caused by the zip archive.
    Zip(ZipError),
    /// An error caused by reading an inner `.npy` file.
    Npy(ReadNpyError),
}

impl Error for ReadNpzError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ReadNpzError::Zip(err) => Some(err),
            ReadNpzError::Npy(err) => Some(err),
        }
    }
}

impl fmt::Display for ReadNpzError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ReadNpzError::Zip(err) => write!(f, "zip file error: {}", err),
            ReadNpzError::Npy(err) => write!(f, "error reading npy file in npz archive: {}", err),
        }
    }
}

impl From<ZipError> for ReadNpzError {
    fn from(err: ZipError) -> ReadNpzError {
        ReadNpzError::Zip(err)
    }
}

impl From<ReadNpyError> for ReadNpzError {
    fn from(err: ReadNpyError) -> ReadNpzError {
        ReadNpzError::Npy(err)
    }
}

/// Reader for `.npz` files.
///
/// # Example
///
/// ```no_run
/// use ndarray::{Array1, Array2};
/// use ndarray_npy::NpzReader;
/// use std::fs::File;
///
/// let mut npz = NpzReader::new(File::open("arrays.npz")?)?;
/// let a: Array2<i32> = npz.by_name("a")?;
/// let b: Array1<i32> = npz.by_name("b")?;
/// # Ok::<_, Box<dyn std::error::Error>>(())
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

    /// Returns `true` iff the `.npz` file doesn't contain any arrays.
    pub fn is_empty(&self) -> bool {
        self.zip.len() == 0
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
