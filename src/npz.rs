use crate::{
    ReadNpyError, ReadNpyExt, ReadableElement,
    WritableElement, WriteNpyError, WriteNpyExt,
    ViewNpyError, ViewNpyExt, ViewNpyMutExt, CastableElement,
};
use ndarray::prelude::*;
use ndarray::{Data, DataOwned};
use std::error::Error;
use std::convert::TryInto;
use std::fmt;
use std::io::{self, Cursor, Read, Seek, Write};
use std::collections::{HashMap, BTreeMap};
use zip::result::ZipError;
use zip::write::FileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};
use crc32fast::Hasher;

/// An error writing a `.npz` file.
#[derive(Debug)]
#[non_exhaustive]
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
}

/// An error reading a `.npz` file.
#[derive(Debug)]
#[non_exhaustive]
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

/// An error viewing a `.npz` file.
#[derive(Debug)]
#[non_exhaustive]
pub enum ViewNpzError {
    /// An error caused by the zip archive.
    Zip(ZipError),
    /// An error caused by viewing an inner `.npy` file.
    Npy(ViewNpyError),
    /// Unaligned `.npy` file within zip archive.
    UnalignedBytes,
    /// A read-write `.npy` file view has already been moved out of its `.npz` file view.
    MovedNpyViewMut,
}

impl Error for ViewNpzError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ViewNpzError::Zip(err) => Some(err),
            ViewNpzError::Npy(err) => Some(err),
            ViewNpzError::UnalignedBytes => None,
            ViewNpzError::MovedNpyViewMut => None,
        }
    }
}

impl fmt::Display for ViewNpzError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ViewNpzError::Zip(err) => write!(f, "zip file error: {}", err),
            ViewNpzError::Npy(err) => write!(f, "error viewing npy file in npz archive: {}", err),
            ViewNpzError::UnalignedBytes => {
                write!(f, "cannot cast unaligned array data, try `zipalign 64 in.npz out.npz'")
            },
            ViewNpzError::MovedNpyViewMut => {
                write!(f, "read-write npy file view already moved out of npz file view")
            },
        }
    }
}

impl From<ZipError> for ViewNpzError {
    fn from(err: ZipError) -> ViewNpzError {
        ViewNpzError::Zip(err)
    }
}

impl From<ViewNpyError> for ViewNpzError {
    fn from(err: ViewNpyError) -> ViewNpzError {
        match err {
            ViewNpyError::UnalignedBytes => ViewNpzError::UnalignedBytes,
            _ => ViewNpzError::Npy(err),
        }
    }
}

/// Read-only view for memory-mapped `.npz` files.
///
/// # Example
///
/// ```no_run
/// use ndarray::Ix1;
/// use ndarray_npy::{NpzView, ViewNpzError};
/// use std::fs::OpenOptions;
///
/// let mmap = &[]; // Memory-mapped `.npz` file.
/// let npz = NpzView::new(mmap)?;
/// for npy in npz.names() {
///     println!("{}", npy);
/// }
/// let x_npy_view = npz.by_name("x.npy")?;
/// let y_npy_view = npz.by_name("y.npy")?;
/// let x_array_view = x_npy_view.view::<f64, Ix1>()?;
/// let y_array_view = y_npy_view.view::<f64, Ix1>()?;
/// println!("{}", x_array_view);
/// println!("{}", y_array_view);
/// # Ok::<(), ViewNpzError>(())
/// ```
#[derive(Debug, Clone)]
pub struct NpzView<'a> {
    files: HashMap<usize, NpyView<'a>>,
    names: HashMap<String, usize>,
}

impl<'a> NpzView<'a> {
    /// Creates a new read-only view of a memory-mapped `.npz` file.
    pub fn new(bytes: &'a [u8]) -> Result<Self, ViewNpzError> {
        let mut zip = ZipArchive::new(Cursor::new(bytes))?;
        let mut files = HashMap::new();
        let mut names = HashMap::new();
        let mut index = 0;
        for zip_index in 0..zip.len() {
            let file = zip.by_index(zip_index)?;
            // Skip compressed files.
            if file.compression() != CompressionMethod::Stored {
                continue;
            }
            // Store file index by file names.
            let name = file.name().to_string();
            names.insert(name, index);
            // Get data slice.
            let data = file.data_start().try_into().ok()
                .and_then(|data_start: usize| file.size().try_into().ok()
                    .and_then(|size: usize| data_start.checked_add(size))
                    .and_then(|data_end| bytes.get(data_start..data_end)));
            // Get central CRC-32 slice.
            let central_crc32 = file.central_header_start().try_into().ok()
                .and_then(|central_header_start: usize| central_header_start.checked_add(16))
                .and_then(|central_crc32_start| central_crc32_start.checked_add(4)
                    .and_then(|central_crc32_end|
                        bytes.get(central_crc32_start..central_crc32_end)));
            if let (Some(data), Some(central_crc32)) = (data, central_crc32) {
                // Store file view by file index.
                files.insert(index, NpyView { data, central_crc32 });
                // Increment index of uncompressed files.
                index += 1;
            } else {
                Err(ZipError::InvalidArchive("Length overflow"))?;
            }
        }
        Ok(Self { files, names })
    }

    /// Returns `true` iff the `.npz` file doesn't contain any **uncompressed** arrays.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Returns the number of **uncompressed** arrays in the `.npz` file.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Returns the names of all of the **uncompressed** arrays in the `.npz` file.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.names.keys().map(String::as_str)
    }

    /// Returns a read-only `.npy` file view by name.
    pub fn by_name(&self, name: &str) -> Result<NpyView<'a>, ViewNpzError> {
        self.by_index(self.names.get(name).copied().ok_or(ZipError::FileNotFound)?)
    }

    /// Returns a read-only `.npy` file view by index.
    pub fn by_index(&self, index: usize) -> Result<NpyView<'a>, ViewNpzError> {
        self.files.get(&index).copied().ok_or(ZipError::FileNotFound.into())
    }
}

/// Read-only view of memory-mapped `.npy` files within an `.npz` file.
///
/// **Note:** Does **not** automatically `verify()` CRC-32 checksum.
#[derive(Debug, Clone, Copy)]
pub struct NpyView<'a> {
    data: &'a [u8],
    central_crc32: &'a [u8],
}

impl<'a> NpyView<'a> {
    /// Verifies CRC-32 checksum by reading the whole array.
    pub fn verify(&self) -> Result<(), ViewNpzError> {
        // Like the `zip` crate, verify only against central CRC-32.
        Ok(crc32_verify(self.data, self.central_crc32)?)
    }

    /// Returns a read-only view of a memory-mapped `.npy` file.
    ///
    /// **Note:** Iterates over `bool` array to ensure `0x00`/`0x01` values.
    pub fn view<A, D>(&self) -> Result<ArrayView<A, D>, ViewNpzError>
    where
        A: CastableElement,
        D: Dimension,
    {
        Ok(ArrayView::view_npy(self.data)?)
    }
}

/// Read-write view for memory-mapped `.npz` files.
///
/// # Example
///
/// ```no_run
/// use ndarray::Ix1;
/// use ndarray_npy::{NpzViewMut, ViewNpzError};
/// use std::fs::OpenOptions;
///
/// let mmap = &mut []; // Memory-mapped `.npz` file.
/// let mut npz = NpzViewMut::new(mmap)?;
/// for npy in npz.names() {
///     println!("{}", npy);
/// }
/// let mut x_npy_view_mut = npz.by_name("x.npy")?;
/// let mut y_npy_view_mut = npz.by_name("y.npy")?;
/// let x_array_view_mut = x_npy_view_mut.view_mut::<f64, Ix1>()?;
/// let y_array_view_mut = y_npy_view_mut.view_mut::<f64, Ix1>()?;
/// // Split borrows: Mutable access to both arrays at the same time.
/// println!("{}", x_array_view_mut);
/// println!("{}", y_array_view_mut);
/// // x_npy_view_mut.update(); // Automatically updated on `drop()`.
/// // y_npy_view_mut.update(); // Automatically updated on `drop()`.
/// # Ok::<(), ViewNpzError>(())
/// ```
#[derive(Debug)]
pub struct NpzViewMut<'a> {
    files: HashMap<usize, NpyViewMut<'a>>,
    names: HashMap<String, usize>,
}

impl<'a> NpzViewMut<'a> {
    /// Creates a new read-write view of a memory-mapped `.npz` file.
    pub fn new(mut bytes: &'a mut [u8]) -> Result<Self, ViewNpzError> {
        let mut zip = ZipArchive::new(Cursor::new(&bytes))?;
        let mut names = HashMap::new();
        let mut ranges = HashMap::new();
        let mut splits = BTreeMap::new();
        let mut index = 0;
        for zip_index in 0..zip.len() {
            let file = zip.by_index(zip_index)?;
            // Skip compressed files.
            if file.compression() != CompressionMethod::Stored {
                continue;
            }
            // Store file index by file names.
            let name = file.name().to_string();
            names.insert(name, index);
            // Get local CRC-32 range.
            let crc32 = file.header_start().try_into().ok()
                .and_then(|header_start: usize| header_start.checked_add(14))
                .and_then(|crc32_start| crc32_start.checked_add(4)
                    .map(|crc32_end| crc32_start..crc32_end));
            // Get data range.
            let data = file.data_start().try_into().ok()
                .and_then(|data_start: usize| file.size().try_into().ok()
                    .and_then(|size: usize| data_start.checked_add(size))
                    .map(|data_end| data_start..data_end));
            // Get central CRC-32 range.
            let central_crc32 = file.central_header_start().try_into().ok()
                .and_then(|central_header_start: usize| central_header_start.checked_add(16))
                .and_then(|central_crc32_start| central_crc32_start.checked_add(4)
                    .map(|central_crc32_end| central_crc32_start..central_crc32_end));
            if let (Some(crc32), Some(data), Some(central_crc32)) = (crc32, data, central_crc32) {
                // Sort ranges by their starts.
                splits.insert(crc32.start, crc32.end);
                splits.insert(data.start, data.end);
                splits.insert(central_crc32.start, central_crc32.end);
                // Store ranges by file index.
                ranges.insert(index, (crc32, data, central_crc32));
                // Increment index of uncompressed files.
                index += 1;
            } else {
                Err(ZipError::InvalidArchive("Length overflow"))?;
            }
        }
        // Split and store borrows by their range starts.
        let mut offset = 0;
        let mut slices = HashMap::new();
        for (&start, &end) in &splits {
            // Split off leading bytes.
            let mid = start - offset;
            if mid > bytes.len() {
                Err(ZipError::InvalidArchive("Offset exceeds EOF"))?;
            }
            let (slice, remaining_bytes) = bytes.split_at_mut(mid);
            offset += slice.len();
            // Split off leading borrow of interest.
            let mid = end - offset;
            if mid > remaining_bytes.len() {
                Err(ZipError::InvalidArchive("Length exceeds EOF"))?;
            }
            let (slice, remaining_bytes) = remaining_bytes.split_at_mut(mid);
            offset += slice.len();
            // Store borrow by its range start.
            slices.insert(start, slice);
            // Store remaining bytes.
            bytes = remaining_bytes;
        }
        // Collect split borrows as file views.
        let mut files = HashMap::new();
        for (&index, (crc32, data, central_crc32)) in &ranges {
            let crc32 = slices.remove(&crc32.start);
            let data = slices.remove(&data.start);
            let central_crc32 = slices.remove(&central_crc32.start);
            if let (Some(crc32), Some(data), Some(central_crc32)) = (crc32, data, central_crc32) {
                files.insert(index, NpyViewMut { crc32, data, central_crc32 });
            } else {
                Err(ZipError::InvalidArchive("Ambiguous offsets"))?;
            }
        }
        Ok(Self { files, names })
    }

    /// Returns `true` iff the `.npz` file doesn't contain any **uncompressed** arrays.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Returns the number of **uncompressed** arrays in the `.npz` file.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Returns the names of all of the **uncompressed** arrays in the file.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.names.keys().map(String::as_str)
    }

    /// Moves a read-write `.npy` file view by name out of the `.npz` file view.
    pub fn by_name(&mut self, name: &str) -> Result<NpyViewMut<'a>, ViewNpzError> {
        self.by_index(self.names.get(name).copied().ok_or(ZipError::FileNotFound)?)
    }

    /// Moves a read-write `.npy` file view by index out of the `.npz` file view.
    pub fn by_index(&mut self, index: usize) -> Result<NpyViewMut<'a>, ViewNpzError> {
        if index > self.names.len() {
            Err(ZipError::FileNotFound.into())
        } else {
            self.files.remove(&index).ok_or(ViewNpzError::MovedNpyViewMut)
        }
    }
}

/// Read-write view of memory-mapped `.npy` files within an `.npz` file.
///
/// **Note:** Does **not** automatically `verify()` CRC-32 checksum.
/// **Note:** Does automatically `update()` CRC-32 checksum on `drop()`.
#[derive(Debug)]
pub struct NpyViewMut<'a> {
    crc32: &'a mut [u8],
    data: &'a mut [u8],
    central_crc32: &'a mut [u8],
}

impl<'a> NpyViewMut<'a> {
    /// Verifies CRC-32 checksum by reading the whole array.
    pub fn verify(&self) -> Result<(), ViewNpzError> {
        // Like the `zip` crate, verify only against central CRC-32.
        Ok(crc32_verify(self.data, self.central_crc32)?)
    }

    /// Returns a read-only view of a memory-mapped `.npy` file.
    ///
    /// **Note:** Iterates over `bool` array to ensure `0x00`/`0x01` values.
    pub fn view<A, D>(&self) -> Result<ArrayView<A, D>, ViewNpzError>
    where
        A: CastableElement,
        D: Dimension,
    {
        Ok(ArrayView::<A, D>::view_npy(self.data)?)
    }

    /// Returns a read-write view of a memory-mapped `.npy` file.
    ///
    /// **Note:** Iterates over `bool` array to ensure `0x00`/`0x01` values.
    pub fn view_mut<A, D>(&mut self) -> Result<ArrayViewMut<A, D>, ViewNpzError>
    where
        A: CastableElement,
        D: Dimension,
    {
        Ok(ArrayViewMut::<A, D>::view_npy_mut(self.data)?)
    }

    /// Updates CRC-32 checksum by reading the whole array.
    ///
    /// **Note:** Automatically updated on `drop()`.
    pub fn update(&mut self) {
        self.central_crc32.copy_from_slice(&crc32_update(&self.data));
        self.crc32.copy_from_slice(self.central_crc32);
    }
}

impl<'a> Drop for NpyViewMut<'a> {
    fn drop(&mut self) {
        self.update();
    }
}

fn crc32_verify(bytes: &[u8], crc32: &[u8]) -> Result<(), ZipError> {
    if crc32_update(bytes) == crc32 {
        Ok(())
    } else {
        Err(ZipError::Io(io::Error::new(io::ErrorKind::Other, "Invalid checksum")))
    }
}

fn crc32_update(bytes: &[u8]) -> [u8; 4] {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    hasher.finalize().to_le_bytes()
}
