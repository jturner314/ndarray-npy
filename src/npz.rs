use crate::{
    ReadNpyError, ReadNpyExt, ReadableElement, ViewElement, ViewMutElement, ViewMutNpyExt,
    ViewNpyError, ViewNpyExt, WritableElement, WriteNpyError, WriteNpyExt,
};
use ndarray::prelude::*;
use ndarray::{Data, DataOwned};
use std::collections::{BTreeMap, HashMap};
use std::convert::TryInto;
use std::error::Error;
use std::fmt;
use std::io::{self, BufWriter, Cursor, Read, Seek, Write};
use std::ops::Range;
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
/// Note that the inner [`ZipWriter`] is wrapped in a [`BufWriter`] when
/// writing each array with [`.add_array()`](NpzWriter::add_array). If desired,
/// you could additionally buffer the innermost writer (e.g. the
/// [`File`](std::fs::File) when writing to a file) by wrapping it in a
/// [`BufWriter`]. This may be somewhat beneficial if the arrays are large and
/// have non-standard layouts but may decrease performance if the arrays have
/// standard or Fortran layout, so it's not recommended without testing to
/// compare.
///
/// # Example
///
/// ```no_run
/// use ndarray::{array, aview0, Array1, Array2};
/// use ndarray_npy::NpzWriter;
/// use std::fs::File;
///
/// let mut npz = NpzWriter::new(File::create("arrays.npz")?);
/// let a: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];
/// let b: Array1<i32> = array![7, 8, 9];
/// npz.add_array("a", &a)?;
/// npz.add_array("b", &b)?;
/// npz.add_array("c", &aview0(&10))?;
/// npz.finish()?;
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub struct NpzWriter<W: Write + Seek> {
    zip: ZipWriter<W>,
    options: FileOptions,
    align: u16,
}

impl<W: Write + Seek> NpzWriter<W> {
    /// Creates a new `.npz` file without compression. See [`numpy.savez`].
    ///
    /// Ensures `.npy` files are 64-byte aligned for memory-mapping via [`NpzView`]/[`NpzViewMut`].
    ///
    /// [`numpy.savez`]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
    pub fn new(writer: W) -> NpzWriter<W> {
        NpzWriter {
            zip: ZipWriter::new(writer),
            options: FileOptions::default().compression_method(CompressionMethod::Stored),
            align: 64,
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
            align: 1,
        }
    }

    /// Adds an array with the specified `name` to the `.npz` file.
    ///
    /// To write a scalar value, create a zero-dimensional array using [`arr0`](ndarray::arr0) or
    /// [`aview0`](ndarray::aview0).
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
        self.zip
            .start_file_aligned(name, self.options, self.align)?;
        // Buffering when writing individual arrays is beneficial even when the
        // underlying writer is `Cursor<Vec<u8>>` instead of a real file. The
        // only exception I saw in testing was the "compressed, in-memory
        // writer, standard layout case". See
        // https://github.com/jturner314/ndarray-npy/issues/50#issuecomment-812802481
        // for details.
        array.write_npy(BufWriter::new(&mut self.zip))?;
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

/// An error viewing a `.npz` file.
#[derive(Debug)]
#[non_exhaustive]
pub enum ViewNpzError {
    /// An error caused by the zip archive.
    Zip(ZipError),
    /// An error caused by viewing an inner `.npy` file.
    Npy(ViewNpyError),
    /// The start of the data is not properly aligned for the element type.
    MisalignedData,
    /// A mutable `.npy` file view has already been moved out of its `.npz` file view.
    MovedNpyViewMut,
}

impl Error for ViewNpzError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ViewNpzError::Zip(err) => Some(err),
            ViewNpzError::Npy(err) => Some(err),
            ViewNpzError::MisalignedData => None,
            ViewNpzError::MovedNpyViewMut => None,
        }
    }
}

impl fmt::Display for ViewNpzError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ViewNpzError::Zip(err) => write!(f, "zip file error: {}", err),
            ViewNpzError::Npy(err) => write!(f, "error viewing npy file in npz archive: {}", err),
            ViewNpzError::MisalignedData => write!(
                f,
                "cannot cast unaligned array data, try 'rezip in.npz -o out.npz'"
            ),
            ViewNpzError::MovedNpyViewMut => write!(
                f,
                "mutable npy file view already moved out of npz file view"
            ),
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
            ViewNpyError::MisalignedData => ViewNpzError::MisalignedData,
            _ => ViewNpzError::Npy(err),
        }
    }
}

/// Immutable view for memory-mapped `.npz` files.
///
/// The primary use-case for this is viewing `.npy` files within a memory-mapped
/// `.npz` archive.
///
/// # Notes
///
/// - For types for which not all bit patterns are valid, such as `bool`, the
///   implementation iterates over all of the elements when creating the view
///   to ensure they have a valid bit pattern.
///
/// - The `.npy` files within the `.npz` archive must be at least 16-byte or
///   ideally 64-byte aligned. Archives not created by this crate can be aligned
///   with the help of the CLI tool [`rezip`] as in `rezip in.npz -o out.npz`.
///
/// [`rezip`]: https://crates.io/crates/rezip
///
/// # Example
///
/// This is an example of opening an immutably memory-mapped `.npz` archive as
/// an [`NpzView`] providing an [`NpyView`] for each uncompressed `.npy` file
/// within the archive which can be accessed via [`NpyView::view`] as
/// immutable [`ArrayView`].
///
/// This example uses the [`memmap2`](https://crates.io/crates/memmap2) crate
/// because that appears to be the best-maintained memory-mapping crate at the
/// moment, but [`Self::new`] takes a `&mut [u8]` instead of a file so that you
/// can use the memory-mapping crate you're most comfortable with.
///
/// ```
/// use std::fs::OpenOptions;
/// use memmap2::MmapOptions;
/// use ndarray_npy::{NpzView, ViewNpzError};
/// use ndarray::Ix1;
///
/// // Open `.npz` archive of uncompressed `.npy` files in native endian.
/// #[cfg(target_endian = "little")]
/// let file = OpenOptions::new().read(true)
///     .open("tests/examples_little_endian_64_byte_aligned.npz").unwrap();
/// #[cfg(target_endian = "big")]
/// let file = OpenOptions::new().read(true)
///     .open("tests/examples_big_endian_64_byte_aligned.npz").unwrap();
/// // Memory-map `.npz` archive of 64-byte aligned `.npy` files.
/// let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
/// let npz = NpzView::new(&mmap)?;
/// // List uncompressed files only.
/// for npy in npz.names() {
///     println!("{}", npy);
/// }
/// // Get immutable `.npy` views.
/// let x_npy_view = npz.by_name("i64.npy")?;
/// let y_npy_view = npz.by_name("f64.npy")?;
/// // Optionally verify CRC-32 checksums.
/// x_npy_view.verify()?;
/// y_npy_view.verify()?;
/// // Get and print immutable `ArrayView`s.
/// let x_array_view = x_npy_view.view::<i64, Ix1>()?;
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
    /// Creates a new immutable view of a memory-mapped `.npz` file.
    pub fn new(bytes: &'a [u8]) -> Result<Self, ViewNpzError> {
        let mut zip = ZipArchive::new(Cursor::new(bytes))?;
        let mut files = HashMap::new();
        let mut names = HashMap::new();
        let mut index = 0;
        for zip_index in 0..zip.len() {
            let file = zip.by_index(zip_index)?;
            // Skip directories and compressed files.
            if file.is_dir() || file.compression() != CompressionMethod::Stored {
                continue;
            }
            // Store file index by file names.
            let name = file.name().to_string();
            names.insert(name, index);
            // Get data slice.
            let data: Option<&[u8]> =
                file.data_start()
                    .try_into()
                    .ok()
                    .and_then(|data_start: usize| {
                        file.size()
                            .try_into()
                            .ok()
                            .and_then(|size: usize| data_start.checked_add(size))
                            .and_then(|data_end| bytes.get(data_start..data_end))
                    });
            // Get central CRC-32 slice.
            let central_crc32: Option<&[u8]> = file
                .central_header_start()
                .try_into()
                .ok()
                .and_then(|central_header_start: usize| central_header_start.checked_add(16))
                .and_then(|central_crc32_start| {
                    central_crc32_start
                        .checked_add(4)
                        .and_then(|central_crc32_end| {
                            bytes.get(central_crc32_start..central_crc32_end)
                        })
                });
            if let (Some(data), Some(central_crc32)) = (data, central_crc32) {
                // Store file view by file index.
                files.insert(
                    index,
                    NpyView {
                        data,
                        central_crc32,
                    },
                );
                // Increment index of uncompressed files.
                index += 1;
            } else {
                return Err(ZipError::InvalidArchive("Length overflow").into());
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

    /// Returns an immutable `.npy` file view by name.
    pub fn by_name(&self, name: &str) -> Result<NpyView<'a>, ViewNpzError> {
        self.by_index(
            self.names
                .get(name)
                .copied()
                .ok_or(ZipError::FileNotFound)?,
        )
    }

    /// Returns an immutable `.npy` file view by index in `0..len()`.
    pub fn by_index(&self, index: usize) -> Result<NpyView<'a>, ViewNpzError> {
        self.files
            .get(&index)
            .copied()
            .ok_or_else(|| ZipError::FileNotFound.into())
    }
}

/// Immutable view of memory-mapped `.npy` files within an `.npz` file.
///
/// Does **not** automatically [verify](`Self::verify`) CRC-32 checksum.
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

    /// Returns an immutable view of a memory-mapped `.npy` file.
    ///
    /// Iterates over `bool` array to ensure `0x00`/`0x01` values.
    pub fn view<A, D>(&self) -> Result<ArrayView<A, D>, ViewNpzError>
    where
        A: ViewElement,
        D: Dimension,
    {
        Ok(ArrayView::view_npy(self.data)?)
    }
}

/// Mutable view for memory-mapped `.npz` files.
///
/// The primary use-case for this is modifying `.npy` files within a
/// memory-mapped `.npz` archive. Modifying the elements in the view will modify
/// the file. Modifying the shape/strides of the view will *not* modify the
/// shape/strides of the array in the file.
///
/// # Notes
///
/// - For types for which not all bit patterns are valid, such as `bool`, the
///   implementation iterates over all of the elements when creating the view
///   to ensure they have a valid bit pattern.
///
/// - The `.npy` files within the `.npz` archive must be at least 16-byte or
///   ideally 64-byte aligned. Archives not created by this crate can be aligned
///   with the help of the CLI tool [`rezip`] as in `rezip in.npz -o out.npz`.
///
/// [`rezip`]: https://crates.io/crates/rezip
///
/// # Example
///
/// This is an example of opening a mutably memory-mapped `.npz` archive as an
/// [`NpzViewMut`] providing an [`NpyViewMut`] for each uncompressed `.npy` file
/// within the archive which can be accessed via [`NpyViewMut::view`] as
/// immutable [`ArrayView`] or via [`NpyViewMut::view_mut`] as mutable
/// [`ArrayViewMut`]. Changes to the data in the view will modify the underlying
/// file within the archive.
///
/// This example uses the [`memmap2`](https://crates.io/crates/memmap2) crate
/// because that appears to be the best-maintained memory-mapping crate at the
/// moment, but [`Self::new`] takes a `&mut [u8]` instead of a file so that you
/// can use the memory-mapping crate you're most comfortable with.
///
/// # Example
///
/// ```
/// use std::fs::OpenOptions;
/// use memmap2::MmapOptions;
/// use ndarray_npy::{NpzViewMut, ViewNpzError};
/// use ndarray::Ix1;
///
/// // Open `.npz` archive of uncompressed `.npy` files in native endian.
/// #[cfg(target_endian = "little")]
/// let mut file = OpenOptions::new().read(true).write(true)
///     .open("tests/examples_little_endian_64_byte_aligned.npz").unwrap();
/// #[cfg(target_endian = "big")]
/// let mut file = OpenOptions::new().read(true).write(true)
///     .open("tests/examples_big_endian_64_byte_aligned.npz").unwrap();
/// // Memory-map `.npz` archive of 64-byte aligned `.npy` files.
/// let mut mmap = unsafe { MmapOptions::new().map_mut(&file).unwrap() };
/// let mut npz = NpzViewMut::new(&mut mmap)?;
/// // List uncompressed files only.
/// for npy in npz.names() {
///     println!("{}", npy);
/// }
/// // Get mutable `.npy` views of both arrays at the same time.
/// let mut x_npy_view_mut = npz.by_name("i64.npy")?;
/// let mut y_npy_view_mut = npz.by_name("f64.npy")?;
/// // Optionally verify CRC-32 checksums.
/// x_npy_view_mut.verify()?;
/// y_npy_view_mut.verify()?;
/// // Get and print mutable `ArrayViewMut`s.
/// let x_array_view_mut = x_npy_view_mut.view_mut::<i64, Ix1>()?;
/// let y_array_view_mut = y_npy_view_mut.view_mut::<f64, Ix1>()?;
/// println!("{}", x_array_view_mut);
/// println!("{}", y_array_view_mut);
/// // Update CRC-32 checksums after changes. Automatically updated on `drop()`.
/// x_npy_view_mut.update();
/// y_npy_view_mut.update();
/// # Ok::<(), ViewNpzError>(())
/// ```
#[derive(Debug)]
pub struct NpzViewMut<'a> {
    files: HashMap<usize, NpyViewMut<'a>>,
    names: HashMap<String, usize>,
}

impl<'a> NpzViewMut<'a> {
    /// Creates a new mutable view of a memory-mapped `.npz` file.
    pub fn new(mut bytes: &'a mut [u8]) -> Result<Self, ViewNpzError> {
        let mut zip = ZipArchive::new(Cursor::new(&bytes))?;
        let mut names = HashMap::new();
        let mut ranges = HashMap::new();
        let mut splits = BTreeMap::new();
        let mut index = 0;
        for zip_index in 0..zip.len() {
            let file = zip.by_index(zip_index)?;
            // Skip directories and compressed files.
            if file.is_dir() || file.compression() != CompressionMethod::Stored {
                continue;
            }
            // Store file index by file names.
            let name = file.name().to_string();
            names.insert(name, index);
            // Get local CRC-32 range.
            let crc32: Option<Range<usize>> = file
                .header_start()
                .try_into()
                .ok()
                .and_then(|header_start: usize| header_start.checked_add(14))
                .and_then(|crc32_start| {
                    crc32_start
                        .checked_add(4)
                        .map(|crc32_end| crc32_start..crc32_end)
                });
            // Get data range.
            let data: Option<Range<usize>> =
                file.data_start()
                    .try_into()
                    .ok()
                    .and_then(|data_start: usize| {
                        file.size()
                            .try_into()
                            .ok()
                            .and_then(|size: usize| data_start.checked_add(size))
                            .map(|data_end| data_start..data_end)
                    });
            // Get central CRC-32 range.
            let central_crc32: Option<Range<usize>> = file
                .central_header_start()
                .try_into()
                .ok()
                .and_then(|central_header_start: usize| central_header_start.checked_add(16))
                .and_then(|central_crc32_start| {
                    central_crc32_start
                        .checked_add(4)
                        .map(|central_crc32_end| central_crc32_start..central_crc32_end)
                });
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
                return Err(ZipError::InvalidArchive("Length overflow").into());
            }
        }
        // Split and store borrows by their range starts.
        let mut offset = 0;
        let mut slices = HashMap::new();
        for (&start, &end) in &splits {
            // Split off leading bytes.
            let mid = start - offset;
            if mid > bytes.len() {
                return Err(ZipError::InvalidArchive("Offset exceeds EOF").into());
            }
            let (slice, remaining_bytes) = bytes.split_at_mut(mid);
            offset += slice.len();
            // Split off leading borrow of interest.
            let mid = end - offset;
            if mid > remaining_bytes.len() {
                return Err(ZipError::InvalidArchive("Length exceeds EOF").into());
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
            let crc32: Option<&mut [u8]> = slices.remove(&crc32.start);
            let data: Option<&mut [u8]> = slices.remove(&data.start);
            let central_crc32: Option<&mut [u8]> = slices.remove(&central_crc32.start);
            if let (Some(crc32), Some(data), Some(central_crc32)) = (crc32, data, central_crc32) {
                files.insert(
                    index,
                    NpyViewMut {
                        crc32,
                        data,
                        central_crc32,
                    },
                );
            } else {
                return Err(ZipError::InvalidArchive("Ambiguous offsets").into());
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

    /// Moves a mutable `.npy` file view by name out of the `.npz` file view.
    pub fn by_name(&mut self, name: &str) -> Result<NpyViewMut<'a>, ViewNpzError> {
        self.by_index(
            self.names
                .get(name)
                .copied()
                .ok_or(ZipError::FileNotFound)?,
        )
    }

    /// Moves a mutable `.npy` file view by index in `0..len()` out of the `.npz` file view.
    pub fn by_index(&mut self, index: usize) -> Result<NpyViewMut<'a>, ViewNpzError> {
        if index > self.names.len() {
            Err(ZipError::FileNotFound.into())
        } else {
            self.files
                .remove(&index)
                .ok_or(ViewNpzError::MovedNpyViewMut)
        }
    }
}

/// Mutable view of memory-mapped `.npy` files within an `.npz` file.
///
/// Does **not** automatically [verify](`Self::verify`) the CRC-32 checksum but **does**
/// [update](`Self::update`) it on [`Drop::drop`].
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

    /// Returns an immutable view of a memory-mapped `.npy` file.
    ///
    /// Iterates over `bool` array to ensure `0x00`/`0x01` values.
    pub fn view<A, D>(&self) -> Result<ArrayView<A, D>, ViewNpzError>
    where
        A: ViewElement,
        D: Dimension,
    {
        Ok(ArrayView::<A, D>::view_npy(self.data)?)
    }

    /// Returns a mutable view of a memory-mapped `.npy` file.
    ///
    /// Iterates over `bool` array to ensure `0x00`/`0x01` values.
    pub fn view_mut<A, D>(&mut self) -> Result<ArrayViewMut<A, D>, ViewNpzError>
    where
        A: ViewMutElement,
        D: Dimension,
    {
        Ok(ArrayViewMut::<A, D>::view_mut_npy(self.data)?)
    }

    /// Updates CRC-32 checksum by reading the whole array.
    ///
    /// Automatically updated on [`Drop::drop`].
    pub fn update(&mut self) {
        self.central_crc32
            .copy_from_slice(&crc32_update(&self.data));
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
        Err(ZipError::Io(io::Error::new(
            io::ErrorKind::Other,
            "Invalid checksum",
        )))
    }
}

fn crc32_update(bytes: &[u8]) -> [u8; 4] {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(bytes);
    hasher.finalize().to_le_bytes()
}
