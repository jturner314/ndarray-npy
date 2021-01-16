pub mod header;

use self::header::{
    FormatHeaderError, Header, ParseHeaderError, ReadHeaderError, WriteHeaderError,
};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use ndarray::prelude::*;
use ndarray::{Data, DataOwned, IntoDimension};
use py_literal::Value as PyValue;
use std::error::Error;
use std::fmt;
use std::io;
use std::mem;
use std::slice;

/// Read an `.npy` file located at the specified path.
///
/// This is a convience function for using `File::open` followed by
/// [`ReadNpyExt::read_npy`](trait.ReadNpyExt.html#tymethod.read_npy).
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use ndarray_npy::read_npy;
/// # use ndarray_npy::ReadNpyError;
///
/// let arr: Array2<i32> = read_npy("resources/array.npy")?;
/// # println!("arr = {}", arr);
/// # Ok::<_, ReadNpyError>(())
/// ```
pub fn read_npy<P, T>(path: P) -> Result<T, ReadNpyError>
where
    P: AsRef<std::path::Path>,
    T: ReadNpyExt,
{
    T::read_npy(std::fs::File::open(path)?)
}

/// Writes an array to an `.npy` file at the specified path.
///
/// This function will create the file if it does not exist, or overwrite it if
/// it does.
///
/// This is a convenience function for using `File::create` followed by
/// [`WriteNpyExt::write_npy`](trait.WriteNpyExt.html#tymethod.write_npy).
///
/// # Example
///
/// ```no_run
/// use ndarray::array;
/// use ndarray_npy::write_npy;
/// # use ndarray_npy::WriteNpyError;
///
/// let arr = array![[1, 2, 3], [4, 5, 6]];
/// write_npy("array.npy", &arr)?;
/// # Ok::<_, WriteNpyError>(())
/// ```
pub fn write_npy<P, T>(path: P, array: &T) -> Result<(), WriteNpyError>
where
    P: AsRef<std::path::Path>,
    T: WriteNpyExt,
{
    array.write_npy(std::fs::File::create(path)?)
}

/// An error writing array data.
#[derive(Debug)]
pub enum WriteDataError {
    /// An error caused by I/O.
    Io(io::Error),
    /// An error formatting the data.
    FormatData(Box<dyn Error + Send + Sync + 'static>),
}

impl Error for WriteDataError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            WriteDataError::Io(err) => Some(err),
            WriteDataError::FormatData(err) => Some(&**err),
        }
    }
}

impl fmt::Display for WriteDataError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            WriteDataError::Io(err) => write!(f, "I/O error: {}", err),
            WriteDataError::FormatData(err) => write!(f, "error formatting data: {}", err),
        }
    }
}

impl From<io::Error> for WriteDataError {
    fn from(err: io::Error) -> WriteDataError {
        WriteDataError::Io(err)
    }
}

/// An array element type that can be written to an `.npy` or `.npz` file.
pub unsafe trait WritableElement: Sized {
    /// Returns a descriptor of the type that can be used in the header.
    fn type_descriptor() -> PyValue;

    /// Writes a single instance of `Self` to the writer.
    fn write<W: io::Write>(&self, writer: W) -> Result<(), WriteDataError>;

    /// Writes a slice of `Self` to the writer.
    fn write_slice<W: io::Write>(slice: &[Self], writer: W) -> Result<(), WriteDataError>;
}

/// An error writing a `.npy` file.
#[derive(Debug)]
pub enum WriteNpyError {
    /// An error caused by I/O.
    Io(io::Error),
    /// An error formatting the header.
    FormatHeader(FormatHeaderError),
    /// An error formatting the data.
    FormatData(Box<dyn Error + Send + Sync + 'static>),
}

impl Error for WriteNpyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            WriteNpyError::Io(err) => Some(err),
            WriteNpyError::FormatHeader(err) => Some(err),
            WriteNpyError::FormatData(err) => Some(&**err),
        }
    }
}

impl fmt::Display for WriteNpyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            WriteNpyError::Io(err) => write!(f, "I/O error: {}", err),
            WriteNpyError::FormatHeader(err) => write!(f, "error formatting header: {}", err),
            WriteNpyError::FormatData(err) => write!(f, "error formatting data: {}", err),
        }
    }
}

impl From<io::Error> for WriteNpyError {
    fn from(err: io::Error) -> WriteNpyError {
        WriteNpyError::Io(err)
    }
}

impl From<WriteHeaderError> for WriteNpyError {
    fn from(err: WriteHeaderError) -> WriteNpyError {
        match err {
            WriteHeaderError::Io(err) => WriteNpyError::Io(err),
            WriteHeaderError::Format(err) => WriteNpyError::FormatHeader(err),
        }
    }
}

impl From<FormatHeaderError> for WriteNpyError {
    fn from(err: FormatHeaderError) -> WriteNpyError {
        WriteNpyError::FormatHeader(err)
    }
}

impl From<WriteDataError> for WriteNpyError {
    fn from(err: WriteDataError) -> WriteNpyError {
        match err {
            WriteDataError::Io(err) => WriteNpyError::Io(err),
            WriteDataError::FormatData(err) => WriteNpyError::FormatData(err),
        }
    }
}

/// Extension trait for writing `ArrayBase` to `.npy` files.
///
/// # Example
///
/// ```no_run
/// use ndarray::{array, Array2};
/// use ndarray_npy::WriteNpyExt;
/// use std::fs::File;
/// # use ndarray_npy::WriteNpyError;
///
/// let arr: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];
/// let writer = File::create("array.npy")?;
/// arr.write_npy(writer)?;
/// # Ok::<_, WriteNpyError>(())
/// ```
pub trait WriteNpyExt {
    /// Writes the array to `writer` in [`.npy`
    /// format](https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html).
    ///
    /// This function is the Rust equivalent of
    /// [`numpy.save`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html).
    fn write_npy<W: io::Write>(&self, writer: W) -> Result<(), WriteNpyError>;
}

impl<A, S, D> WriteNpyExt for ArrayBase<S, D>
where
    A: WritableElement,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn write_npy<W: io::Write>(&self, mut writer: W) -> Result<(), WriteNpyError> {
        let write_contiguous = |mut writer: W, fortran_order: bool| {
            Header {
                type_descriptor: A::type_descriptor(),
                fortran_order,
                shape: self.shape().to_owned(),
            }
            .write(&mut writer)?;
            A::write_slice(self.as_slice_memory_order().unwrap(), &mut writer)?;
            Ok(())
        };
        if self.is_standard_layout() {
            write_contiguous(writer, false)
        } else if self.view().reversed_axes().is_standard_layout() {
            write_contiguous(writer, true)
        } else {
            Header {
                type_descriptor: A::type_descriptor(),
                fortran_order: false,
                shape: self.shape().to_owned(),
            }
            .write(&mut writer)?;
            for elem in self.iter() {
                elem.write(&mut writer)?;
            }
            Ok(())
        }
    }
}

/// An error reading array data.
#[derive(Debug)]
pub enum ReadDataError {
    /// An error caused by I/O.
    Io(io::Error),
    /// The type descriptor does not match the element type.
    WrongDescriptor(PyValue),
    /// The file does not contain all the data described in the header.
    MissingData,
    /// Extra bytes are present between the end of the data and the end of the
    /// file.
    ExtraBytes(usize),
    /// An error parsing the data.
    ParseData(Box<dyn Error + Send + Sync + 'static>),
}

impl Error for ReadDataError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ReadDataError::Io(err) => Some(err),
            ReadDataError::WrongDescriptor(_) => None,
            ReadDataError::MissingData => None,
            ReadDataError::ExtraBytes(_) => None,
            ReadDataError::ParseData(err) => Some(&**err),
        }
    }
}

impl fmt::Display for ReadDataError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ReadDataError::Io(err) => write!(f, "I/O error: {}", err),
            ReadDataError::WrongDescriptor(desc) => {
                write!(f, "incorrect descriptor ({}) for this type", desc)
            }
            ReadDataError::MissingData => write!(f, "reached EOF before reading all data"),
            ReadDataError::ExtraBytes(num_extra_bytes) => {
                write!(f, "file had {} extra bytes before EOF", num_extra_bytes)
            }
            ReadDataError::ParseData(err) => write!(f, "error parsing data: {}", err),
        }
    }
}

impl From<io::Error> for ReadDataError {
    /// Performs the conversion.
    ///
    /// If the error kind is `UnexpectedEof`, the `MissingData` variant is
    /// returned. Otherwise, the `Io` variant is returned.
    fn from(err: io::Error) -> ReadDataError {
        if err.kind() == io::ErrorKind::UnexpectedEof {
            ReadDataError::MissingData
        } else {
            ReadDataError::Io(err)
        }
    }
}

/// An array element type that can be read from an `.npy` or `.npz` file.
pub trait ReadableElement: Sized {
    /// Reads to the end of the `reader`, creating a `Vec` of length `len`.
    ///
    /// This method should return `Err(_)` in at least the following cases:
    ///
    /// * if the `type_desc` does not match `Self`
    /// * if the `reader` has fewer elements than `len`
    /// * if the `reader` has extra bytes after reading `len` elements
    fn read_to_end_exact_vec<R: io::Read>(
        reader: R,
        type_desc: &PyValue,
        len: usize,
    ) -> Result<Vec<Self>, ReadDataError>;
}

/// An error reading a `.npy` file.
#[derive(Debug)]
pub enum ReadNpyError {
    /// An error caused by I/O.
    Io(io::Error),
    /// An error parsing the file header.
    ParseHeader(ParseHeaderError),
    /// An error parsing the data.
    ParseData(Box<dyn Error + Send + Sync + 'static>),
    /// Overflow while computing the length of the array (in units of bytes or
    /// the number of elements) from the shape described in the file header.
    LengthOverflow,
    /// An error caused by incorrect `Dimension` type.
    WrongNdim(Option<usize>, usize),
    /// The type descriptor does not match the element type.
    WrongDescriptor(PyValue),
    /// The file does not contain all the data described in the header.
    MissingData,
    /// Extra bytes are present between the end of the data and the end of the
    /// file.
    ExtraBytes(usize),
}

impl Error for ReadNpyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ReadNpyError::Io(err) => Some(err),
            ReadNpyError::ParseHeader(err) => Some(err),
            ReadNpyError::ParseData(err) => Some(&**err),
            ReadNpyError::LengthOverflow => None,
            ReadNpyError::WrongNdim(_, _) => None,
            ReadNpyError::WrongDescriptor(_) => None,
            ReadNpyError::MissingData => None,
            ReadNpyError::ExtraBytes(_) => None,
        }
    }
}

impl fmt::Display for ReadNpyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ReadNpyError::Io(err) => write!(f, "I/O error: {}", err),
            ReadNpyError::ParseHeader(err) => write!(f, "error parsing header: {}", err),
            ReadNpyError::ParseData(err) => write!(f, "error parsing data: {}", err),
            ReadNpyError::LengthOverflow => write!(f, "overflow computing length from shape"),
            ReadNpyError::WrongNdim(expected, actual) => write!(
                f,
                "ndim {} of array did not match Dimension type with NDIM = {:?}",
                actual, expected
            ),
            ReadNpyError::WrongDescriptor(desc) => {
                write!(f, "incorrect descriptor ({}) for this type", desc)
            }
            ReadNpyError::MissingData => write!(f, "reached EOF before reading all data"),
            ReadNpyError::ExtraBytes(num_extra_bytes) => {
                write!(f, "file had {} extra bytes before EOF", num_extra_bytes)
            }
        }
    }
}

impl From<io::Error> for ReadNpyError {
    fn from(err: io::Error) -> ReadNpyError {
        ReadNpyError::Io(err)
    }
}

impl From<ReadHeaderError> for ReadNpyError {
    fn from(err: ReadHeaderError) -> ReadNpyError {
        match err {
            ReadHeaderError::Io(err) => ReadNpyError::Io(err),
            ReadHeaderError::Parse(err) => ReadNpyError::ParseHeader(err),
        }
    }
}

impl From<ParseHeaderError> for ReadNpyError {
    fn from(err: ParseHeaderError) -> ReadNpyError {
        ReadNpyError::ParseHeader(err)
    }
}

impl From<ReadDataError> for ReadNpyError {
    fn from(err: ReadDataError) -> ReadNpyError {
        match err {
            ReadDataError::Io(err) => ReadNpyError::Io(err),
            ReadDataError::WrongDescriptor(desc) => ReadNpyError::WrongDescriptor(desc),
            ReadDataError::MissingData => ReadNpyError::MissingData,
            ReadDataError::ExtraBytes(nbytes) => ReadNpyError::ExtraBytes(nbytes),
            ReadDataError::ParseData(err) => ReadNpyError::ParseData(err),
        }
    }
}

/// Extension trait for reading `Array` from `.npy` files.
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use ndarray_npy::ReadNpyExt;
/// use std::fs::File;
/// # use ndarray_npy::ReadNpyError;
///
/// let reader = File::open("resources/array.npy")?;
/// let arr = Array2::<i32>::read_npy(reader)?;
/// # println!("arr = {}", arr);
/// # Ok::<_, ReadNpyError>(())
/// ```
pub trait ReadNpyExt: Sized {
    /// Reads the array from `reader` in [`.npy`
    /// format](https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html).
    ///
    /// This function is the Rust equivalent of
    /// [`numpy.load`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html)
    /// for `.npy` files.
    fn read_npy<R: io::Read>(reader: R) -> Result<Self, ReadNpyError>;
}

impl<A, S, D> ReadNpyExt for ArrayBase<S, D>
where
    A: ReadableElement,
    S: DataOwned<Elem = A>,
    D: Dimension,
{
    fn read_npy<R: io::Read>(mut reader: R) -> Result<Self, ReadNpyError> {
        let header = Header::from_reader(&mut reader)?;
        let shape = header.shape.into_dimension();
        let ndim = shape.ndim();
        let len = shape_length_checked::<A>(&shape).ok_or(ReadNpyError::LengthOverflow)?;
        let data = A::read_to_end_exact_vec(&mut reader, &header.type_descriptor, len)?;
        ArrayBase::from_shape_vec(shape.set_f(header.fortran_order), data)
            .unwrap()
            .into_dimensionality()
            .map_err(|_| ReadNpyError::WrongNdim(D::NDIM, ndim))
    }
}

macro_rules! impl_writable_primitive {
    ($elem:ty, $little_desc:expr, $big_desc:expr) => {
        unsafe impl WritableElement for $elem {
            fn type_descriptor() -> PyValue {
                if cfg!(target_endian = "little") {
                    PyValue::String($little_desc.into())
                } else if cfg!(target_endian = "big") {
                    PyValue::String($big_desc.into())
                } else {
                    unreachable!()
                }
            }

            fn write<W: io::Write>(&self, mut writer: W) -> Result<(), WriteDataError> {
                // Function to ensure lifetime of bytes slice is correct.
                fn cast(self_: &$elem) -> &[u8] {
                    unsafe {
                        let ptr: *const $elem = self_;
                        slice::from_raw_parts(ptr.cast::<u8>(), mem::size_of::<$elem>())
                    }
                }
                writer.write_all(cast(self))?;
                Ok(())
            }

            fn write_slice<W: io::Write>(
                slice: &[Self],
                mut writer: W,
            ) -> Result<(), WriteDataError> {
                // Function to ensure lifetime of bytes slice is correct.
                fn cast(slice: &[$elem]) -> &[u8] {
                    unsafe {
                        slice::from_raw_parts(
                            slice.as_ptr().cast::<u8>(),
                            slice.len() * mem::size_of::<$elem>(),
                        )
                    }
                }
                writer.write_all(cast(slice))?;
                Ok(())
            }
        }
    };
}

/// Returns `Ok(_)` iff the `reader` had no more bytes on entry to this
/// function.
///
/// **Warning** This will consume the remainder of the reader.
pub fn check_for_extra_bytes<R: io::Read>(reader: &mut R) -> Result<(), ReadDataError> {
    let num_extra_bytes = reader.read_to_end(&mut Vec::new())?;
    if num_extra_bytes == 0 {
        Ok(())
    } else {
        Err(ReadDataError::ExtraBytes(num_extra_bytes))
    }
}

macro_rules! impl_readable_primitive_one_byte {
    ($elem:ty, [$($desc:expr),*], $zero:expr, $read_into:ident) => {
        impl ReadableElement for $elem {
            fn read_to_end_exact_vec<R: io::Read>(
                mut reader: R,
                type_desc: &PyValue,
                len: usize,
            ) -> Result<Vec<Self>, ReadDataError> {
                match *type_desc {
                    PyValue::String(ref s) if $(s == $desc)||* => {
                        let mut out = vec![$zero; len];
                        reader.$read_into(&mut out)?;
                        check_for_extra_bytes(&mut reader)?;
                        Ok(out)
                    }
                    ref other => Err(ReadDataError::WrongDescriptor(other.clone())),
                }
            }
        }
    };
}

macro_rules! impl_view_and_view_mut_primitive_one_byte {
    ($elem:ty, [$($desc:expr),*]) => {
        impl ViewElement for $elem {
            fn bytes_as_slice<'a>(
                bytes: &'a [u8],
                type_desc: &PyValue,
                len: usize,
            ) -> Result<&'a [Self], ViewDataError> {
                match *type_desc {
                    PyValue::String(ref s) if $(s == $desc)||* => unsafe {
                        bytes_as_slice(bytes, len)
                    }
                    ref other => Err(ViewDataError::WrongDescriptor(other.clone())),
                }
            }
        }

        impl ViewMutElement for $elem {
            fn bytes_as_mut_slice<'a>(
                bytes: &'a mut [u8],
                type_desc: &PyValue,
                len: usize,
            ) -> Result<&'a mut [Self], ViewDataError> {
                match *type_desc {
                    PyValue::String(ref s) if $(s == $desc)||* => unsafe {
                        bytes_as_mut_slice(bytes, len)
                    }
                    ref other => Err(ViewDataError::WrongDescriptor(other.clone())),
                }
            }
        }
    };
}

macro_rules! impl_primitive_one_byte {
    ($elem:ty, $write_desc:expr, [$($read_desc:expr),*], $zero:expr, $read_into:ident) => {
        impl_writable_primitive!($elem, $write_desc, $write_desc);
        impl_readable_primitive_one_byte!($elem, [$($read_desc),*], $zero, $read_into);
        impl_view_and_view_mut_primitive_one_byte!($elem, [$($read_desc),*]);
    };
}

impl_primitive_one_byte!(i8, "|i1", ["|i1", "i1", "b"], 0, read_i8_into);
impl_primitive_one_byte!(u8, "|u1", ["|u1", "u1", "B"], 0, read_exact);

macro_rules! impl_readable_primitive_multi_byte {
    ($elem:ty, $little_desc:expr, $big_desc:expr, $zero:expr, $read_into:ident) => {
        impl ReadableElement for $elem {
            fn read_to_end_exact_vec<R: io::Read>(
                mut reader: R,
                type_desc: &PyValue,
                len: usize,
            ) -> Result<Vec<Self>, ReadDataError> {
                let mut out = vec![$zero; len];
                match *type_desc {
                    PyValue::String(ref s) if s == $little_desc => {
                        reader.$read_into::<LittleEndian>(&mut out)?;
                    }
                    PyValue::String(ref s) if s == $big_desc => {
                        reader.$read_into::<BigEndian>(&mut out)?;
                    }
                    ref other => {
                        return Err(ReadDataError::WrongDescriptor(other.clone()));
                    }
                }
                check_for_extra_bytes(&mut reader)?;
                Ok(out)
            }
        }
    };
}

macro_rules! impl_view_and_view_mut_primitive_multi_byte {
    ($elem:ty, $native_desc:expr, $non_native_desc:expr) => {
        impl ViewElement for $elem {
            fn bytes_as_slice<'a>(
                bytes: &'a [u8],
                type_desc: &PyValue,
                len: usize,
            ) -> Result<&'a [Self], ViewDataError> {
                match *type_desc {
                    PyValue::String(ref s) if s == $native_desc => unsafe {
                        bytes_as_slice(bytes, len)
                    },
                    PyValue::String(ref s) if s == $non_native_desc => {
                        Err(ViewDataError::NonNativeEndian)
                    }
                    ref other => Err(ViewDataError::WrongDescriptor(other.clone())),
                }
            }
        }

        impl ViewMutElement for $elem {
            fn bytes_as_mut_slice<'a>(
                bytes: &'a mut [u8],
                type_desc: &PyValue,
                len: usize,
            ) -> Result<&'a mut [Self], ViewDataError> {
                match *type_desc {
                    PyValue::String(ref s) if s == $native_desc => unsafe {
                        bytes_as_mut_slice(bytes, len)
                    },
                    PyValue::String(ref s) if s == $non_native_desc => {
                        Err(ViewDataError::NonNativeEndian)
                    }
                    ref other => Err(ViewDataError::WrongDescriptor(other.clone())),
                }
            }
        }
    };
}

macro_rules! impl_primitive_multi_byte {
    ($elem:ty, $little_desc:expr, $big_desc:expr, $zero:expr, $read_into:ident) => {
        impl_writable_primitive!($elem, $little_desc, $big_desc);
        impl_readable_primitive_multi_byte!($elem, $little_desc, $big_desc, $zero, $read_into);
        #[cfg(target_endian = "little")]
        impl_view_and_view_mut_primitive_multi_byte!($elem, $little_desc, $big_desc);
        #[cfg(target_endian = "big")]
        impl_view_and_view_mut_primitive_multi_byte!($elem, $big_desc, $little_desc);
    };
}

impl_primitive_multi_byte!(i16, "<i2", ">i2", 0, read_i16_into);
impl_primitive_multi_byte!(i32, "<i4", ">i4", 0, read_i32_into);
impl_primitive_multi_byte!(i64, "<i8", ">i8", 0, read_i64_into);

impl_primitive_multi_byte!(u16, "<u2", ">u2", 0, read_u16_into);
impl_primitive_multi_byte!(u32, "<u4", ">u4", 0, read_u32_into);
impl_primitive_multi_byte!(u64, "<u8", ">u8", 0, read_u64_into);

impl_primitive_multi_byte!(f32, "<f4", ">f4", 0., read_f32_into);
impl_primitive_multi_byte!(f64, "<f8", ">f8", 0., read_f64_into);

/// An error parsing a `bool` from a byte.
#[derive(Debug)]
struct ParseBoolError {
    bad_value: u8,
}

impl Error for ParseBoolError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl fmt::Display for ParseBoolError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "error parsing value {:#04x} as a bool", self.bad_value)
    }
}

impl From<ParseBoolError> for ReadDataError {
    fn from(err: ParseBoolError) -> ReadDataError {
        ReadDataError::ParseData(Box::new(err))
    }
}

impl From<ParseBoolError> for ViewDataError {
    fn from(err: ParseBoolError) -> ViewDataError {
        ViewDataError::InvalidData(Box::new(err))
    }
}

/// Returns `Ok(_)` iff each of the bytes is a valid bitwise representation for
/// `bool`.
///
/// In other words, this checks that each byte is `0x00` or `0x01`, which is
/// important for the bytes to be reinterpreted as `bool`, since creating a
/// `bool` with an invalid value is undefined behavior. Rust guarantees that
/// `false` is represented as `0x00` and `true` is represented as `0x01`.
fn check_valid_for_bool(bytes: &[u8]) -> Result<(), ParseBoolError> {
    for &byte in bytes {
        if byte > 1 {
            return Err(ParseBoolError { bad_value: byte });
        }
    }
    Ok(())
}

impl ReadableElement for bool {
    fn read_to_end_exact_vec<R: io::Read>(
        mut reader: R,
        type_desc: &PyValue,
        len: usize,
    ) -> Result<Vec<Self>, ReadDataError> {
        match *type_desc {
            PyValue::String(ref s) if s == "|b1" => {
                // Read the data.
                let mut bytes: Vec<u8> = vec![0; len];
                reader.read_exact(&mut bytes)?;
                check_for_extra_bytes(&mut reader)?;

                // Check that the data is valid for interpretation as `bool`.
                check_valid_for_bool(&bytes)?;

                // Cast the `Vec<u8>` to `Vec<bool>`.
                {
                    let ptr: *mut u8 = bytes.as_mut_ptr();
                    let len: usize = bytes.len();
                    let cap: usize = bytes.capacity();
                    mem::forget(bytes);
                    // This is safe because:
                    //
                    // * All elements are valid `bool`s. (See the call to
                    //   `check_valid_for_bool` above.)
                    //
                    // * `ptr` was originally allocated by `Vec`.
                    //
                    // * `bool` has the same size and alignment as `u8`.
                    //
                    // * `len` and `cap` are copied directly from the
                    //   `Vec<u8>`, so `len <= cap` and `cap` is the capacity
                    //   `ptr` was allocated with.
                    Ok(unsafe { Vec::from_raw_parts(ptr.cast::<bool>(), len, cap) })
                }
            }
            ref other => Err(ReadDataError::WrongDescriptor(other.clone())),
        }
    }
}

// Rust guarantees that `bool` is one byte, the bitwise representation of
// `false` is `0x00`, and the bitwise representation of `true` is `0x01`, so we
// can just cast the data in-place.
impl_writable_primitive!(bool, "|b1", "|b1");

impl ViewElement for bool {
    fn bytes_as_slice<'a>(
        bytes: &'a [u8],
        type_desc: &PyValue,
        len: usize,
    ) -> Result<&'a [Self], ViewDataError> {
        match *type_desc {
            PyValue::String(ref s) if s == "|b1" => {
                check_valid_for_bool(bytes)?;
                unsafe { bytes_as_slice(bytes, len) }
            }
            ref other => Err(ViewDataError::WrongDescriptor(other.clone())),
        }
    }
}

impl ViewMutElement for bool {
    fn bytes_as_mut_slice<'a>(
        bytes: &'a mut [u8],
        type_desc: &PyValue,
        len: usize,
    ) -> Result<&'a mut [Self], ViewDataError> {
        match *type_desc {
            PyValue::String(ref s) if s == "|b1" => {
                check_valid_for_bool(bytes)?;
                unsafe { bytes_as_mut_slice(bytes, len) }
            }
            ref other => Err(ViewDataError::WrongDescriptor(other.clone())),
        }
    }
}

/// An error viewing a `.npy` file.
#[derive(Debug)]
#[non_exhaustive]
pub enum ViewNpyError {
    /// An error caused by I/O.
    Io(io::Error),
    /// An error parsing the file header.
    ParseHeader(ParseHeaderError),
    /// Some of the data is invalid for the element type.
    InvalidData(Box<dyn Error + Send + Sync + 'static>),
    /// Overflow while computing the length of the array (in units of bytes or
    /// the number of elements) from the shape described in the file header.
    LengthOverflow,
    /// An error caused by incorrect `Dimension` type.
    WrongNdim(Option<usize>, usize),
    /// The type descriptor does not match the element type.
    WrongDescriptor(PyValue),
    /// The type descriptor does not match the native endianness.
    NonNativeEndian,
    /// The start of the data is not properly aligned for the element type.
    MisalignedData,
    /// The file does not contain all the data described in the header.
    MissingBytes(usize),
    /// Extra bytes are present between the end of the data and the end of the
    /// file.
    ExtraBytes(usize),
}

impl Error for ViewNpyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ViewNpyError::Io(err) => Some(err),
            ViewNpyError::ParseHeader(err) => Some(err),
            ViewNpyError::InvalidData(err) => Some(&**err),
            ViewNpyError::LengthOverflow => None,
            ViewNpyError::WrongNdim(_, _) => None,
            ViewNpyError::WrongDescriptor(_) => None,
            ViewNpyError::NonNativeEndian => None,
            ViewNpyError::MisalignedData => None,
            ViewNpyError::MissingBytes(_) => None,
            ViewNpyError::ExtraBytes(_) => None,
        }
    }
}

impl fmt::Display for ViewNpyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ViewNpyError::Io(err) => write!(f, "I/O error: {}", err),
            ViewNpyError::ParseHeader(err) => write!(f, "error parsing header: {}", err),
            ViewNpyError::InvalidData(err) => write!(f, "invalid data for element type: {}", err),
            ViewNpyError::LengthOverflow => write!(f, "overflow computing length from shape"),
            ViewNpyError::WrongNdim(expected, actual) => write!(
                f,
                "ndim {} of array did not match Dimension type with NDIM = {:?}",
                actual, expected
            ),
            ViewNpyError::WrongDescriptor(desc) => {
                write!(f, "incorrect descriptor ({}) for this type", desc)
            }
            ViewNpyError::NonNativeEndian => {
                write!(f, "descriptor does not match native endianness")
            }
            ViewNpyError::MisalignedData => write!(
                f,
                "start of data is not properly aligned for the element type"
            ),
            ViewNpyError::MissingBytes(num_missing_bytes) => write!(
                f,
                "missing {} bytes of data specified in header",
                num_missing_bytes
            ),
            ViewNpyError::ExtraBytes(num_extra_bytes) => {
                write!(f, "file had {} extra bytes before EOF", num_extra_bytes)
            }
        }
    }
}

impl From<ReadHeaderError> for ViewNpyError {
    fn from(err: ReadHeaderError) -> ViewNpyError {
        match err {
            ReadHeaderError::Io(err) => ViewNpyError::Io(err),
            ReadHeaderError::Parse(err) => ViewNpyError::ParseHeader(err),
        }
    }
}

impl From<ParseHeaderError> for ViewNpyError {
    fn from(err: ParseHeaderError) -> ViewNpyError {
        ViewNpyError::ParseHeader(err)
    }
}

impl From<ViewDataError> for ViewNpyError {
    fn from(err: ViewDataError) -> ViewNpyError {
        match err {
            ViewDataError::WrongDescriptor(desc) => ViewNpyError::WrongDescriptor(desc),
            ViewDataError::NonNativeEndian => ViewNpyError::NonNativeEndian,
            ViewDataError::Misaligned => ViewNpyError::MisalignedData,
            ViewDataError::MissingBytes(nbytes) => ViewNpyError::MissingBytes(nbytes),
            ViewDataError::ExtraBytes(nbytes) => ViewNpyError::ExtraBytes(nbytes),
            ViewDataError::InvalidData(err) => ViewNpyError::InvalidData(err),
        }
    }
}

/// Extension trait for creating an [`ArrayView`] from a buffer containing an
/// `.npy` file.
///
/// The primary use-case for this is viewing a memory-mapped `.npy` file.
///
/// # Notes
///
/// - For types for which not all bit patterns are valid, such as `bool`, the
///   implementation iterates over all of the elements when creating the view
///   to ensure they have a valid bit pattern.
///
/// - The data in the buffer must be properly aligned for the element type.
///   Typically, this should not be a concern for memory-mapped files (unless
///   an option like `MAP_FIXED` is used), since memory mappings are usually
///   aligned to a page boundary, and the `.npy` format has padding such that
///   the header size is a multiple of 64 bytes.
///
/// # Example
///
/// This is an example of opening a readonly memory-mapped file as an
/// [`ArrayView`].
///
/// This example uses the [`memmap2`](https://crates.io/crates/memmap2) crate
/// because that appears to be the best-maintained memory-mapping crate at the
/// moment, but `view_npy` takes a `&[u8]` instead of a file so that you can
/// use the memory-mapping crate you're most comfortable with.
///
/// ```
/// use memmap2::Mmap;
/// use ndarray::ArrayView2;
/// use ndarray_npy::ViewNpyExt;
/// use std::fs::File;
///
/// let file = File::open("resources/array.npy")?;
/// let mmap = unsafe { Mmap::map(&file)? };
/// let view = ArrayView2::<i32>::view_npy(&mmap)?;
/// # println!("view = {}", view);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub trait ViewNpyExt<'a>: Sized {
    /// Creates an `ArrayView` from a buffer containing an `.npy` file.
    fn view_npy(buf: &'a [u8]) -> Result<Self, ViewNpyError>;
}

/// Extension trait for creating an [`ArrayViewMut`] from a mutable buffer
/// containing an `.npy` file.
///
/// The primary use-case for this is modifying a memory-mapped `.npy` file.
/// Modifying the elements in the view will modify the file. Modifying the
/// shape/strides of the view will *not* modify the shape/strides of the array
/// in the file.
///
/// Notes:
///
/// - For types for which not all bit patterns are valid, such as `bool`, the
///   implementation iterates over all of the elements when creating the view
///   to ensure they have a valid bit pattern.
///
/// - The data in the buffer must be properly aligned for the element type.
///   Typically, this should not be a concern for memory-mapped files (unless
///   an option like `MAP_FIXED` is used), since memory mappings are usually
///   aligned to a page boundary, and the `.npy` format has padding such that
///   the header size is a multiple of 64 bytes.
///
/// # Example
///
/// This is an example of opening a writable memory-mapped file as an
/// [`ArrayViewMut`]. Changes to the data in the view will modify the
/// underlying file.
///
/// This example uses the [`memmap2`](https://crates.io/crates/memmap2) crate
/// because that appears to be the best-maintained memory-mapping crate at the
/// moment, but `view_mut_npy` takes a `&mut [u8]` instead of a file so that
/// you can use the memory-mapping crate you're most comfortable with.
///
/// ```
/// use memmap2::MmapMut;
/// use ndarray::ArrayViewMut2;
/// use ndarray_npy::ViewMutNpyExt;
/// use std::fs;
///
/// let file = fs::OpenOptions::new()
///     .read(true)
///     .write(true)
///     .open("resources/array.npy")?;
/// let mut mmap = unsafe { MmapMut::map_mut(&file)? };
/// let view_mut = ArrayViewMut2::<i32>::view_mut_npy(&mut mmap)?;
/// # println!("view_mut = {}", view_mut);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub trait ViewMutNpyExt<'a>: Sized {
    /// Creates an `ArrayViewMut` from a mutable buffer containing an `.npy`
    /// file.
    fn view_mut_npy(buf: &'a mut [u8]) -> Result<Self, ViewNpyError>;
}

impl<'a, A, D> ViewNpyExt<'a> for ArrayView<'a, A, D>
where
    A: ViewElement,
    D: Dimension,
{
    fn view_npy(buf: &'a [u8]) -> Result<Self, ViewNpyError> {
        let mut reader = buf;
        let header = Header::from_reader(&mut reader)?;
        let shape = header.shape.into_dimension();
        let ndim = shape.ndim();
        let len = shape_length_checked::<A>(&shape).ok_or(ViewNpyError::LengthOverflow)?;
        let data = A::bytes_as_slice(&reader, &header.type_descriptor, len)?;
        ArrayView::from_shape(shape.set_f(header.fortran_order), data)
            .unwrap()
            .into_dimensionality()
            .map_err(|_| ViewNpyError::WrongNdim(D::NDIM, ndim))
    }
}

impl<'a, A, D> ViewMutNpyExt<'a> for ArrayViewMut<'a, A, D>
where
    A: ViewMutElement,
    D: Dimension,
{
    fn view_mut_npy(buf: &'a mut [u8]) -> Result<Self, ViewNpyError> {
        let mut reader = &*buf;
        let header = Header::from_reader(&mut reader)?;
        let shape = header.shape.into_dimension();
        let ndim = shape.ndim();
        let len = shape_length_checked::<A>(&shape).ok_or(ViewNpyError::LengthOverflow)?;
        let mid = buf.len() - reader.len();
        let data = A::bytes_as_mut_slice(&mut buf[mid..], &header.type_descriptor, len)?;
        ArrayViewMut::from_shape(shape.set_f(header.fortran_order), data)
            .unwrap()
            .into_dimensionality()
            .map_err(|_| ViewNpyError::WrongNdim(D::NDIM, ndim))
    }
}

/// An error viewing array data.
#[derive(Debug)]
#[non_exhaustive]
pub enum ViewDataError {
    /// The type descriptor does not match the element type.
    WrongDescriptor(PyValue),
    /// The type descriptor does not match the native endianness.
    NonNativeEndian,
    /// The start of the data is not properly aligned for the element type.
    Misaligned,
    /// The file does not contain all the data described in the header.
    MissingBytes(usize),
    /// Extra bytes are present between the end of the data and the end of the
    /// file.
    ExtraBytes(usize),
    /// Some of the data is invalid for the element type.
    InvalidData(Box<dyn Error + Send + Sync + 'static>),
}

impl Error for ViewDataError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ViewDataError::WrongDescriptor(_) => None,
            ViewDataError::NonNativeEndian => None,
            ViewDataError::Misaligned => None,
            ViewDataError::MissingBytes(_) => None,
            ViewDataError::ExtraBytes(_) => None,
            ViewDataError::InvalidData(err) => Some(&**err),
        }
    }
}

impl fmt::Display for ViewDataError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ViewDataError::WrongDescriptor(desc) => {
                write!(f, "incorrect descriptor ({}) for this type", desc)
            }
            ViewDataError::NonNativeEndian => {
                write!(f, "descriptor does not match native endianness")
            }
            ViewDataError::Misaligned => write!(
                f,
                "start of data is not properly aligned for the element type"
            ),
            ViewDataError::MissingBytes(num_missing_bytes) => write!(
                f,
                "missing {} bytes of data specified in header",
                num_missing_bytes
            ),
            ViewDataError::ExtraBytes(num_extra_bytes) => {
                write!(f, "file had {} extra bytes before EOF", num_extra_bytes)
            }
            ViewDataError::InvalidData(err) => write!(f, "invalid data for element type: {}", err),
        }
    }
}

/// An array element type that can be viewed (without copying) in an `.npy`
/// file.
pub trait ViewElement: Sized {
    /// Casts `bytes` into a slice of elements of length `len`.
    ///
    /// Returns `Err(_)` in at least the following cases:
    ///
    ///   * if the `type_desc` does not match `Self` with native endianness
    ///   * if the `bytes` slice is misaligned for elements of type `Self`
    ///   * if the `bytes` slice is too short for `len` elements
    ///   * if the `bytes` slice has extra bytes after `len` elements
    ///
    /// May panic if `len * size_of::<Self>()` overflows.
    fn bytes_as_slice<'a>(
        bytes: &'a [u8],
        type_desc: &PyValue,
        len: usize,
    ) -> Result<&'a [Self], ViewDataError>;
}

/// An array element type that can be mutably viewed (without copying) in an
/// `.npy` file.
pub trait ViewMutElement: Sized {
    /// Casts `bytes` into a mutable slice of elements of length `len`.
    ///
    /// Returns `Err(_)` in at least the following cases:
    ///
    ///   * if the `type_desc` does not match `Self` with native endianness
    ///   * if the `bytes` slice is misaligned for elements of type `Self`
    ///   * if the `bytes` slice is too short for `len` elements
    ///   * if the `bytes` slice has extra bytes after `len` elements
    ///
    /// May panic if `len * size_of::<Self>()` overflows.
    fn bytes_as_mut_slice<'a>(
        bytes: &'a mut [u8],
        type_desc: &PyValue,
        len: usize,
    ) -> Result<&'a mut [Self], ViewDataError>;
}

/// Returns `Ok(_)` iff a slice containing `bytes_len` bytes is the correct length to cast to
/// a slice with element type `T` and length `len`.
///
/// **Panics** if `len * size_of::<T>()` overflows.
fn check_bytes_len<T>(bytes_len: usize, len: usize) -> Result<(), ViewDataError> {
    let needed_bytes = len
        .checked_mul(mem::size_of::<T>())
        .expect("Required number of bytes should not overflow.");
    if bytes_len < needed_bytes {
        Err(ViewDataError::MissingBytes(needed_bytes - bytes_len))
    } else if bytes_len > needed_bytes {
        Err(ViewDataError::ExtraBytes(bytes_len - needed_bytes))
    } else {
        Ok(())
    }
}

/// Returns `Ok(_)` iff the slice of bytes is properly aligned to be cast to a
/// slice with element type `T`.
fn check_bytes_align<T>(bytes: &[u8]) -> Result<(), ViewDataError> {
    if bytes.as_ptr() as usize % mem::align_of::<T>() == 0 {
        Ok(())
    } else {
        Err(ViewDataError::Misaligned)
    }
}

/// Cast `&[u8]` to `&[T]`, where the resulting slice should have length `len`.
///
/// Returns `Err` if the length or alignment of `bytes` is incorrect.
///
/// # Safety
///
/// The caller must ensure that the cast is valid for the type `T`. For
/// example, this is not true for `bool` unless the caller has previously
/// checked that the byte slice contains only `0x00` and `0x01` values. (This
/// function checks only that the length and alignment are correct.)
unsafe fn bytes_as_slice<T>(bytes: &[u8], len: usize) -> Result<&[T], ViewDataError> {
    check_bytes_len::<T>(bytes.len(), len)?;
    check_bytes_align::<T>(bytes)?;
    Ok(slice::from_raw_parts(bytes.as_ptr().cast(), len))
}

/// Cast `&mut [u8]` to `&mut [T]`, where the resulting slice should have
/// length `len`.
///
/// Returns `Err` if the length or alignment of `bytes` is incorrect.
///
/// # Safety
///
/// The caller must ensure that the cast is valid for the type `T`. For
/// example, this is not true for `bool` unless the caller has previously
/// checked that the byte slice contains only `0x00` and `0x01` values. (This
/// function checks only that the length and alignment are correct.)
unsafe fn bytes_as_mut_slice<T>(bytes: &mut [u8], len: usize) -> Result<&mut [T], ViewDataError> {
    check_bytes_len::<T>(bytes.len(), len)?;
    check_bytes_align::<T>(bytes)?;
    Ok(slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), len))
}

/// Computes the length associated with the shape (i.e. the product of the axis
/// lengths), where the element type is `T`.
///
/// Returns `None` if the number of elements or the length in bytes would
/// overflow `isize`.
fn shape_length_checked<T>(shape: &IxDyn) -> Option<usize> {
    let len = shape.size_checked()?;
    if len > std::isize::MAX as usize {
        return None;
    }
    let bytes = len.checked_mul(mem::size_of::<T>())?;
    if bytes > std::isize::MAX as usize {
        return None;
    }
    Some(len)
}

#[cfg(test)]
mod test {
    use super::{
        ReadDataError, ReadableElement, ViewDataError, ViewElement, ViewMutElement, WritableElement,
    };
    use py_literal::Value as PyValue;
    use std::convert::TryInto;
    use std::io::Cursor;
    use std::mem;

    #[test]
    fn view_i32() {
        let elems: &[i32] = &[34234324, -980780878, 2849874];
        let mut buf: Vec<u8> = Vec::new();
        <i32>::write_slice(elems, &mut buf).unwrap();
        #[cfg(target_endian = "little")]
        let type_desc = PyValue::String(String::from("<i4"));
        #[cfg(target_endian = "big")]
        let type_desc = PyValue::String(String::from(">i4"));
        let out: &[i32] = <i32>::bytes_as_slice(&buf, &type_desc, elems.len()).unwrap();
        assert_eq!(out, elems);
    }

    #[test]
    fn view_i32_mut() {
        let elems: &[i32] = &[34234324, -980780878, 2849874];
        let mut buf: Vec<u8> = Vec::new();
        <i32>::write_slice(elems, &mut buf).unwrap();
        #[cfg(target_endian = "little")]
        let type_desc = PyValue::String(String::from("<i4"));
        #[cfg(target_endian = "big")]
        let type_desc = PyValue::String(String::from(">i4"));
        let out: &mut [i32] = <i32>::bytes_as_mut_slice(&mut buf, &type_desc, elems.len()).unwrap();
        assert_eq!(out, elems);
        out[2] += 1;
        let buf_last = i32::from_ne_bytes(buf[2 * mem::size_of::<i32>()..].try_into().unwrap());
        assert_eq!(buf_last, elems[2] + 1);
    }

    #[test]
    fn view_i32_non_native_endian() {
        const LEN: usize = 3;
        let buf: &[u8] = &[0; LEN * mem::size_of::<i32>()];
        #[cfg(target_endian = "little")]
        let type_desc = PyValue::String(String::from(">i4"));
        #[cfg(target_endian = "big")]
        let type_desc = PyValue::String(String::from("<i4"));
        let out = <i32>::bytes_as_slice(buf, &type_desc, LEN);
        assert!(matches!(out, Err(ViewDataError::NonNativeEndian)));
    }

    #[test]
    fn view_bool() {
        let data = &[0x00, 0x01, 0x00, 0x00, 0x01];
        let type_desc = PyValue::String(String::from("|b1"));
        let out = <bool>::bytes_as_slice(data, &type_desc, data.len()).unwrap();
        assert_eq!(out, vec![false, true, false, false, true]);
    }

    #[test]
    fn view_bool_bad_value() {
        let data = &[0x00, 0x01, 0x05, 0x00, 0x01];
        let type_desc = PyValue::String(String::from("|b1"));
        let out = <bool>::bytes_as_slice(data, &type_desc, data.len());
        assert!(matches!(out, Err(ViewDataError::InvalidData(_))));
    }

    #[test]
    fn view_bool_mut() {
        let data = &mut [0x00, 0x01, 0x00, 0x00, 0x01];
        let len = data.len();
        let type_desc = PyValue::String(String::from("|b1"));
        let out = <bool>::bytes_as_mut_slice(data, &type_desc, len).unwrap();
        out[0] = true;
        out[1] = false;
        assert_eq!(data, &[0x01, 0x00, 0x00, 0x00, 0x01]);
    }

    #[test]
    fn view_bool_mut_bad_value() {
        let data = &mut [0x00, 0x01, 0x05, 0x00, 0x01];
        let len = data.len();
        let type_desc = PyValue::String(String::from("|b1"));
        let out = <bool>::bytes_as_mut_slice(data, &type_desc, len);
        assert!(matches!(out, Err(ViewDataError::InvalidData(_))));
    }

    #[test]
    fn read_bool() {
        let data = &[0x00, 0x01, 0x00, 0x00, 0x01];
        let type_desc = PyValue::String(String::from("|b1"));
        let out = <bool>::read_to_end_exact_vec(Cursor::new(data), &type_desc, data.len()).unwrap();
        assert_eq!(out, vec![false, true, false, false, true]);
    }

    #[test]
    fn read_bool_bad_value() {
        let data = &[0x00, 0x01, 0x05, 0x00, 0x01];
        let type_desc = PyValue::String(String::from("|b1"));
        match <bool>::read_to_end_exact_vec(Cursor::new(data), &type_desc, data.len()) {
            Err(ReadDataError::ParseData(err)) => {
                assert_eq!(format!("{}", err), "error parsing value 0x05 as a bool");
            }
            _ => panic!(),
        }
    }
}
