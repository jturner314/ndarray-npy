pub mod header;

use self::header::{FormatHeaderError, Header, HeaderParseError, HeaderReadError};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use ndarray::prelude::*;
use ndarray::{Data, DataOwned, IntoDimension};
use py_literal::Value as PyValue;
use std::error::Error;
use std::fmt;
use std::io;
use std::mem;

/// An error writing array elements.
#[derive(Debug)]
pub enum WriteElementsError {
    /// An error caused by I/O.
    Io(io::Error),
    /// An error formatting the elements.
    FormatElements(Box<dyn Error + Send + Sync + 'static>),
}

impl Error for WriteElementsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            WriteElementsError::Io(err) => Some(err),
            WriteElementsError::FormatElements(err) => Some(&**err),
        }
    }
}

impl fmt::Display for WriteElementsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            WriteElementsError::Io(err) => write!(f, "I/O error: {}", err),
            WriteElementsError::FormatElements(err) => {
                write!(f, "error formatting data elements: {}", err)
            }
        }
    }
}

impl From<io::Error> for WriteElementsError {
    fn from(err: io::Error) -> WriteElementsError {
        WriteElementsError::Io(err)
    }
}

/// An array element type that can be written to an `.npy` or `.npz` file.
pub unsafe trait WritableElement: Sized {
    /// Returns a descriptor of the type that can be used in the header.
    fn type_descriptor() -> PyValue;

    /// Writes a single instance of `Self` to the writer.
    fn write<W: io::Write>(&self, writer: W) -> Result<(), WriteElementsError>;

    /// Writes a slice of `Self` to the writer.
    fn write_slice<W: io::Write>(slice: &[Self], writer: W) -> Result<(), WriteElementsError>;
}

/// An error writing a `.npy` file.
#[derive(Debug)]
pub enum WriteNpyError {
    /// An error caused by I/O.
    Io(io::Error),
    /// An error formatting the header.
    FormatHeader(FormatHeaderError),
    /// An error formatting the data elements.
    FormatElements(Box<dyn Error + Send + Sync + 'static>),
}

impl Error for WriteNpyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            WriteNpyError::Io(err) => Some(err),
            WriteNpyError::FormatHeader(err) => Some(err),
            WriteNpyError::FormatElements(err) => Some(&**err),
        }
    }
}

impl fmt::Display for WriteNpyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            WriteNpyError::Io(err) => write!(f, "I/O error: {}", err),
            WriteNpyError::FormatHeader(err) => write!(f, "error formatting header: {}", err),
            WriteNpyError::FormatElements(err) => {
                write!(f, "error formatting data elements: {}", err)
            }
        }
    }
}

impl From<io::Error> for WriteNpyError {
    fn from(err: io::Error) -> WriteNpyError {
        WriteNpyError::Io(err)
    }
}

impl From<FormatHeaderError> for WriteNpyError {
    fn from(err: FormatHeaderError) -> WriteNpyError {
        WriteNpyError::FormatHeader(err)
    }
}

impl From<WriteElementsError> for WriteNpyError {
    fn from(err: WriteElementsError) -> WriteNpyError {
        match err {
            WriteElementsError::Io(err) => WriteNpyError::Io(err),
            WriteElementsError::FormatElements(err) => WriteNpyError::FormatElements(err),
        }
    }
}

/// Extension trait for writing `ArrayBase` to `.npy` files.
///
/// # Example
///
/// ```
/// #[macro_use]
/// extern crate ndarray;
/// extern crate ndarray_npy;
///
/// use ndarray::prelude::*;
/// use ndarray_npy::WriteNpyExt;
/// use std::fs::File;
/// # use ndarray_npy::WriteNpyError;
///
/// # fn write_example() -> Result<(), WriteNpyError> {
/// let arr: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];
/// let writer = File::create("array.npy")?;
/// arr.write_npy(writer)?;
/// # Ok(())
/// # }
/// # fn main () {}
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
            let header = Header {
                type_descriptor: A::type_descriptor(),
                fortran_order,
                shape: self.shape().to_owned(),
            }
            .to_bytes()?;
            writer.write_all(&header)?;
            A::write_slice(self.as_slice_memory_order().unwrap(), &mut writer)?;
            Ok(())
        };
        if self.is_standard_layout() {
            write_contiguous(writer, false)
        } else if self.view().reversed_axes().is_standard_layout() {
            write_contiguous(writer, true)
        } else {
            writer.write_all(
                &Header {
                    type_descriptor: A::type_descriptor(),
                    fortran_order: false,
                    shape: self.shape().to_owned(),
                }
                .to_bytes()?,
            )?;
            for elem in self.iter() {
                elem.write(&mut writer)?;
            }
            Ok(())
        }
    }
}

/// An error reading array elements.
#[derive(Debug)]
pub enum ReadElementsError {
    /// An error caused by I/O.
    Io(io::Error),
    /// The type descriptor does not match the element type.
    WrongDescriptor(PyValue),
    /// The file does not contain all the data described in the header.
    MissingData,
    /// Extra bytes are present between the end of the data and the end of the
    /// file.
    ExtraBytes(usize),
    /// An error parsing the elements data.
    ParseElements(Box<dyn Error + Send + Sync + 'static>),
}

impl Error for ReadElementsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ReadElementsError::Io(err) => Some(err),
            ReadElementsError::WrongDescriptor(_) => None,
            ReadElementsError::MissingData => None,
            ReadElementsError::ExtraBytes(_) => None,
            ReadElementsError::ParseElements(err) => Some(&**err),
        }
    }
}

impl fmt::Display for ReadElementsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ReadElementsError::Io(err) => write!(f, "I/O error: {}", err),
            ReadElementsError::WrongDescriptor(desc) => {
                write!(f, "incorrect descriptor ({}) for this type", desc)
            }
            ReadElementsError::MissingData => write!(f, "reached EOF before reading all data"),
            ReadElementsError::ExtraBytes(num_extra_bytes) => {
                write!(f, "file had {} extra bytes before EOF", num_extra_bytes)
            }
            ReadElementsError::ParseElements(err) => {
                write!(f, "error parsing data elements: {}", err)
            }
        }
    }
}

impl From<io::Error> for ReadElementsError {
    /// Performs the conversion.
    ///
    /// If the error kind is `UnexpectedEof`, the `MissingData` variant is
    /// returned. Otherwise, the `Io` variant is returned.
    fn from(err: io::Error) -> ReadElementsError {
        if err.kind() == io::ErrorKind::UnexpectedEof {
            ReadElementsError::MissingData
        } else {
            ReadElementsError::Io(err)
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
    ) -> Result<Vec<Self>, ReadElementsError>;
}

/// An error reading a `.npy` file.
#[derive(Debug)]
pub enum ReadNpyError {
    /// An error caused by I/O.
    Io(io::Error),
    /// An error parsing the file header.
    ParseHeader(HeaderParseError),
    /// An error parsing the elements.
    ParseElements(Box<dyn Error + Send + Sync + 'static>),
    /// Overflow while computing the length of the array from the shape
    /// described in the file header.
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
            ReadNpyError::ParseElements(err) => Some(&**err),
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
            ReadNpyError::ParseElements(err) => write!(f, "error parsing data elements: {}", err),
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

impl From<HeaderReadError> for ReadNpyError {
    fn from(err: HeaderReadError) -> ReadNpyError {
        match err {
            HeaderReadError::Io(err) => ReadNpyError::Io(err),
            HeaderReadError::Parse(err) => ReadNpyError::ParseHeader(err),
        }
    }
}

impl From<HeaderParseError> for ReadNpyError {
    fn from(err: HeaderParseError) -> ReadNpyError {
        ReadNpyError::ParseHeader(err)
    }
}

impl From<ReadElementsError> for ReadNpyError {
    fn from(err: ReadElementsError) -> ReadNpyError {
        match err {
            ReadElementsError::Io(err) => ReadNpyError::Io(err),
            ReadElementsError::WrongDescriptor(desc) => ReadNpyError::WrongDescriptor(desc),
            ReadElementsError::MissingData => ReadNpyError::MissingData,
            ReadElementsError::ExtraBytes(nbytes) => ReadNpyError::ExtraBytes(nbytes),
            ReadElementsError::ParseElements(err) => ReadNpyError::ParseElements(err),
        }
    }
}

/// Extension trait for reading `Array` from `.npy` files.
///
/// # Example
///
/// ```
/// #[macro_use]
/// extern crate ndarray;
/// extern crate ndarray_npy;
///
/// use ndarray::prelude::*;
/// use ndarray_npy::ReadNpyExt;
/// use std::fs::File;
///
/// # fn read_example() -> Result<(), Box<std::error::Error>> {
/// let reader = File::open("array.npy")?;
/// let arr = Array2::<i32>::read_npy(reader)?;
/// # println!("arr = {}", arr);
/// # Ok(())
/// # }
/// # fn main () {}
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
        let len = match shape.size_checked() {
            Some(len) if len <= std::isize::MAX as usize => len,
            _ => return Err(ReadNpyError::LengthOverflow),
        };
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

            fn write<W: io::Write>(&self, mut writer: W) -> Result<(), WriteElementsError> {
                // Function to ensure lifetime of bytes slice is correct.
                fn cast(self_: &$elem) -> &[u8] {
                    unsafe {
                        std::slice::from_raw_parts(
                            self_ as *const $elem as *const u8,
                            mem::size_of::<$elem>(),
                        )
                    }
                }
                writer.write_all(cast(self))?;
                Ok(())
            }

            fn write_slice<W: io::Write>(
                slice: &[Self],
                mut writer: W,
            ) -> Result<(), WriteElementsError> {
                // Function to ensure lifetime of bytes slice is correct.
                fn cast(slice: &[$elem]) -> &[u8] {
                    unsafe {
                        std::slice::from_raw_parts(
                            slice.as_ptr() as *const u8,
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
pub fn check_for_extra_bytes<R: io::Read>(reader: &mut R) -> Result<(), ReadElementsError> {
    let num_extra_bytes = reader.read_to_end(&mut Vec::new())?;
    if num_extra_bytes == 0 {
        Ok(())
    } else {
        Err(ReadElementsError::ExtraBytes(num_extra_bytes))
    }
}

macro_rules! impl_readable_primitive_one_byte {
    ($elem:ty, [$($desc:expr),*], $zero:expr, $read_into:ident) => {
        impl ReadableElement for $elem {
            fn read_to_end_exact_vec<R: io::Read>(
                mut reader: R,
                type_desc: &PyValue,
                len: usize,
            ) -> Result<Vec<Self>, ReadElementsError> {
                match *type_desc {
                    PyValue::String(ref s) if $(s == $desc)||* => {
                        let mut out = vec![$zero; len];
                        reader.$read_into(&mut out)?;
                        check_for_extra_bytes(&mut reader)?;
                        Ok(out)
                    }
                    ref other => Err(ReadElementsError::WrongDescriptor(other.clone())),
                }
            }
        }
    };
}

macro_rules! impl_primitive_one_byte {
    ($elem:ty, $write_desc:expr, [$($read_desc:expr),*], $zero:expr, $read_into:ident) => {
        impl_writable_primitive!($elem, $write_desc, $write_desc);
        impl_readable_primitive_one_byte!($elem, [$($read_desc),*], $zero, $read_into);
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
            ) -> Result<Vec<Self>, ReadElementsError> {
                let mut out = vec![$zero; len];
                match *type_desc {
                    PyValue::String(ref s) if s == $little_desc => {
                        reader.$read_into::<LittleEndian>(&mut out)?;
                    }
                    PyValue::String(ref s) if s == $big_desc => {
                        reader.$read_into::<BigEndian>(&mut out)?;
                    }
                    ref other => {
                        return Err(ReadElementsError::WrongDescriptor(other.clone()));
                    }
                }
                check_for_extra_bytes(&mut reader)?;
                Ok(out)
            }
        }
    };
}

macro_rules! impl_primitive_multi_byte {
    ($elem:ty, $little_desc:expr, $big_desc:expr, $zero:expr, $read_into:ident) => {
        impl_writable_primitive!($elem, $little_desc, $big_desc);
        impl_readable_primitive_multi_byte!($elem, $little_desc, $big_desc, $zero, $read_into);
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

impl From<ParseBoolError> for ReadElementsError {
    fn from(err: ParseBoolError) -> ReadElementsError {
        ReadElementsError::ParseElements(Box::new(err))
    }
}

impl ReadableElement for bool {
    fn read_to_end_exact_vec<R: io::Read>(
        mut reader: R,
        type_desc: &PyValue,
        len: usize,
    ) -> Result<Vec<Self>, ReadElementsError> {
        match *type_desc {
            PyValue::String(ref s) if s == "|b1" => {
                // Read the data.
                let mut bytes: Vec<u8> = vec![0; len];
                reader.read_exact(&mut bytes)?;
                check_for_extra_bytes(&mut reader)?;

                // Check that all the data is valid, because creating a `bool`
                // with an invalid value is undefined behavior. Rust guarantees
                // that `false` is represented as `0x00` and `true` is
                // represented as `0x01`.
                for &byte in &bytes {
                    if byte > 1 {
                        return Err(ReadElementsError::from(ParseBoolError { bad_value: byte }));
                    }
                }

                // Cast the `Vec<u8>` to `Vec<bool>`.
                {
                    let ptr = bytes.as_mut_ptr();
                    let len = bytes.len();
                    let cap = bytes.capacity();
                    mem::forget(bytes);
                    // This is safe because:
                    //
                    // * All elements are valid `bool`s. (See the loop above.)
                    //
                    // * `ptr` was originally allocated by `Vec`.
                    //
                    // * `bool` has the same size and alignment as `u8`.
                    //
                    // * `len` and `cap` are copied directly from the
                    //   `Vec<u8>`, so `len <= cap` and `cap` is the capacity
                    //   `ptr` was allocated with.
                    Ok(unsafe { Vec::from_raw_parts(ptr as *mut bool, len, cap) })
                }
            }
            ref other => Err(ReadElementsError::WrongDescriptor(other.clone())),
        }
    }
}

// Rust guarantees that `bool` is one byte, the bitwise representation of
// `false` is `0x00`, and the bitwise representation of `true` is `0x01`, so we
// can just cast the data in-place.
impl_writable_primitive!(bool, "|b1", "|b1");

#[cfg(test)]
mod test {
    use super::{ReadElementsError, ReadableElement};
    use py_literal::Value as PyValue;
    use std::io::Cursor;

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
            Err(ReadElementsError::ParseElements(err)) => {
                assert_eq!(format!("{}", err), "error parsing value 0x05 as a bool");
            }
            _ => panic!(),
        }
    }
}
