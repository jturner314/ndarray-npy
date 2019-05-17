pub mod header;

use byteorder::{self, ByteOrder};
use ndarray::{Data, DataOwned, ShapeError};
use ndarray::prelude::*;
use py_literal::Value as PyValue;
use std::error::Error;
use std::io;
use std::mem;
use self::header::{FormatHeaderError, Header, HeaderReadError};

/// An array element type that can be written to an `.npy` or `.npz` file.
pub unsafe trait WritableElement: Sized {
    type Error: 'static + Error + Send + Sync;

    /// Returns a descriptor of the type that can be used in the header.
    fn type_descriptor() -> PyValue;

    /// Writes a single instance of `Self` to the writer.
    fn write<W: io::Write>(&self, writer: W) -> Result<(), Self::Error>;

    /// Writes a slice of `Self` to the writer.
    fn write_slice<W: io::Write>(slice: &[Self], writer: W) -> Result<(), Self::Error>;
}

quick_error! {
    /// An error writing a `.npy` file.
    #[derive(Debug)]
    pub enum WriteNpyError {
        /// An error caused by I/O.
        Io(err: io::Error) {
            description("I/O error")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
        /// An error formatting the header.
        FormatHeader(err: FormatHeaderError) {
            description("error formatting header")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
        /// An error issued by the element type when writing the data.
        WritableElement(err: Box<Error + Send + Sync>) {
            description("WritableElement error")
            display(x) -> ("{}: {}", x.description(), err)
            cause(&**err)
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
            }.to_bytes()?;
            writer.write_all(&header)?;
            A::write_slice(self.as_slice_memory_order().unwrap(), &mut writer)
                .map_err(|err| WriteNpyError::WritableElement(Box::new(err)))?;
            Ok(())
        };
        if self.is_standard_layout() {
            write_contiguous(writer, false)
        } else if self.view().reversed_axes().is_standard_layout() {
            write_contiguous(writer, true)
        } else {
            writer.write_all(&Header {
                type_descriptor: A::type_descriptor(),
                fortran_order: false,
                shape: self.shape().to_owned(),
            }.to_bytes()?)?;
            for elem in self.iter() {
                elem.write(&mut writer)
                    .map_err(|err| WriteNpyError::WritableElement(Box::new(err)))?;
            }
            Ok(())
        }
    }
}

/// An array element type that can be read from an `.npy` or `.npz` file.
pub trait ReadableElement: Sized {
    type Error: 'static + Error + Send + Sync;

    /// Checks `type_desc` and reads a `Vec` of length `len` from `reader`.
    fn read_vec<R: io::Read>(
        reader: R,
        type_desc: &PyValue,
        len: usize,
    ) -> Result<Vec<Self>, Self::Error>;
}

quick_error! {
    /// An error reading a `.npy` file.
    #[derive(Debug)]
    pub enum ReadNpyError {
        /// An error caused by reading the file header.
        Header(err: HeaderReadError) {
            description("error reading header")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
        /// An error issued by the element type when reading the data.
        ReadableElement(err: Box<Error + Send + Sync>) {
            description("ReadableElement error")
            display(x) -> ("{}: {}", x.description(), err)
            cause(&**err)
            from(ReadPrimitiveError)
        }
        /// Overflow while computing the length of the array from the shape
        /// described in the file header.
        LengthOverflow {
            description("overflow computing length from shape")
            display(x) -> ("{}", x.description())
        }
        /// Extra bytes are present between the end of the data and the end of
        /// the file.
        ExtraBytes {
            description("file had extra bytes before EOF")
            display(x) -> ("{}", x.description())
        }
        /// An error caused by incorrect array length or dimensionality.
        Shape(err: ShapeError) {
            description("data did not match shape in header")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
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
        let shape = if header.fortran_order {
            header.shape.f()
        } else {
            header.shape.into_shape()
        };
        // let len = shape.size_checked().ok_or(ReadNpyError::LengthOverFlow)?;
        let len = shape.size();
        let data = A::read_vec(&mut reader, &header.type_descriptor, len)
            .map_err(|err| ReadNpyError::ReadableElement(Box::new(err)))?;
        {
            let mut buf = [0];
            match reader.read_exact(&mut buf) {
                Err(ref err) if err.kind() == io::ErrorKind::UnexpectedEof => (),
                _ => return Err(ReadNpyError::ExtraBytes),
            }
        }
        Ok(ArrayBase::from_shape_vec(shape, data)?.into_dimensionality()?)
    }
}

macro_rules! impl_writable_primitive {
    ($elem:ty, $little_desc:expr, $big_desc:expr) => {
        unsafe impl WritableElement for $elem {
            type Error = io::Error;

            fn type_descriptor() -> PyValue {
                if cfg!(target_endian = "little") {
                    PyValue::String($little_desc.into())
                } else if cfg!(target_endian = "big") {
                    PyValue::String($big_desc.into())
                } else {
                    unreachable!()
                }
            }

            fn write<W: io::Write>(&self, mut writer: W) -> Result<(), Self::Error> {
                // Function to ensure lifetime of bytes slice is correct.
                fn cast(self_: &$elem) -> &[u8] {
                    unsafe {
                        std::slice::from_raw_parts(
                            self_ as *const $elem as *const u8,
                            mem::size_of::<$elem>(),
                        )
                    }
                }
                writer.write_all(cast(self))
            }

            fn write_slice<W: io::Write>(slice: &[Self], mut writer: W) -> Result<(), Self::Error> {
                // Function to ensure lifetime of bytes slice is correct.
                fn cast(slice: &[$elem]) -> &[u8] {
                    unsafe {
                        std::slice::from_raw_parts(
                            slice.as_ptr() as *const u8,
                            slice.len() * mem::size_of::<$elem>(),
                        )
                    }
                }
                writer.write_all(cast(slice))
            }
        }
    };
}

quick_error! {
    #[derive(Debug)]
    pub enum ReadPrimitiveError {
        BadDescriptor(desc: PyValue) {
            description("bad descriptor for this type")
            display(x) -> ("{}: {}", x.description(), desc)
        }
        BadValue {
            description("invalid value for this type")
        }
        Io(err: io::Error) {
            description("I/O error")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
    }
}

macro_rules! impl_readable_primitive {
    ($elem:ty, $zero:expr, $native_desc:expr, $other_desc:expr, $other_read_into:path) => {
        impl ReadableElement for $elem {
            type Error = ReadPrimitiveError;

            fn read_vec<R: io::Read>(mut reader: R, type_desc: &PyValue, len: usize)
                                     -> Result<Vec<Self>, Self::Error>
            {
                match *type_desc {
                    PyValue::String(ref s) if s == $native_desc => {
                        // Function to ensure lifetime of bytes slice is correct.
                        fn cast_slice(slice: &mut [$elem]) -> &mut [u8] {
                            unsafe {
                                std::slice::from_raw_parts_mut(
                                    slice.as_mut_ptr() as *mut u8,
                                    slice.len() * mem::size_of::<$elem>(),
                                )
                            }
                        }
                        let mut out = vec![$zero; len];
                        reader.read_exact(cast_slice(&mut out))?;
                        Ok(out)
                    }
                    PyValue::String(ref s) if s == $other_desc => {
                        let mut bytes = vec![0; len * mem::size_of::<$elem>()];
                        reader.read_exact(&mut bytes)?;
                        let mut out = vec![$zero; len];
                        $other_read_into(&bytes, &mut out);
                        Ok(out)
                    }
                    ref other => Err(ReadPrimitiveError::BadDescriptor(other.clone())),
                }
            }
        }
    };
}

macro_rules! impl_primitive_multibyte {
    ($elem:ty, $little_desc:expr, $big_desc:expr, $zero:expr, $read_into:ident) => {
        impl_writable_primitive!($elem, $little_desc, $big_desc);
        #[cfg(target_endian = "little")]
        impl_readable_primitive!(
            $elem, $zero, $little_desc, $big_desc, byteorder::BigEndian::$read_into
        );
        #[cfg(target_endian = "big")]
        impl_readable_primitive!(
            $elem, $zero, $big_desc, $little_desc, byteorder::LittleEndian::$read_into
        );
    };
}

impl ReadableElement for i8 {
    type Error = ReadPrimitiveError;

    fn read_vec<R: io::Read>(
        mut reader: R,
        type_desc: &PyValue,
        len: usize,
    ) -> Result<Vec<Self>, Self::Error> {
        match *type_desc {
            PyValue::String(ref s) if s == "|i1" || s == "i1" || s == "b" => {
                // Function to ensure lifetime of bytes slice is correct.
                fn cast_slice(slice: &mut [i8]) -> &mut [u8] {
                    unsafe {
                        std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u8, slice.len())
                    }
                }
                let mut out = vec![0; len];
                reader.read_exact(cast_slice(&mut out))?;
                Ok(out)
            }
            ref other => Err(ReadPrimitiveError::BadDescriptor(other.clone())),
        }
    }
}

impl ReadableElement for u8 {
    type Error = ReadPrimitiveError;

    fn read_vec<R: io::Read>(
        mut reader: R,
        type_desc: &PyValue,
        len: usize,
    ) -> Result<Vec<Self>, Self::Error> {
        match *type_desc {
            PyValue::String(ref s) if s == "|u1" || s == "u1" || s == "B" => {
                let mut out = vec![0; len];
                reader.read_exact(&mut out)?;
                Ok(out)
            }
            ref other => Err(ReadPrimitiveError::BadDescriptor(other.clone())),
        }
    }
}

impl_writable_primitive!(i8, "|i1", "|i1");
impl_writable_primitive!(u8, "|u1", "|u1");

impl_primitive_multibyte!(i16, "<i2", ">i2", 0, read_i16_into);
impl_primitive_multibyte!(i32, "<i4", ">i4", 0, read_i32_into);
impl_primitive_multibyte!(i64, "<i8", ">i8", 0, read_i64_into);

impl_primitive_multibyte!(u16, "<u2", ">u2", 0, read_u16_into);
impl_primitive_multibyte!(u32, "<u4", ">u4", 0, read_u32_into);
impl_primitive_multibyte!(u64, "<u8", ">u8", 0, read_u64_into);

impl_primitive_multibyte!(f32, "<f4", ">f4", 0., read_f32_into);
impl_primitive_multibyte!(f64, "<f8", ">f8", 0., read_f64_into);

impl ReadableElement for bool {
    type Error = ReadPrimitiveError;

    fn read_vec<R: io::Read>(
        mut reader: R,
        type_desc: &PyValue,
        len: usize,
    ) -> Result<Vec<Self>, Self::Error> {
        match *type_desc {
            PyValue::String(ref s) if s == "|b1" => {
                // Read the data.
                let mut bytes: Vec<u8> = vec![0; len];
                reader.read_exact(&mut bytes)?;

                // Check that all the data is valid, because creating a `bool`
                // with an invalid value is undefined behavior. Rust guarantees
                // that `false` is represented as `0x00` and `true` is
                // represented as `0x01`.
                for &byte in &bytes {
                    if byte > 1 {
                        return Err(ReadPrimitiveError::BadValue);
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
            ref other => Err(ReadPrimitiveError::BadDescriptor(other.clone())),
        }
    }
}

// Rust guarantees that `bool` is one byte, the bitwise representation of
// `false` is `0x00`, and the bitwise representation of `true` is `0x01`, so we
// can just cast the data in-place.
impl_writable_primitive!(bool, "|b1", "|b1");

#[cfg(test)]
mod test {
    use super::{ReadableElement, ReadPrimitiveError};
    use py_literal::Value as PyValue;
    use std::io::Cursor;

    #[test]
    fn read_bool() {
        let data = &[0x00, 0x01, 0x00, 0x00, 0x01];
        let type_desc = PyValue::String(String::from("|b1"));
        let out = <bool>::read_vec(Cursor::new(data), &type_desc, data.len()).unwrap();
        assert_eq!(out, vec![false, true, false, false, true]);
    }

    #[test]
    fn read_bool_bad_value() {
        let data = &[0x00, 0x01, 0x05, 0x00, 0x01];
        let type_desc = PyValue::String(String::from("|b1"));
        match <bool>::read_vec(Cursor::new(data), &type_desc, data.len()) {
            Err(ReadPrimitiveError::BadValue) => {}
            _ => panic!(),
        }
    }
}
