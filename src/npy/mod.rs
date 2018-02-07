pub mod header;

use byteorder::{BigEndian, ByteOrder, LittleEndian};
use ndarray::{Data, DataOwned, ShapeError};
use ndarray::prelude::*;
use std::error::Error;
use std::io;
use self::header::{Header, HeaderParseError, PyExpr};

/// An array element type that can be written to an `.npy` or `.npz` file.
pub unsafe trait WritableElement: Sized {
    /// A descriptor of the type that can be used in the header.
    /// This must match the representation of the type in memory.
    fn type_descriptor() -> PyExpr;

    fn as_bytes(&self) -> &[u8];

    fn slice_as_bytes(slice: &[Self]) -> &[u8];
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
///
/// # fn write_example() -> std::io::Result<()> {
/// let arr: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];
/// let writer = File::create("array.npy")?;
/// arr.write_npy(writer)?;
/// # Ok(())
/// # }
/// # fn main () {}
/// ```
pub trait WriteNpyExt {
    /// Writes the array to `writer` in [`.npy`
    /// format](https://docs.scipy.org/doc/numpy/neps/npy-format.html).
    fn write_npy<W: io::Write>(&self, writer: W) -> io::Result<()>;
}

impl<A, S, D> WriteNpyExt for ArrayBase<S, D>
where
    A: WritableElement,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn write_npy<'a, W: io::Write>(&'a self, mut writer: W) -> io::Result<()> {
        let write_contiguous = |mut writer: W, fortran_order: bool| {
            let header = Header {
                type_descriptor: A::type_descriptor(),
                fortran_order,
                shape: self.shape().to_owned(),
            }.to_bytes();
            writer.write_all(&header)?;
            writer.write_all(A::slice_as_bytes(self.as_slice_memory_order().unwrap()))?;
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
            }.to_bytes())?;
            for elem in self.iter() {
                writer.write_all(A::as_bytes(elem))?;
            }
            Ok(())
        }
    }
}

/// An array element type that can be read from an `.npy` or `.npz` file.
pub trait ReadableElement: Sized {
    type Error: 'static + Error;

    fn from_bytes_owned(type_desc: &PyExpr, bytes: Vec<u8>) -> Result<Vec<Self>, Self::Error>;
}

quick_error! {
    /// An error reading a `.npy` file.
    #[derive(Debug)]
    pub enum ReadNpyError {
        Io(err: io::Error) {
            description("I/O error")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
        HeaderParse(err: HeaderParseError) {
            description("error parsing header")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
        ReadableElement(err: Box<Error>) {
            description("ReadableElement error")
            display(x) -> ("{}: {}", x.description(), err)
            cause(&**err)
            from(ReadPrimitiveError)
        }
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
/// # use ndarray_npy::ReadNpyError;
///
/// # fn read_example() -> Result<(), ReadNpyError> {
/// let reader = File::open("array.npy")?;
/// let arr = Array2::<i32>::read_npy(reader)?;
/// # println!("arr = {}", arr);
/// # Ok(())
/// # }
/// # fn main () {}
/// ```
pub trait ReadNpyExt: Sized {
    /// Reads the array from `reader` in [`.npy`
    /// format](https://docs.scipy.org/doc/numpy/neps/npy-format.html).
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
        let mut buf =
            Vec::with_capacity(header.shape.iter().product::<usize>() * ::std::mem::size_of::<A>());
        reader.read_to_end(&mut buf)?;
        let data = A::from_bytes_owned(&header.type_descriptor, buf)
            .map_err(|err| ReadNpyError::ReadableElement(Box::new(err)))?;
        let shape = if header.fortran_order {
            header.shape.f()
        } else {
            header.shape.into_shape()
        };
        Ok(ArrayBase::from_shape_vec(shape, data)?.into_dimensionality()?)
    }
}

macro_rules! impl_writable_primitive {
    ($elem:ty, $little_desc:expr, $big_desc:expr) => {
        unsafe impl WritableElement for $elem {
            fn type_descriptor() -> PyExpr {
                if cfg!(target_endian = "little") {
                    $little_desc.into()
                } else if cfg!(target_endian = "big") {
                    $big_desc.into()
                } else {
                    unreachable!()
                }
            }

            fn as_bytes(&self) -> &[u8] {
                unsafe {
                    ::std::slice::from_raw_parts(
                        self as *const Self as *const u8,
                        ::std::mem::size_of::<Self>(),
                    )
                }
            }

            fn slice_as_bytes(slice: &[Self]) -> &[u8] {
                unsafe {
                    ::std::slice::from_raw_parts(
                        slice.as_ptr() as *const u8,
                        slice.len() * ::std::mem::size_of::<Self>(),
                    )
                }
            }
        }
    };
}

quick_error! {
    #[derive(Debug)]
    pub enum ReadPrimitiveError {
        BadDescriptor(desc: PyExpr) {
            description("bad descriptor for this type")
            display(x) -> ("{}: {}", x.description(), desc)
        }
    }
}

macro_rules! impl_primitive_multibyte {
    ($elem:ty, $little_desc:expr, $big_desc:expr, $zero:expr, $read_into:ident) => {
        impl_writable_primitive!($elem, $little_desc, $big_desc);

        impl ReadableElement for $elem {
            type Error = ReadPrimitiveError;

            /// Unlike with `i8` and `u8`, this implementation cannot cast the
            /// data in-place because `u8` might not have the same alignment as
            /// `Self`.
            fn from_bytes_owned(type_desc: &PyExpr, bytes: Vec<u8>)
                                -> Result<Vec<Self>, Self::Error>
            {
                match *type_desc {
                    PyExpr::String(ref s) if s == $little_desc => {
                        let mut out = vec![$zero; bytes.len() / ::std::mem::size_of::<Self>()];
                        LittleEndian::$read_into(&bytes, &mut out);
                        Ok(out)
                    }
                    PyExpr::String(ref s) if s == $big_desc => {
                        let mut out = vec![$zero; bytes.len() / ::std::mem::size_of::<Self>()];
                        BigEndian::$read_into(&bytes, &mut out);
                        Ok(out)
                    }
                    ref other => Err(ReadPrimitiveError::BadDescriptor(other.clone())),
                }
            }
        }
    };
}

impl ReadableElement for i8 {
    type Error = ReadPrimitiveError;

    fn from_bytes_owned(type_desc: &PyExpr, mut bytes: Vec<u8>) -> Result<Vec<Self>, Self::Error> {
        match *type_desc {
            PyExpr::String(ref s) if s == "|i1" => {
                let ptr = bytes.as_mut_ptr() as *mut Self;
                let len = bytes.len();
                let capacity = bytes.capacity();
                ::std::mem::forget(bytes);
                Ok(unsafe { Vec::from_raw_parts(ptr, len, capacity) })
            }
            ref other => Err(ReadPrimitiveError::BadDescriptor(other.clone())),
        }
    }
}

impl ReadableElement for u8 {
    type Error = ReadPrimitiveError;

    fn from_bytes_owned(type_desc: &PyExpr, bytes: Vec<u8>) -> Result<Vec<Self>, Self::Error> {
        match *type_desc {
            PyExpr::String(ref s) if s == "|u1" => Ok(bytes),
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

impl_primitive_multibyte!(f32, "<f4", ">f4", 0., read_f32_into_unchecked);
impl_primitive_multibyte!(f64, "<f8", ">f8", 0., read_f64_into_unchecked);
