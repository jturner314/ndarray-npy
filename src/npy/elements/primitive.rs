//! Trait implementations for primitive element types.

use crate::{
    ReadDataError, ReadableElement, ViewDataError, ViewElement, ViewMutElement, WritableElement,
    WriteDataError,
};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use py_literal::Value as PyValue;
use std::error::Error;
use std::fmt;
use std::io;
use std::mem;
use std::slice;

/// Returns `Ok(_)` iff the `reader` had no more bytes on entry to this
/// function.
///
/// **Warning** This will consume the remainder of the reader.
fn check_for_extra_bytes<R: io::Read>(reader: &mut R) -> Result<(), ReadDataError> {
    let num_extra_bytes = reader.read_to_end(&mut Vec::new())?;
    if num_extra_bytes == 0 {
        Ok(())
    } else {
        Err(ReadDataError::ExtraBytes(num_extra_bytes))
    }
}

/// Returns `Ok(_)` iff a slice containing `bytes_len` bytes is the correct length to cast to
/// a slice with element type `T` and length `len`.
///
/// **Panics** if `len * size_of::<T>()` overflows.
fn check_bytes_len<T>(bytes_len: usize, len: usize) -> Result<(), ViewDataError> {
    use std::cmp::Ordering;
    let needed_bytes = len
        .checked_mul(mem::size_of::<T>())
        .expect("Required number of bytes should not overflow.");
    match bytes_len.cmp(&needed_bytes) {
        Ordering::Less => Err(ViewDataError::MissingBytes(needed_bytes - bytes_len)),
        Ordering::Equal => Ok(()),
        Ordering::Greater => Err(ViewDataError::ExtraBytes(bytes_len - needed_bytes)),
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
                        #[allow(clippy::unknown_clippy_lints, clippy::size_of_in_element_count)]
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
                        #[allow(clippy::unknown_clippy_lints, clippy::size_of_in_element_count)]
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
