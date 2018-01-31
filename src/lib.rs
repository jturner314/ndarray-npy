//! This crate provides an extension method for [`ndarray`]'s `ArrayBase` type to
//! write an array in [`.npy` format].
//!
//! [`ndarray`]: https://github.com/bluss/rust-ndarray
//! [`.npy` format]: https://docs.scipy.org/doc/numpy/neps/npy-format.html
//!
//! # Example
//!
//! ```ignore
//! #[macro_use]
//! extern crate ndarray;
//! extern crate ndarray_npy;
//!
//! use ndarray_npy::NpyExt;
//! use std::fs::File;
//!
//! # fn example() -> std::io::Result<()> {
//! let arr = array![[1, 2, 3], [4, 5, 6]];
//! let mut file = File::create("array.npy")?;
//! arr.write_npy(&mut file)?;
//! # Ok(())
//! # }
//! # fn main () {}
//! ```

extern crate byteorder;
extern crate ndarray;

use byteorder::{ByteOrder, LittleEndian};
use ndarray::Data;
use ndarray::prelude::*;
use std::io;

/// Trait that must be implemented by array elements.
pub unsafe trait Element: Sized {
    /// Size of element in bytes. This must be correct for safety.
    const SIZE: usize;

    /// Returns a descriptor of the type that can be used in the header.
    /// This must match the representation of the type in memory.
    fn type_descriptor() -> String;
}

unsafe impl Element for i8 {
    const SIZE: usize = 1;
    fn type_descriptor() -> String {
        String::from("'|i1'")
    }
}

unsafe impl Element for u8 {
    const SIZE: usize = 1;
    fn type_descriptor() -> String {
        String::from("'|u1'")
    }
}

macro_rules! impl_element {
    ($elem:ty, $size:expr, $big_desc:expr, $little_desc:expr) => {
        unsafe impl Element for $elem {
            const SIZE: usize = $size;
            fn type_descriptor() -> String {
                if cfg!(target_endian = "big") {
                    $big_desc
                } else if cfg!(target_endian = "little") {
                    $little_desc
                } else {
                    unreachable!()
                }
            }
        }
    }
}

impl_element!(i16, 2, String::from("'>i2'"), String::from("'<i2'"));
impl_element!(i32, 4, String::from("'>i4'"), String::from("'<i4'"));
impl_element!(i64, 8, String::from("'>i8'"), String::from("'<i8'"));

impl_element!(u16, 2, String::from("'>u2'"), String::from("'<u2'"));
impl_element!(u32, 4, String::from("'>u4'"), String::from("'<u4'"));
impl_element!(u64, 8, String::from("'>u8'"), String::from("'<u8'"));

impl_element!(f32, 4, String::from("'>f4'"), String::from("'<f4'"));
impl_element!(f64, 8, String::from("'>f8'"), String::from("'<f8'"));

/// Magic string to indicate npy format.
const MAGIC_STRING: &[u8] = b"\x93NUMPY";

#[allow(non_camel_case_types)]
enum Version {
    V1_0,
    V2_0,
}

impl Version {
    /// Number of bytes taken up by version number (1 byte for major version, 1
    /// byte for minor version).
    const VERSION_NUM_BYTES: usize = 2;

    /// Major version number.
    fn major_version(&self) -> u8 {
        match *self {
            Version::V1_0 => 1,
            Version::V2_0 => 2,
        }
    }

    /// Major version number.
    fn minor_version(&self) -> u8 {
        match *self {
            Version::V1_0 => 0,
            Version::V2_0 => 0,
        }
    }

    /// Number of bytes in representation of header length.
    fn header_len_num_bytes(&self) -> usize {
        match *self {
            Version::V1_0 => 2,
            Version::V2_0 => 4,
        }
    }

    /// Format header length as bytes for writing to file.
    fn format_header_len(&self, header_len: usize) -> Vec<u8> {
        let mut out = vec![0; self.header_len_num_bytes()];
        match *self {
            Version::V1_0 => {
                assert!(header_len <= std::u16::MAX as usize);
                LittleEndian::write_u16(&mut out, header_len as u16);
            }
            Version::V2_0 => {
                assert!(header_len <= std::u32::MAX as usize);
                LittleEndian::write_u32(&mut out, header_len as u32);
            }
        }
        out
    }
}

/// Formats file header.
fn format_header<A: Element>(fortran_order: bool, shape: &[usize]) -> Vec<u8> {
    // Metadata describing array's format as ASCII string.
    let mut arr_format = format!(
        "{{'descr': {}, 'fortran_order': {}, 'shape': (",
        A::type_descriptor(),
        if fortran_order { "True" } else { "False" }
    );
    for (i, axis_len) in shape.iter().enumerate() {
        if i != 0 {
            arr_format.push_str(", ");
        }
        arr_format.push_str(&format!("{}", axis_len));
    }
    arr_format.push_str("), }");

    // Length of a '\n' char in bytes.
    const NEWLINE_LEN: usize = 1;

    // Determine appropriate version based on minimum number of bytes needed to
    // represent header length (including final newline).
    let version = if arr_format.len() + NEWLINE_LEN > std::u16::MAX as usize {
        Version::V2_0
    } else {
        Version::V1_0
    };
    let prefix_len =
        MAGIC_STRING.len() + Version::VERSION_NUM_BYTES + version.header_len_num_bytes();

    // Add padding spaces to make total header length divisible by 16.
    for _ in 0..(16 - (prefix_len + arr_format.len() + NEWLINE_LEN) % 16) {
        arr_format.push(' ');
    }
    // Add final newline.
    arr_format.push('\n');

    // Determine length of header.
    let header_len = arr_format.len();

    let mut out = Vec::with_capacity(prefix_len + header_len);
    out.extend_from_slice(MAGIC_STRING);
    out.push(version.major_version());
    out.push(version.minor_version());
    out.extend_from_slice(&version.format_header_len(header_len));
    out.extend_from_slice(arr_format.as_bytes());

    // Verify that length of header is divisible by 16.
    debug_assert_eq!(out.len() % 16, 0);

    out
}

/// Extension trait for `ArrayBase` that adds `.npy`-related methods.
pub trait NpyExt {
    /// Writes the array to `writer` in [`.npy`
    /// format](https://docs.scipy.org/doc/numpy/neps/npy-format.html).
    fn write_npy<W: io::Write>(&self, writer: W) -> io::Result<()>;
}

impl<A, S, D> NpyExt for ArrayBase<S, D>
where
    A: Clone + Element,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn write_npy<'a, W: io::Write>(&'a self, mut writer: W) -> io::Result<()> {
        let mut write_contiguous = |view: ArrayView<A, D>, fortran_order| {
            let header = format_header::<A>(fortran_order, view.shape());
            writer.write_all(&header)?;
            writer.write_all(unsafe {
                std::slice::from_raw_parts::<'a, u8>(
                    view.as_ptr() as *const u8,
                    view.len() * A::SIZE,
                )
            })?;
            Ok(())
        };
        if self.is_standard_layout() {
            write_contiguous(self.view(), false)
        } else if self.view().reversed_axes().is_standard_layout() {
            write_contiguous(self.view(), true)
        } else {
            let tmp = self.to_owned();
            debug_assert!(tmp.is_standard_layout());
            write_contiguous(tmp.view(), false)
        }
    }
}
