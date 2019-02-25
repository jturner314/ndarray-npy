use byteorder::{ByteOrder, LittleEndian};
use num_traits::ToPrimitive;
use py_literal::{ParseError as PyValueParseError, FormatError as PyValueFormatError, Value as PyValue};
use std::error::Error;
use std::fmt;
use std::io;
use super::ReadNpyError;

/// Magic string to indicate npy format.
const MAGIC_STRING: &[u8] = b"\x93NUMPY";

quick_error! {
    #[derive(Debug)]
    pub enum HeaderParseError {
        MagicString {
            description("start does not match magic string")
            display(x) -> ("{}", x.description())
        }
        Version(major: u8, minor: u8) {
            description("unknown version number")
            display(x) -> ("{}: {}.{}", x.description(), major, minor)
        }
        NonAscii {
            description("non-ascii in array format string")
            display(x) -> ("{}", x.description())
            from(std::str::Utf8Error)
        }
        UnknownKey(key: PyValue) {
            description("unknown key")
            display(x) -> ("{}: {}", x.description(), key)
        }
        MissingKey(key: String) {
            description("missing key")
            display(x) -> ("{}: {}", x.description(), key)
        }
        IllegalValue(key: String, value: PyValue) {
            description("illegal value for key")
            display(x) -> ("{} {}: {}", x.description(), key, value)
        }
        DictParse(err: PyValueParseError){
            description("error parsing metadata dict")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
        MetaNotDict(value: PyValue) {
            description("metadata is not a dict")
            display(x) -> ("{}: {}", x.description(), value)
        }
        MissingNewline {
            description("newline missing at end of header")
            display(x) -> ("{}", x.description())
        }
    }
}

quick_error! {
    #[derive(Debug)]
    pub enum HeaderReadError {
        Io(err: io::Error) {
            description("I/O error")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
        Parse(err: HeaderParseError) {
            description("error parsing header")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
    }
}

#[allow(non_camel_case_types)]
enum Version {
    V1_0,
    V2_0,
}

impl Version {
    /// Number of bytes taken up by version number (1 byte for major version, 1
    /// byte for minor version).
    const VERSION_NUM_BYTES: usize = 2;

    fn from_bytes(bytes: &[u8]) -> Result<Self, HeaderParseError> {
        debug_assert_eq!(bytes.len(), Self::VERSION_NUM_BYTES);
        match (bytes[0], bytes[1]) {
            (0x01, 0x00) => Ok(Version::V1_0),
            (0x02, 0x00) => Ok(Version::V2_0),
            (major, minor) => Err(HeaderParseError::Version(major, minor)),
        }
    }

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

    /// Read header length.
    fn read_header_len<R: io::Read>(&self, mut reader: R) -> Result<usize, io::Error> {
        let mut buf = [0; 4];
        reader.read_exact(&mut buf[..self.header_len_num_bytes()])?;
        match *self {
            Version::V1_0 => Ok(LittleEndian::read_u16(&buf) as usize),
            Version::V2_0 => Ok(LittleEndian::read_u32(&buf) as usize),
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

#[derive(Clone, Debug)]
pub struct Header {
    pub type_descriptor: PyValue,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

quick_error! {
    #[derive(Debug)]
    pub enum FormatHeaderError {
        PyValue(err: PyValueFormatError) {
            description("error formatting Python value")
            display(x) -> ("{}", x.description())
            cause(err)
            from()
        }
    }
}

impl fmt::Display for Header {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.to_py_value())
    }
}

impl Header {
    fn from_py_value(value: PyValue) -> Result<Self, HeaderParseError> {
        if let PyValue::Dict(dict) = value {
            let mut type_descriptor: Option<PyValue> = None;
            let mut fortran_order: Option<bool> = None;
            let mut shape: Option<Vec<usize>> = None;
            for (key, val) in dict {
                match key {
                    PyValue::String(ref k) if k == "descr" => {
                        type_descriptor = Some(val);
                    }
                    PyValue::String(ref k) if k == "fortran_order" => {
                        if let PyValue::Boolean(b) = val {
                            fortran_order = Some(b);
                        } else {
                            return Err(HeaderParseError::IllegalValue(
                                "fortran_order".to_owned(),
                                val,
                            ));
                        }
                    }
                    PyValue::String(ref k) if k == "shape" => {
                        if let PyValue::Tuple(ref tuple) = val {
                            let mut out = Vec::with_capacity(tuple.len());
                            for elem in tuple {
                                if let &PyValue::Integer(ref int) = elem {
                                    out.push(int.to_usize().ok_or_else(|| {
                                        HeaderParseError::IllegalValue(
                                            "shape".to_owned(),
                                            val.clone(),
                                        )
                                    })?);
                                } else {
                                    return Err(HeaderParseError::IllegalValue(
                                        "shape".to_owned(),
                                        val.clone(),
                                    ));
                                }
                            }
                            shape = Some(out);
                        } else {
                            return Err(HeaderParseError::IllegalValue("shape".to_owned(), val));
                        }
                    }
                    k => return Err(HeaderParseError::UnknownKey(k)),
                }
            }
            match (type_descriptor, fortran_order, shape) {
                (Some(type_descriptor), Some(fortran_order), Some(shape)) => Ok(Header {
                    type_descriptor: type_descriptor,
                    fortran_order: fortran_order,
                    shape: shape,
                }),
                (None, _, _) => Err(HeaderParseError::MissingKey("descr".to_owned())),
                (_, None, _) => Err(HeaderParseError::MissingKey("fortran_order".to_owned())),
                (_, _, None) => Err(HeaderParseError::MissingKey("shaper".to_owned())),
            }
        } else {
            Err(HeaderParseError::MetaNotDict(value))
        }
    }

    pub fn from_reader<R: io::Read>(mut reader: R) -> Result<Self, ReadNpyError> {
        // Check for magic string.
        let mut buf = vec![0; MAGIC_STRING.len()];
        reader.read_exact(&mut buf)?;
        if buf != MAGIC_STRING {
            return Err(HeaderParseError::MagicString)?;
        }

        // Get version number.
        let mut buf = [0; Version::VERSION_NUM_BYTES];
        reader.read_exact(&mut buf)?;
        let version = Version::from_bytes(&buf)?;

        // Get `HEADER_LEN`.
        let header_len = version.read_header_len(&mut reader)?;

        // Parse the dictionary describing the array's format.
        let mut buf = vec![0; header_len];
        reader.read_exact(&mut buf)?;
        let without_newline = match buf.split_last() {
            Some((&b'\n', rest)) => rest,
            Some(_) | None => Err(HeaderParseError::MissingNewline)?,
        };
        let header_str = if without_newline.is_ascii() {
            unsafe { std::str::from_utf8_unchecked(without_newline) }
        } else {
            return Err(HeaderParseError::NonAscii)?;
        };
        Ok(Header::from_py_value(header_str
            .parse()
            .map_err(|err| HeaderParseError::from(err))?)?)
    }

    fn to_py_value(&self) -> PyValue {
        PyValue::Dict(vec![
            (
                PyValue::String("descr".into()),
                self.type_descriptor.clone(),
            ),
            (
                PyValue::String("fortran_order".into()),
                PyValue::Boolean(self.fortran_order),
            ),
            (
                PyValue::String("shape".into()),
                PyValue::Tuple(
                    self.shape
                        .iter()
                        .map(|&elem| PyValue::Integer(elem.into()))
                        .collect(),
                ),
            ),
        ])
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, FormatHeaderError> {
        // Metadata describing array's format as ASCII string.
        let mut arr_format = Vec::new();
        self.to_py_value().write_ascii(&mut arr_format)?;

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
            arr_format.push(b' ');
        }
        // Add final newline.
        arr_format.push(b'\n');

        // Determine length of header.
        let header_len = arr_format.len();

        let mut out = Vec::with_capacity(prefix_len + header_len);
        out.extend_from_slice(MAGIC_STRING);
        out.push(version.major_version());
        out.push(version.minor_version());
        out.extend_from_slice(&version.format_header_len(header_len));
        out.extend_from_slice(&arr_format);

        // Verify that length of header is divisible by 16.
        debug_assert_eq!(out.len() % 16, 0);

        Ok(out)
    }
}
