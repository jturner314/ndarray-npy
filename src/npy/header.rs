use byteorder::{ByteOrder, LittleEndian};
use num_traits::ToPrimitive;
use py_literal::{
    FormatError as PyValueFormatError, ParseError as PyValueParseError, Value as PyValue,
};
use std::error::Error;
use std::fmt;
use std::io;

/// Magic string to indicate npy format.
const MAGIC_STRING: &[u8] = b"\x93NUMPY";

#[derive(Debug)]
pub enum ParseHeaderError {
    MagicString,
    Version {
        major: u8,
        minor: u8,
    },
    /// Indicates that the array format string contains non-ASCII characters.
    /// This is an error for .npy format versions 1.0 and 2.0.
    NonAscii,
    /// Error parsing the array format string as UTF-8. This does not apply to
    /// .npy format versions 1.0 and 2.0, which require the array format string
    /// to be ASCII.
    Utf8Parse(std::str::Utf8Error),
    UnknownKey(PyValue),
    MissingKey(String),
    IllegalValue {
        key: String,
        value: PyValue,
    },
    DictParse(PyValueParseError),
    MetaNotDict(PyValue),
    MissingNewline,
}

impl Error for ParseHeaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        use ParseHeaderError::*;
        match self {
            MagicString => None,
            Version { .. } => None,
            NonAscii => None,
            Utf8Parse(err) => Some(err),
            UnknownKey(_) => None,
            MissingKey(_) => None,
            IllegalValue { .. } => None,
            DictParse(err) => Some(err),
            MetaNotDict(_) => None,
            MissingNewline => None,
        }
    }
}

impl fmt::Display for ParseHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ParseHeaderError::*;
        match self {
            MagicString => write!(f, "start does not match magic string"),
            Version { major, minor } => write!(f, "unknown version number: {}.{}", major, minor),
            NonAscii => write!(f, "non-ascii in array format string; this is not supported in .npy format versions 1.0 and 2.0"),
            Utf8Parse(err) => write!(f, "error parsing array format string as UTF-8: {}", err),
            UnknownKey(key) => write!(f, "unknown key: {}", key),
            MissingKey(key) => write!(f, "missing key: {}", key),
            IllegalValue { key, value } => write!(f, "illegal value for key {}: {}", key, value),
            DictParse(err) => write!(f, "error parsing metadata dict: {}", err),
            MetaNotDict(value) => write!(f, "metadata is not a dict: {}", value),
            MissingNewline => write!(f, "newline missing at end of header"),
        }
    }
}

impl From<std::str::Utf8Error> for ParseHeaderError {
    fn from(err: std::str::Utf8Error) -> ParseHeaderError {
        ParseHeaderError::Utf8Parse(err)
    }
}

impl From<PyValueParseError> for ParseHeaderError {
    fn from(err: PyValueParseError) -> ParseHeaderError {
        ParseHeaderError::DictParse(err)
    }
}

#[derive(Debug)]
pub enum ReadHeaderError {
    Io(io::Error),
    Parse(ParseHeaderError),
}

impl Error for ReadHeaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ReadHeaderError::Io(err) => Some(err),
            ReadHeaderError::Parse(err) => Some(err),
        }
    }
}

impl fmt::Display for ReadHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ReadHeaderError::Io(err) => write!(f, "I/O error: {}", err),
            ReadHeaderError::Parse(err) => write!(f, "error parsing header: {}", err),
        }
    }
}

impl From<io::Error> for ReadHeaderError {
    fn from(err: io::Error) -> ReadHeaderError {
        ReadHeaderError::Io(err)
    }
}

impl From<ParseHeaderError> for ReadHeaderError {
    fn from(err: ParseHeaderError) -> ReadHeaderError {
        ReadHeaderError::Parse(err)
    }
}

#[allow(non_camel_case_types)]
enum Version {
    V1_0,
    V2_0,
    V3_0,
}

impl Version {
    /// Number of bytes taken up by version number (1 byte for major version, 1
    /// byte for minor version).
    const VERSION_NUM_BYTES: usize = 2;

    fn from_bytes(bytes: &[u8]) -> Result<Self, ParseHeaderError> {
        debug_assert_eq!(bytes.len(), Self::VERSION_NUM_BYTES);
        match (bytes[0], bytes[1]) {
            (0x01, 0x00) => Ok(Version::V1_0),
            (0x02, 0x00) => Ok(Version::V2_0),
            (0x03, 0x00) => Ok(Version::V3_0),
            (major, minor) => Err(ParseHeaderError::Version { major, minor }),
        }
    }

    /// Major version number.
    fn major_version(&self) -> u8 {
        match *self {
            Version::V1_0 => 1,
            Version::V2_0 => 2,
            Version::V3_0 => 3,
        }
    }

    /// Major version number.
    fn minor_version(&self) -> u8 {
        match *self {
            Version::V1_0 => 0,
            Version::V2_0 => 0,
            Version::V3_0 => 0,
        }
    }

    /// Number of bytes in representation of header length.
    fn header_len_num_bytes(&self) -> usize {
        match *self {
            Version::V1_0 => 2,
            Version::V2_0 | Version::V3_0 => 4,
        }
    }

    /// Read header length.
    fn read_header_len<R: io::Read>(&self, mut reader: R) -> Result<usize, io::Error> {
        let mut buf = [0; 4];
        reader.read_exact(&mut buf[..self.header_len_num_bytes()])?;
        match *self {
            Version::V1_0 => Ok(LittleEndian::read_u16(&buf) as usize),
            Version::V2_0 | Version::V3_0 => Ok(LittleEndian::read_u32(&buf) as usize),
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
            Version::V2_0 | Version::V3_0 => {
                assert!(header_len <= std::u32::MAX as usize);
                LittleEndian::write_u32(&mut out, header_len as u32);
            }
        }
        out
    }
}

#[derive(Debug)]
pub enum FormatHeaderError {
    PyValue(PyValueFormatError),
}

impl Error for FormatHeaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            FormatHeaderError::PyValue(err) => Some(err),
        }
    }
}

impl fmt::Display for FormatHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FormatHeaderError::PyValue(err) => write!(f, "error formatting Python value: {}", err),
        }
    }
}

impl From<PyValueFormatError> for FormatHeaderError {
    fn from(err: PyValueFormatError) -> FormatHeaderError {
        FormatHeaderError::PyValue(err)
    }
}

#[derive(Debug)]
pub enum WriteHeaderError {
    Io(io::Error),
    Format(FormatHeaderError),
}

impl Error for WriteHeaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            WriteHeaderError::Io(err) => Some(err),
            WriteHeaderError::Format(err) => Some(err),
        }
    }
}

impl fmt::Display for WriteHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            WriteHeaderError::Io(err) => write!(f, "I/O error: {}", err),
            WriteHeaderError::Format(err) => write!(f, "error formatting header: {}", err),
        }
    }
}

impl From<io::Error> for WriteHeaderError {
    fn from(err: io::Error) -> WriteHeaderError {
        WriteHeaderError::Io(err)
    }
}

impl From<FormatHeaderError> for WriteHeaderError {
    fn from(err: FormatHeaderError) -> WriteHeaderError {
        WriteHeaderError::Format(err)
    }
}

#[derive(Clone, Debug)]
pub struct Header {
    pub type_descriptor: PyValue,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

impl fmt::Display for Header {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.to_py_value())
    }
}

impl Header {
    fn from_py_value(value: PyValue) -> Result<Self, ParseHeaderError> {
        if let PyValue::Dict(dict) = value {
            let mut type_descriptor: Option<PyValue> = None;
            let mut fortran_order: Option<bool> = None;
            let mut shape: Option<Vec<usize>> = None;
            for (key, value) in dict {
                match key {
                    PyValue::String(ref k) if k == "descr" => {
                        type_descriptor = Some(value);
                    }
                    PyValue::String(ref k) if k == "fortran_order" => {
                        if let PyValue::Boolean(b) = value {
                            fortran_order = Some(b);
                        } else {
                            return Err(ParseHeaderError::IllegalValue {
                                key: "fortran_order".to_owned(),
                                value,
                            });
                        }
                    }
                    PyValue::String(ref k) if k == "shape" => {
                        fn parse_shape(value: &PyValue) -> Option<Vec<usize>> {
                            value
                                .as_tuple()?
                                .iter()
                                .map(|elem| elem.as_integer()?.to_usize())
                                .collect()
                        }
                        if let Some(s) = parse_shape(&value) {
                            shape = Some(s);
                        } else {
                            return Err(ParseHeaderError::IllegalValue {
                                key: "shape".to_owned(),
                                value,
                            });
                        }
                    }
                    k => return Err(ParseHeaderError::UnknownKey(k)),
                }
            }
            match (type_descriptor, fortran_order, shape) {
                (Some(type_descriptor), Some(fortran_order), Some(shape)) => Ok(Header {
                    type_descriptor,
                    fortran_order,
                    shape,
                }),
                (None, _, _) => Err(ParseHeaderError::MissingKey("descr".to_owned())),
                (_, None, _) => Err(ParseHeaderError::MissingKey("fortran_order".to_owned())),
                (_, _, None) => Err(ParseHeaderError::MissingKey("shaper".to_owned())),
            }
        } else {
            Err(ParseHeaderError::MetaNotDict(value))
        }
    }

    pub fn from_reader<R: io::Read>(mut reader: R) -> Result<Self, ReadHeaderError> {
        // Check for magic string.
        let mut buf = vec![0; MAGIC_STRING.len()];
        reader.read_exact(&mut buf)?;
        if buf != MAGIC_STRING {
            return Err(ParseHeaderError::MagicString.into());
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
            Some(_) | None => return Err(ParseHeaderError::MissingNewline.into()),
        };
        let header_str = match version {
            Version::V1_0 | Version::V2_0 => {
                if without_newline.is_ascii() {
                    // ASCII strings are always valid UTF-8.
                    unsafe { std::str::from_utf8_unchecked(without_newline) }
                } else {
                    return Err(ParseHeaderError::NonAscii.into());
                }
            }
            Version::V3_0 => {
                std::str::from_utf8(without_newline).map_err(ParseHeaderError::from)?
            }
        };
        let header_dict: PyValue = header_str.parse().map_err(ParseHeaderError::from)?;
        Ok(Header::from_py_value(header_dict)?)
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

    pub fn write<W: io::Write>(&self, mut writer: W) -> Result<(), WriteHeaderError> {
        let bytes = self.to_bytes()?;
        writer.write_all(&bytes)?;
        Ok(())
    }
}
