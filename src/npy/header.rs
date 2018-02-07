use byteorder::{ByteOrder, LittleEndian};
use pest::Parser;
use pest::iterators::Pair;
use std::error::Error;
use std::fmt;
use std::io;
use std::num::ParseIntError;
use super::ReadNpyError;

/// Magic string to indicate npy format.
const MAGIC_STRING: &[u8] = b"\x93NUMPY";

#[cfg(debug_assertions)]
const _GRAMMAR: &'static str = include_str!("header.pest");

#[derive(Parser)]
#[grammar = "npy/header.pest"]
struct HeaderParser;

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
            from(::std::str::Utf8Error)
        }
        Pest(msg: String) {
            description("dict parse error")
            display(x) -> ("{}: {}", x.description(), msg)
        }
        ParseInt(err: ParseIntError) {
            description("integer parsing error")
            display(x) -> ("{}: {}", x.description(), err)
            cause(err)
            from()
        }
        UnknownKey(key: String) {
            description("unknown key")
            display(x) -> ("{}: {}", x.description(), key)
        }
        MissingKey(key: String) {
            description("missing key")
            display(x) -> ("{}: {}", x.description(), key)
        }
        IllegalValue(key: String, value: String) {
            description("illegal value for key")
            display(x) -> ("{} {}: {}", x.description(), key, value)
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

/// Debug assertion that the value matches the pattern.
macro_rules! debug_assert_match {
    ($pattern:pat, $value:expr) => {
        if cfg!(debug_assertions) {
            #[allow(unreachable_patterns)]
            match $value {
                $pattern => {},
                _ => panic!("\
assertion failed: `(value matches pattern)`
 pattern: `{}`,
   value: `{:?}`", stringify!($pattern), $value),
            }
        }
    }
}

/// Extracts inner pairs matching the specified rules from the given pair.
///
/// Uses `debug_assert_match!` to assert that the inner pairs match the
/// specified rules. Uses `debug_assert_match!` to verify that there are no
/// additional pairs unless there is a `..` after the patterns (e.g.
/// `parse_pairs_as!(pair.into_inner(), (Rule::foo, Rule::bar | Rule::baz), ..)`).
///
/// For example:
///
/// ```rust,ignore
/// parse_pairs_as!(pair.into_inner(), (Rule::foo, Rule::bar))
/// ```
///
/// becomes
///
/// ```rust,ignore
/// {
///     let mut iter = &mut pair.into_inner();
///     let out = (
///         {
///             let tmp = iter.next().unwrap();
///             debug_assert_match!(Rule::foo, tmp.as_rule()),
///             tmp
///         },
///         {
///             let tmp = iter.next().unwrap();
///             debug_assert_match!(Rule::bar, tmp.as_rule()),
///             tmp
///         },
///     );
///     debug_assert_match!(None, iter.next());
///     out
/// }
/// ```
macro_rules! parse_pairs_as {
    // Entry point, fixed number of pairs, no trailing comma.
    ($pair:expr, ($($rules:pat),*)) => {
        parse_pairs_as!($pair, ($($rules),*,))
    };
    // Entry point, variable number of pairs, no trailing comma.
    ($pair:expr, ($($rules:pat),*), ..) => {
        parse_pairs_as!($pair, ($($rules),*,), ..)
    };
    // Entry point, fixed number of pairs, with trailing comma.
    ($pair:expr, ($($rules:pat),*,)) => {
        {
            let iter = &mut $pair;
            let out = parse_pairs_as!(@recur iter, (), ($($rules),*,));
            debug_assert_match!(None, iter.next());
            out
        }
    };
    // Entry point, variable number of pairs, with trailing comma.
    ($pair:expr, ($($rules:pat),*,), ..) => {
        {
            let iter = &mut $pair;
            parse_pairs_as!(@recur iter, (), ($($rules),*,))
        }
    };
    // Start processing the patterns.
    (@recur
        $iter:ident,
        (),
        ($head_rule:pat, $($remaining_rules:pat),*,)
    ) => {
        parse_pairs_as!(@recur
            $iter,
            (
                {
                    let tmp = $iter.next().unwrap();
                    debug_assert_match!($head_rule, tmp.as_rule());
                    tmp
                },
            ),
            ($($remaining_rules),*,)
        )
    };
    // Continue processing the patterns.
    (@recur
        $iter:ident,
        ($($processed:expr),*,),
        ($head_rule:pat, $($remaining_rules:pat),*,)
    ) => {
        parse_pairs_as!(@recur
            $iter,
            (
                $($processed),*,
                {
                    let tmp = $iter.next().unwrap();
                    debug_assert_match!($head_rule, tmp.as_rule());
                    tmp
                },
            ),
            ($($remaining_rules),*,)
        )
    };
    // Start processing the (single) pattern.
    (@recur $iter:ident, (), ($head_rule:pat,)) => {
        (
            {
                let tmp = $iter.next().unwrap();
                debug_assert_match!($head_rule, tmp.as_rule());
                tmp
            },
        )
    };
    // Finish processing the patterns.
    (@recur $iter:ident, ($($processed:expr),*,), ($head_rule:pat,)) => {
        (
            $($processed),*,
            {
                let tmp = $iter.next().unwrap();
                debug_assert_match!($head_rule, tmp.as_rule());
                tmp
            },
        )
    };
}

fn parse_string(string: Pair<Rule>) -> &str {
    debug_assert_eq!(string.as_rule(), Rule::string);
    let (string_body,) = parse_pairs_as!(string.into_inner(), (Rule::string_body,));
    string_body.as_str()
}

fn parse_bool(b: Pair<Rule>) -> bool {
    debug_assert_eq!(b.as_rule(), Rule::bool);
    match b.as_str() {
        "True" => true,
        "False" => false,
        _ => unreachable!(),
    }
}

fn parse_usize(u: Pair<Rule>) -> Result<usize, HeaderParseError> {
    debug_assert_eq!(u.as_rule(), Rule::usize);
    Ok(u.as_str().parse()?)
}

fn parse_shape(t: Pair<Rule>) -> Result<Vec<usize>, HeaderParseError> {
    debug_assert_eq!(t.as_rule(), Rule::shape);
    let mut shape = Vec::new();
    for pair in t.into_inner() {
        if let Rule::usize = pair.as_rule() {
            shape.push(parse_usize(pair)?);
        } else {
            unreachable!()
        }
    }
    Ok(shape)
}

#[derive(Clone, Debug)]
pub enum PyExpr {
    String(String),
    // TODO
}

impl<'a> From<&'a str> for PyExpr {
    fn from(s: &'a str) -> PyExpr {
        PyExpr::String(s.to_owned())
    }
}

impl From<String> for PyExpr {
    fn from(s: String) -> PyExpr {
        PyExpr::String(s)
    }
}

impl fmt::Display for PyExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            PyExpr::String(ref s) => {
                if s.contains("'") {
                    unimplemented!()
                } else {
                    write!(f, "'{}'", s)
                }
            }
        }
    }
}

fn parse_py_expr(expr: Pair<Rule>) -> Result<PyExpr, HeaderParseError> {
    match expr.as_rule() {
        Rule::string => Ok(PyExpr::String(parse_string(expr).to_owned())),
        _ => unimplemented!(),
    }
}

fn parse_header(h: Pair<Rule>) -> Result<Header, HeaderParseError> {
    debug_assert_eq!(h.as_rule(), Rule::header);
    let mut key: Option<&str> = None;
    let mut type_descriptor: Option<PyExpr> = None;
    let mut fortran_order: Option<bool> = None;
    let mut shape: Option<Vec<usize>> = None;
    for pair in h.into_inner() {
        match pair.as_rule() {
            Rule::key => {
                let (k,) = parse_pairs_as!(pair.into_inner(), (Rule::string,));
                key = Some(parse_string(k))
            }
            Rule::value => match key {
                Some("descr") => {
                    let (value,) = parse_pairs_as!(pair.into_inner(), (_,));
                    type_descriptor = Some(parse_py_expr(value)?);
                }
                Some("fortran_order") => {
                    let (value,) = parse_pairs_as!(pair.into_inner(), (_,));
                    if let Rule::bool = value.as_rule() {
                        fortran_order = Some(parse_bool(value));
                    } else {
                        return Err(HeaderParseError::IllegalValue(
                            "fortran_order".to_owned(),
                            value.as_str().to_owned(),
                        ));
                    }
                }
                Some("shape") => {
                    let (value,) = parse_pairs_as!(pair.into_inner(), (_,));
                    if let Rule::shape = value.as_rule() {
                        shape = Some(parse_shape(value)?);
                    } else {
                        return Err(HeaderParseError::IllegalValue(
                            "shape".to_owned(),
                            value.as_str().to_owned(),
                        ));
                    }
                }
                Some(k) => return Err(HeaderParseError::UnknownKey(k.to_owned())),
                None => unreachable!(),
            },
            _ => unreachable!(),
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
                assert!(header_len <= ::std::u16::MAX as usize);
                LittleEndian::write_u16(&mut out, header_len as u16);
            }
            Version::V2_0 => {
                assert!(header_len <= ::std::u32::MAX as usize);
                LittleEndian::write_u32(&mut out, header_len as u32);
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct Header {
    pub type_descriptor: PyExpr,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

impl Header {
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
        let header_str = ::std::str::from_utf8(&buf).map_err(|err| HeaderParseError::from(err))?;
        if !header_str.is_ascii() {
            return Err(HeaderParseError::NonAscii)?;
        }
        let mut parsed = HeaderParser::parse(Rule::header, header_str)
            .map_err(|e| HeaderParseError::Pest(format!("{}", e)))?;
        let (header,) = parse_pairs_as!(parsed, (Rule::header,));
        Ok(parse_header(header)?)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        // Metadata describing array's format as ASCII string.
        let mut arr_format = format!(
            "{{'descr': {}, 'fortran_order': {}, 'shape': (",
            self.type_descriptor,
            if self.fortran_order { "True" } else { "False" },
        );
        for (i, axis_len) in self.shape.iter().enumerate() {
            arr_format.push_str(&format!("{}", axis_len));
            if self.shape.len() == 1 || self.shape.len() != 0 && i != self.shape.len() - 1 {
                arr_format.push(',');
            }
        }
        arr_format.push_str("), }");

        // Length of a '\n' char in bytes.
        const NEWLINE_LEN: usize = 1;

        // Determine appropriate version based on minimum number of bytes needed to
        // represent header length (including final newline).
        let version = if arr_format.len() + NEWLINE_LEN > ::std::u16::MAX as usize {
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
}
