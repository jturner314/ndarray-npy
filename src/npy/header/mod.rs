#[macro_use]
mod parse_macros;
mod py_expr;

pub use self::py_expr::PyExpr;

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
#[grammar = "npy/header/header.pest"]
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
        UnknownKey(key: PyExpr) {
            description("unknown key")
            display(x) -> ("{}: {}", x.description(), key)
        }
        MissingKey(key: String) {
            description("missing key")
            display(x) -> ("{}: {}", x.description(), key)
        }
        IllegalValue(key: String, value: PyExpr) {
            description("illegal value for key")
            display(x) -> ("{} {}: {}", x.description(), key, value)
        }
        Custom(msg: String) {
            description("custom error")
            display(x) -> ("{}: {}", x.description(), msg)
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

fn parse_header(h: Pair<Rule>) -> Result<Header, HeaderParseError> {
    debug_assert_eq!(h.as_rule(), Rule::header);
    let (dict,) = parse_pairs_as!(h.into_inner(), (Rule::dict,));
    let mut type_descriptor: Option<PyExpr> = None;
    let mut fortran_order: Option<bool> = None;
    let mut shape: Option<Vec<usize>> = None;
    for (key, value) in py_expr::parse_dict(dict)? {
        match key {
            PyExpr::String(ref k) if k == "descr" => {
                type_descriptor = Some(value);
            }
            PyExpr::String(ref k) if k == "fortran_order" => {
                if let PyExpr::Boolean(b) = value {
                    fortran_order = Some(b);
                } else {
                    return Err(HeaderParseError::IllegalValue(
                        "fortran_order".to_owned(),
                        value,
                    ));
                }
            }
            PyExpr::String(ref k) if k == "shape" => {
                if let PyExpr::Tuple(ref tuple) = value {
                    let mut out = Vec::with_capacity(tuple.len());
                    for elem in tuple {
                        if let &PyExpr::Integer(int) = elem {
                            out.push(int as usize);
                        } else {
                            return Err(HeaderParseError::IllegalValue(
                                "shape".to_owned(),
                                value.clone(),
                            ));
                        }
                    }
                    shape = Some(out);
                } else {
                    return Err(HeaderParseError::IllegalValue("shape".to_owned(), value));
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

quick_error! {
    #[derive(Debug)]
    pub enum FormatHeaderError {
        Format(err: fmt::Error) {
            description("error formatting header; this is most likely due to a zero-size set")
            display(x) -> ("{}", x.description())
            cause(err)
            from()
        }
    }
}

impl fmt::Display for Header {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.to_py_expr())
    }
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

    fn to_py_expr(&self) -> PyExpr {
        PyExpr::Dict(vec![
            (PyExpr::String("descr".into()), self.type_descriptor.clone()),
            (
                PyExpr::String("fortran_order".into()),
                PyExpr::Boolean(self.fortran_order),
            ),
            (
                PyExpr::String("shape".into()),
                PyExpr::Tuple(
                    self.shape
                        .iter()
                        .map(|&elem| PyExpr::Integer(elem as i64))
                        .collect(),
                ),
            ),
        ])
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, FormatHeaderError> {
        // Metadata describing array's format as ASCII string.
        let mut arr_format = String::new();
        fmt::write(&mut arr_format, format_args!("{}", self))?;

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

        Ok(out)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse_string_example() {
        let input = r"'hello\th\03o\x1bw\
are\a\'y\u1234o\U00031234u'";
        let mut parsed = HeaderParser::parse(Rule::string, input)
            .unwrap_or_else(|err| panic!("failed to parse: {}", err));
        let s = parse_string(parse_pairs_as!(parsed, (Rule::string,)).0).unwrap();
        assert_eq!(s, "hello\th\x03o\x1bware\x07'y\u{1234}o\u{31234}u");
    }

    #[test]
    fn parse_number_expr_example() {
        let input = "+-23 + 4.5 -+- -5j - 3e2";
        let mut parsed = HeaderParser::parse(Rule::number_expr, input)
            .unwrap_or_else(|err| panic!("failed to parse: {}", err));
        let expr = parse_number_expr(parse_pairs_as!(parsed, (Rule::number_expr,)).0);
        assert_eq!(expr, PyExpr::Complex(-23. + 4.5 - 3e2, -5.));
    }

    #[test]
    fn parse_integer_example() {
        let inputs = ["0b_1001_0010_1010", "0o44_52", "0x9_2a", "2_346"];
        for input in &inputs {
            let mut parsed = HeaderParser::parse(Rule::integer, input)
                .unwrap_or_else(|err| panic!("failed to parse: {}", err));
            let int = parse_integer(parse_pairs_as!(parsed, (Rule::integer,)).0);
            assert_eq!(int, 2346);
        }
    }

    #[test]
    fn parse_float_example() {
        let input = "3_51.4_6e-2_7";
        let mut parsed = HeaderParser::parse(Rule::float, input)
            .unwrap_or_else(|err| panic!("failed to parse: {}", err));
        let float = parse_float(parse_pairs_as!(parsed, (Rule::float,)).0);
        assert_eq!(float, 351.46e-27);
    }

    #[test]
    fn parse_tuple_example() {
        use self::PyExpr::*;
        for &(input, ref correct) in &[
            ("()", Tuple(vec![])),
            ("(5, )", Tuple(vec![Integer(5)])),
            ("(1, 2)", Tuple(vec![Integer(1), Integer(2)])),
            ("(1, 2,)", Tuple(vec![Integer(1), Integer(2)])),
        ] {
            let mut parsed = HeaderParser::parse(Rule::expr, input)
                .unwrap_or_else(|err| panic!("failed to parse: {}", err));
            let tuple = parse_expr(parse_pairs_as!(parsed, (Rule::expr,)).0).unwrap();
            assert_eq!(tuple, *correct);
        }
    }

    #[test]
    fn parse_list_example() {
        use self::PyExpr::*;
        for &(input, ref correct) in &[
            ("[]", List(vec![])),
            ("[3]", List(vec![Integer(3)])),
            ("[5,]", List(vec![Integer(5)])),
            ("[1, 2]", List(vec![Integer(1), Integer(2)])),
            (
                "[5, 6., \"foo\", 2+7j,]",
                List(vec![
                    Integer(5),
                    Float(6.),
                    String("foo".into()),
                    Complex(2., 7.),
                ]),
            ),
        ] {
            let mut parsed = HeaderParser::parse(Rule::expr, input)
                .unwrap_or_else(|err| panic!("failed to parse: {}", err));
            let list = parse_expr(parse_pairs_as!(parsed, (Rule::expr,)).0).unwrap();
            assert_eq!(list, *correct);
        }
    }

    #[test]
    fn parse_dict_example() {
        use self::PyExpr::*;
        for &(input, ref correct) in &[
            ("{}", Dict(vec![])),
            ("{3: \"hi\"}", Dict(vec![(Integer(3), String("hi".into()))])),
            (
                "{5: 6., \"foo\" : True}",
                Dict(vec![
                    (Integer(5), Float(6.)),
                    (String("foo".into()), Boolean(true)),
                ]),
            ),
        ] {
            let mut parsed = HeaderParser::parse(Rule::expr, input)
                .unwrap_or_else(|err| panic!("failed to parse: {}", err));
            let dict = parse_expr(parse_pairs_as!(parsed, (Rule::expr,)).0).unwrap();
            assert_eq!(dict, *correct);
        }
    }

    #[test]
    fn parse_set_example() {
        use self::PyExpr::*;
        for &(input, ref correct) in &[
            ("{3}", Set(vec![Integer(3)])),
            ("{5,}", Set(vec![Integer(5)])),
            ("{1, 2}", Set(vec![Integer(1), Integer(2)])),
        ] {
            let mut parsed = HeaderParser::parse(Rule::expr, input)
                .unwrap_or_else(|err| panic!("failed to parse: {}", err));
            let set = parse_expr(parse_pairs_as!(parsed, (Rule::expr,)).0).unwrap();
            assert_eq!(set, *correct);
        }
    }

    #[test]
    fn parse_list_of_tuples_example() {
        use self::PyExpr::*;
        for &(input, ref correct) in &[
            (
                "[('big', '>i4'), ('little', '<i4')]",
                List(vec![
                    Tuple(vec![String("big".into()), String(">i4".into())]),
                    Tuple(vec![String("little".into()), "<i4".into()]),
                ]),
            ),
            (
                "[(1, 2, 3), (4,)]",
                List(vec![
                    Tuple(vec![Integer(1), Integer(2), Integer(3)]),
                    Tuple(vec![Integer(4)]),
                ]),
            ),
        ] {
            let mut parsed = HeaderParser::parse(Rule::expr, input)
                .unwrap_or_else(|err| panic!("failed to parse: {}", err));
            let list = parse_expr(parse_pairs_as!(parsed, (Rule::expr,)).0).unwrap();
            assert_eq!(list, *correct);
        }
    }

    #[test]
    fn format_string() {
        let expr = PyExpr::String("hello\th\x03\u{ff}o\x1bware\x07'y\u{1234}o\u{31234}u".into());
        let formatted = format!("{}", expr);
        assert_eq!(
            formatted,
            "'hello\th\x03\\xffo\x1bware\x07\\'y\\u1234o\\U00031234u'"
        )
    }

    #[test]
    fn format_bytes() {
        let expr = PyExpr::Bytes(b"hello\th\x03\xffo\x1bware\x07'you"[..].into());
        let formatted = format!("{}", expr);
        assert_eq!(formatted, "b'hello\th\x03\\xffo\x1bware\x07\\'you'")
    }

    #[test]
    fn format_complex() {
        use self::PyExpr::*;
        assert_eq!("1e0+3e0j", format!("{}", Complex(1., 3.)));
        assert_eq!("1e0-3e0j", format!("{}", Complex(1., -3.)));
        assert_eq!("-1e0+3e0j", format!("{}", Complex(-1., 3.)));
        assert_eq!("-1e0-3e0j", format!("{}", Complex(-1., -3.)));
    }

    #[test]
    fn format_tuple() {
        use self::PyExpr::*;
        assert_eq!("()", format!("{}", Tuple(vec![])));
        assert_eq!("(1,)", format!("{}", Tuple(vec![Integer(1)])));
        assert_eq!("(1, 2)", format!("{}", Tuple(vec![Integer(1), Integer(2)])));
        assert_eq!(
            "(1, 2, 'hi')",
            format!(
                "{}",
                Tuple(vec![Integer(1), Integer(2), String("hi".into())])
            ),
        );
    }

    #[test]
    fn format_list() {
        use self::PyExpr::*;
        assert_eq!("[]", format!("{}", List(vec![])));
        assert_eq!("[1]", format!("{}", List(vec![Integer(1)])));
        assert_eq!("[1, 2]", format!("{}", List(vec![Integer(1), Integer(2)])));
        assert_eq!(
            "[1, 2, 'hi']",
            format!(
                "{}",
                List(vec![Integer(1), Integer(2), String("hi".into())])
            ),
        );
    }

    #[test]
    fn format_dict() {
        use self::PyExpr::*;
        assert_eq!("{}", format!("{}", Dict(vec![])));
        assert_eq!(
            "{1: 2}",
            format!("{}", Dict(vec![(Integer(1), Integer(2))]))
        );
        assert_eq!(
            "{1: 2, 'foo': 'bar'}",
            format!(
                "{}",
                Dict(vec![
                    (Integer(1), Integer(2)),
                    (String("foo".into()), String("bar".into())),
                ])
            ),
        );
    }

    #[test]
    #[should_panic]
    fn format_empty_set() {
        use self::PyExpr::*;
        format!("{}", Set(vec![]));
    }

    #[test]
    fn format_set() {
        use self::PyExpr::*;
        assert_eq!("{1}", format!("{}", Set(vec![Integer(1)])));
        assert_eq!("{1, 2}", format!("{}", Set(vec![Integer(1), Integer(2)])));
        assert_eq!(
            "{1, 2, 'hi'}",
            format!("{}", Set(vec![Integer(1), Integer(2), String("hi".into())])),
        );
    }

    #[test]
    fn format_nested() {
        use self::PyExpr::*;
        assert_eq!(
            "{'foo': [1, True], {2e0+3e0j}: 4}",
            format!(
                "{}",
                Dict(vec![
                    (String("foo".into()), List(vec![Integer(1), Boolean(true)])),
                    (Set(vec![Complex(2., 3.)]), Integer(4)),
                ])
            ),
        );
    }
}
