use byteorder::{ByteOrder, LittleEndian};
use pest::Parser;
use pest::iterators::Pair;
use std::error::Error;
use std::fmt::{self, Write};
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
            debug_assert_match!(Option::None, iter.next());
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

fn parse_string_escape_seq(escape_seq: Pair<Rule>) -> Result<char, HeaderParseError> {
    debug_assert_eq!(escape_seq.as_rule(), Rule::string_escape_seq);
    let (seq,) = parse_pairs_as!(escape_seq.into_inner(), (_,));
    match seq.as_rule() {
        Rule::char_escape => Ok(match seq.as_str() {
            "\\" => '\\',
            "'" => '\'',
            "\"" => '"',
            "a" => '\x07',
            "b" => '\x08',
            "f" => '\x0C',
            "n" => '\n',
            "r" => '\r',
            "t" => '\t',
            "v" => '\x0B',
            _ => unreachable!(),
        }),
        Rule::octal_escape => ::std::char::from_u32(u32::from_str_radix(seq.as_str(), 8).unwrap())
            .ok_or_else(|| {
                HeaderParseError::Custom(format!("Octal escape is invalid: {}", seq.as_str()))
            }),
        Rule::hex_escape => {
            ::std::char::from_u32(u32::from_str_radix(&seq.as_str()[1..], 16).unwrap()).ok_or_else(
                || HeaderParseError::Custom(format!("Hex escape is invalid: {}", seq.as_str())),
            )
        }
        Rule::name_escape => Err(HeaderParseError::Custom(
            "Unicode name escapes are not supported.".into(),
        )),
        Rule::any_escape => Ok(seq.as_str().chars().next().unwrap()),
        _ => unreachable!(),
    }
}

fn parse_string(string: Pair<Rule>) -> Result<String, HeaderParseError> {
    debug_assert_eq!(string.as_rule(), Rule::string);
    let (string_body,) = parse_pairs_as!(string.into_inner(), (_,));
    match string_body.as_rule() {
        Rule::short_string_body | Rule::long_string_body => {
            let mut out = String::new();
            for item in string_body.into_inner() {
                match item.as_rule() {
                    Rule::short_string_non_escape | Rule::long_string_non_escape => {
                        out.push_str(item.as_str())
                    }
                    Rule::line_continuation_seq => (),
                    Rule::string_escape_seq => out.push(parse_string_escape_seq(item)?),
                    _ => unreachable!(),
                }
            }
            Ok(out)
        }
        _ => unreachable!(),
    }
}

fn parse_number_expr(expr: Pair<Rule>) -> PyExpr {
    debug_assert_eq!(expr.as_rule(), Rule::number_expr);
    let mut result = PyExpr::Integer(0);
    let mut neg = false;
    for pair in expr.into_inner() {
        match pair.as_rule() {
            Rule::minus_sign => neg = !neg,
            Rule::number => {
                let num = parse_number(pair);
                if neg {
                    result = sub_numbers(result, num).unwrap();
                } else {
                    result = add_numbers(result, num).unwrap();
                }
                neg = false;
            }
            _ => unreachable!(),
        }
    }
    result
}

fn parse_number(number: Pair<Rule>) -> PyExpr {
    debug_assert_eq!(number.as_rule(), Rule::number);
    let (inner,) = parse_pairs_as!(number.into_inner(), (_,));
    match inner.as_rule() {
        Rule::imag => parse_imag(inner),
        Rule::float => PyExpr::Float(parse_float(inner)),
        Rule::integer => PyExpr::Integer(parse_integer(inner)),
        _ => unreachable!(),
    }
}

fn parse_integer(int: Pair<Rule>) -> i64 {
    debug_assert_eq!(int.as_rule(), Rule::integer);
    let (inner,) = parse_pairs_as!(int.into_inner(), (_,));
    match inner.as_rule() {
        Rule::bin_integer => {
            let digits: String = inner.into_inner().map(|digit| digit.as_str()).collect();
            i64::from_str_radix(&digits, 2).expect(&format!(
                "failure parsing binary integer with digits {}",
                digits
            ))
        }
        Rule::oct_integer => {
            let digits: String = inner.into_inner().map(|digit| digit.as_str()).collect();
            i64::from_str_radix(&digits, 8).expect(&format!(
                "failure parsing octal integer with digits {}",
                digits
            ))
        }
        Rule::hex_integer => {
            let digits: String = inner.into_inner().map(|digit| digit.as_str()).collect();
            i64::from_str_radix(&digits, 16).expect(&format!(
                "failure parsing hexadecimal integer with digits {}",
                digits
            ))
        }
        Rule::dec_integer => {
            let digits: String = inner.into_inner().map(|digit| digit.as_str()).collect();
            digits
                .parse()
                .expect(&format!("failure parsing integer with digits {}", digits))
        }
        _ => unreachable!(),
    }
}

fn parse_float(float: Pair<Rule>) -> f64 {
    debug_assert_eq!(float.as_rule(), Rule::float);
    let (inner,) = parse_pairs_as!(float.into_inner(), (_,));
    let mut parsable = String::new();
    for pair in inner.into_inner().flatten() {
        match pair.as_rule() {
            Rule::digit => parsable.push_str(pair.as_str()),
            Rule::fraction => parsable.push('.'),
            Rule::pos_exponent => parsable.push('e'),
            Rule::neg_exponent => parsable.push_str("e-"),
            _ => (),
        }
    }
    parsable
        .parse()
        .expect(&format!("Failure parsing {}", parsable))
}

fn parse_imag(imag: Pair<Rule>) -> PyExpr {
    debug_assert_eq!(imag.as_rule(), Rule::imag);
    let (inner,) = parse_pairs_as!(imag.into_inner(), (_,));
    let imag = match inner.as_rule() {
        Rule::float => parse_float(inner),
        Rule::digit_part => {
            let digits: String = inner.into_inner().map(|digit| digit.as_str()).collect();
            digits
                .parse()
                .expect(&format!("failure parsing imag with digits {}", digits))
        }
        _ => unreachable!(),
    };
    PyExpr::Complex(0., imag)
}

/// Parses a tuple, list, or set.
fn parse_seq(seq: Pair<Rule>) -> Result<Vec<PyExpr>, HeaderParseError> {
    debug_assert!([Rule::tuple, Rule::list, Rule::set].contains(&seq.as_rule()));
    seq.into_inner().map(|elem| parse_expr(elem)).collect()
}

fn parse_dict(dict: Pair<Rule>) -> Result<Vec<(PyExpr, PyExpr)>, HeaderParseError> {
    debug_assert_eq!(dict.as_rule(), Rule::dict);
    let mut out = Vec::new();
    for elem in dict.into_inner() {
        let (key, value) = parse_pairs_as!(elem.into_inner(), (Rule::expr, Rule::expr));
        out.push((parse_expr(key)?, parse_expr(value)?));
    }
    Ok(out)
}

fn parse_boolean(b: Pair<Rule>) -> bool {
    debug_assert_eq!(b.as_rule(), Rule::boolean);
    match b.as_str() {
        "True" => true,
        "False" => false,
        _ => unreachable!(),
    }
}

fn parse_none(none: Pair<Rule>) -> PyExpr {
    debug_assert_eq!(none.as_rule(), Rule::none);
    PyExpr::None
}

/// NumPy uses [`ast.literal_eval()`] to parse the header dictionary.
/// `literal_eval()` supports only the following Python literals: strings,
/// bytes, numbers, tuples, lists, dicts, sets, booleans, and `None`.
///
/// [`ast.literal_eval()`]: https://docs.python.org/3/library/ast.html#ast.literal_eval
fn parse_expr(expr: Pair<Rule>) -> Result<PyExpr, HeaderParseError> {
    debug_assert_eq!(expr.as_rule(), Rule::expr);
    let (inner,) = parse_pairs_as!(expr.into_inner(), (_,));
    match inner.as_rule() {
        Rule::string => Ok(PyExpr::String(parse_string(inner)?)),
        Rule::number_expr => Ok(parse_number_expr(inner)),
        Rule::tuple => Ok(PyExpr::Tuple(parse_seq(inner)?)),
        Rule::list => Ok(PyExpr::List(parse_seq(inner)?)),
        Rule::dict => Ok(PyExpr::Dict(parse_dict(inner)?)),
        Rule::set => Ok(PyExpr::Set(parse_seq(inner)?)),
        Rule::boolean => Ok(PyExpr::Boolean(parse_boolean(inner))),
        Rule::none => Ok(parse_none(inner)),
        _ => unreachable!(),
    }
}

/// Represents a Python literal expression.
///
/// This should be able to express everything that Python's
/// [`ast.literal_eval()`] can evaluate, except for operators. Similar to
/// `literal_eval()`, addition and subtraction of numbers is supported in the
/// parser. However, binary addition and subtraction operators cannot be
/// formatted using `PyExpr`.
///
/// [`ast.literal_eval()`]: https://docs.python.org/3/library/ast.html#ast.literal_eval
#[derive(Clone, Debug, PartialEq)]
pub enum PyExpr {
    /// Python string (`str`). When parsing, backslash escapes are interpreted.
    /// When formatting, backslash escapes are used to ensure the result is
    /// contains only ASCII chars.
    String(String),
    /// Python byte sequence (`bytes`). When parsing, backslash escapes are
    /// interpreted. When formatting, backslash escapes are used to ensure the
    /// result is contains only ASCII chars.
    Bytes(Vec<u8>),
    /// Python integer (`int`). Python integers have unlimited precision, but
    /// `i64` should be good enough for our needs.
    Integer(i64),
    /// Python floating-point number (`float`). The representation and
    /// precision of the Python `float` type varies by the machine where the
    /// program is executing, but `f64` should be good enough.
    Float(f64),
    /// Python complex number (`complex`). The Python `complex` type contains
    /// two `float` types.
    Complex(f64, f64),
    /// Python tuple (`tuple`).
    Tuple(Vec<PyExpr>),
    /// Python list (`list`).
    List(Vec<PyExpr>),
    /// Python dictionary (`dict`).
    Dict(Vec<(PyExpr, PyExpr)>),
    /// Python set (`set`).
    Set(Vec<PyExpr>),
    /// Python boolean (`bool`).
    Boolean(bool),
    /// Python `None`.
    None,
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

/// Adds two numbers.
///
/// Returns `Err` if either of the arguments is not a number.
fn add_numbers(lhs: PyExpr, rhs: PyExpr) -> Result<PyExpr, ()> {
    use self::PyExpr::*;
    match (lhs, rhs) {
        (Integer(int1), Integer(int2)) => Ok(Integer(int1 + int2)),
        (Float(float1), Float(float2)) => Ok(Float(float1 + float2)),
        (Complex(real1, imag1), Complex(real2, imag2)) => Ok(Complex(real1 + real2, imag1 + imag2)),
        (Integer(int), Float(float)) | (Float(float), Integer(int)) => {
            Ok(Float(int as f64 + float))
        }
        (Integer(int), Complex(real, imag)) | (Complex(real, imag), Integer(int)) => {
            Ok(Complex(int as f64 + real, imag))
        }
        (Float(float), Complex(real, imag)) | (Complex(real, imag), Float(float)) => {
            Ok(Complex(float + real, imag))
        }
        _ => Err(()),
    }
}

/// Subtracts two numbers.
///
/// Returns `Err` if either of the arguments is not a number.
fn sub_numbers(lhs: PyExpr, rhs: PyExpr) -> Result<PyExpr, ()> {
    use self::PyExpr::*;
    match (lhs, rhs) {
        (Integer(int1), Integer(int2)) => Ok(Integer(int1 - int2)),
        (Integer(int), Float(float)) => Ok(Float(int as f64 - float)),
        (Integer(int), Complex(real, imag)) => Ok(Complex(int as f64 - real, -imag)),
        (Float(float), Integer(int)) => Ok(Float(float - int as f64)),
        (Float(float1), Float(float2)) => Ok(Float(float1 - float2)),
        (Float(float), Complex(real, imag)) => Ok(Complex(float - real, -imag)),
        (Complex(real, imag), Integer(int)) => Ok(Complex(real - int as f64, imag)),
        (Complex(real, imag), Float(float)) => Ok(Complex(real - float, imag)),
        (Complex(real1, imag1), Complex(real2, imag2)) => Ok(Complex(real1 - real2, imag1 - imag2)),
        _ => Err(()),
    }
}

impl fmt::Display for PyExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            PyExpr::String(ref s) => {
                f.write_char('\'')?;
                for c in s.chars() {
                    match c {
                        '\\' => f.write_str(r"\\")?,
                        '\r' => f.write_str(r"\r")?,
                        '\n' => f.write_str(r"\n")?,
                        '\'' => f.write_str(r"\'")?,
                        c if c.is_ascii() => f.write_char(c)?,
                        c => match c as u32 {
                            n @ 0...0xff => write!(f, r"\x{:0>2x}", n)?,
                            n @ 0...0xffff => write!(f, r"\u{:0>4x}", n)?,
                            n @ 0...0xffffffff => write!(f, r"\U{:0>8x}", n)?,
                            _ => unreachable!(),
                        },
                    }
                }
                f.write_char('\'')?;
                Ok(())
            }
            PyExpr::Bytes(ref bytes) => {
                f.write_str("b'")?;
                for byte in bytes {
                    match *byte {
                        b'\\' => f.write_str(r"\\")?,
                        b'\r' => f.write_str(r"\r")?,
                        b'\n' => f.write_str(r"\n")?,
                        b'\'' => f.write_str(r"\'")?,
                        b if b.is_ascii() => f.write_char(b.into())?,
                        b => write!(f, r"\x{:0>2x}", b)?,
                    }
                }
                f.write_char('\'')?;
                Ok(())
            }
            PyExpr::Integer(int) => write!(f, "{}", int),
            PyExpr::Float(float) => {
                // Use scientific notation to make this unambiguously a float.
                write!(f, "{:e}", float)
            }
            PyExpr::Complex(real, imag) => {
                // Use scientific notation to make the parts unambiguously floats.
                write!(f, "{:e}{:+e}j", real, imag)
            }
            PyExpr::Tuple(ref tup) => {
                f.write_char('(')?;
                match tup.len() {
                    0 => (),
                    1 => write!(f, "{},", tup[0])?,
                    _ => {
                        write!(f, "{}", tup[0])?;
                        for expr in &tup[1..] {
                            write!(f, ", {}", expr)?;
                        }
                    }
                }
                f.write_char(')')
            }
            PyExpr::List(ref list) => {
                f.write_char('[')?;
                if !list.is_empty() {
                    write!(f, "{}", list[0])?;
                    for expr in &list[1..] {
                        write!(f, ", {}", expr)?;
                    }
                }
                f.write_char(']')
            }
            PyExpr::Dict(ref dict) => {
                f.write_char('{')?;
                if !dict.is_empty() {
                    write!(f, "{}: {}", dict[0].0, dict[0].1)?;
                    for expr in &dict[1..] {
                        write!(f, ", {}: {}", expr.0, expr.1)?;
                    }
                }
                f.write_char('}')
            }
            PyExpr::Set(ref set) => {
                if set.is_empty() {
                    // There is no way to write an empty set literal in Python.
                    return Err(fmt::Error);
                } else {
                    f.write_char('{')?;
                    write!(f, "{}", set[0])?;
                    for expr in &set[1..] {
                        write!(f, ", {}", expr)?;
                    }
                    f.write_char('}')
                }
            }
            PyExpr::Boolean(b) => {
                if b {
                    f.write_str("True")
                } else {
                    f.write_str("False")
                }
            }
            PyExpr::None => f.write_str("None"),
        }
    }
}

fn parse_header(h: Pair<Rule>) -> Result<Header, HeaderParseError> {
    debug_assert_eq!(h.as_rule(), Rule::header);
    let (dict,) = parse_pairs_as!(h.into_inner(), (Rule::dict,));
    let mut type_descriptor: Option<PyExpr> = None;
    let mut fortran_order: Option<bool> = None;
    let mut shape: Option<Vec<usize>> = None;
    for (key, value) in parse_dict(dict)? {
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
