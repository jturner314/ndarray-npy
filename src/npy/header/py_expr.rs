use pest::iterators::Pair;
use std::fmt::{self, Write};
use super::{HeaderParseError, Rule};

pub fn parse_string_escape_seq(escape_seq: Pair<Rule>) -> Result<char, HeaderParseError> {
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

pub fn parse_string(string: Pair<Rule>) -> Result<String, HeaderParseError> {
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

pub fn parse_number_expr(expr: Pair<Rule>) -> PyExpr {
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

pub fn parse_number(number: Pair<Rule>) -> PyExpr {
    debug_assert_eq!(number.as_rule(), Rule::number);
    let (inner,) = parse_pairs_as!(number.into_inner(), (_,));
    match inner.as_rule() {
        Rule::imag => parse_imag(inner),
        Rule::float => PyExpr::Float(parse_float(inner)),
        Rule::integer => PyExpr::Integer(parse_integer(inner)),
        _ => unreachable!(),
    }
}

pub fn parse_integer(int: Pair<Rule>) -> i64 {
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

pub fn parse_float(float: Pair<Rule>) -> f64 {
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

pub fn parse_imag(imag: Pair<Rule>) -> PyExpr {
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
pub fn parse_seq(seq: Pair<Rule>) -> Result<Vec<PyExpr>, HeaderParseError> {
    debug_assert!([Rule::tuple, Rule::list, Rule::set].contains(&seq.as_rule()));
    seq.into_inner().map(|elem| parse_expr(elem)).collect()
}

pub fn parse_dict(dict: Pair<Rule>) -> Result<Vec<(PyExpr, PyExpr)>, HeaderParseError> {
    debug_assert_eq!(dict.as_rule(), Rule::dict);
    let mut out = Vec::new();
    for elem in dict.into_inner() {
        let (key, value) = parse_pairs_as!(elem.into_inner(), (Rule::expr, Rule::expr));
        out.push((parse_expr(key)?, parse_expr(value)?));
    }
    Ok(out)
}

pub fn parse_boolean(b: Pair<Rule>) -> bool {
    debug_assert_eq!(b.as_rule(), Rule::boolean);
    match b.as_str() {
        "True" => true,
        "False" => false,
        _ => unreachable!(),
    }
}

/// NumPy uses [`ast.literal_eval()`] to parse the header dictionary.
/// `literal_eval()` supports only the following Python literals: strings,
/// bytes, numbers, tuples, lists, dicts, sets, booleans, and `None`.
///
/// [`ast.literal_eval()`]: https://docs.python.org/3/library/ast.html#ast.literal_eval
pub fn parse_expr(expr: Pair<Rule>) -> Result<PyExpr, HeaderParseError> {
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
        Rule::none => Ok(PyExpr::None),
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
