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
                HeaderParseError::IllegalEscapeSequence(format!(
                    "Octal escape is invalid: \\{}",
                    seq.as_str()
                ))
            }),
        Rule::hex_escape | Rule::unicode_hex_escape => ::std::char::from_u32(
            u32::from_str_radix(&seq.as_str()[1..], 16).unwrap(),
        ).ok_or_else(|| {
            HeaderParseError::IllegalEscapeSequence(format!(
                "Hex escape is invalid: \\x{}",
                seq.as_str()
            ))
        }),
        Rule::name_escape => Err(HeaderParseError::IllegalEscapeSequence(
            "Unicode name escapes are not supported.".into(),
        )),
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
                    Rule::short_string_non_escape
                    | Rule::long_string_non_escape
                    | Rule::string_unknown_escape => out.push_str(item.as_str()),
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

pub fn parse_bytes_escape_seq(escape_seq: Pair<Rule>) -> Result<u8, HeaderParseError> {
    debug_assert_eq!(escape_seq.as_rule(), Rule::bytes_escape_seq);
    let (seq,) = parse_pairs_as!(escape_seq.into_inner(), (_,));
    match seq.as_rule() {
        Rule::char_escape => Ok(match seq.as_str() {
            "\\" => b'\\',
            "'" => b'\'',
            "\"" => b'"',
            "a" => b'\x07',
            "b" => b'\x08',
            "f" => b'\x0C',
            "n" => b'\n',
            "r" => b'\r',
            "t" => b'\t',
            "v" => b'\x0B',
            _ => unreachable!(),
        }),
        Rule::octal_escape => u8::from_str_radix(seq.as_str(), 8).map_err(|err| {
            HeaderParseError::IllegalEscapeSequence(format!(
                "failed to parse \\{} as u8: {}",
                seq.as_str(),
                err,
            ))
        }),
        Rule::hex_escape => Ok(u8::from_str_radix(&seq.as_str()[1..], 16).unwrap()),
        _ => unreachable!(),
    }
}

pub fn parse_bytes(bytes: Pair<Rule>) -> Result<Vec<u8>, HeaderParseError> {
    debug_assert_eq!(bytes.as_rule(), Rule::bytes);
    let (bytes_body,) = parse_pairs_as!(bytes.into_inner(), (_,));
    match bytes_body.as_rule() {
        Rule::short_bytes_body | Rule::long_bytes_body => {
            let mut out = Vec::new();
            for item in bytes_body.into_inner() {
                match item.as_rule() {
                    Rule::short_bytes_non_escape
                    | Rule::long_bytes_non_escape
                    | Rule::bytes_unknown_escape => out.extend_from_slice(item.as_str().as_bytes()),
                    Rule::line_continuation_seq => (),
                    Rule::bytes_escape_seq => out.push(parse_bytes_escape_seq(item)?),
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
        Rule::bytes => Ok(PyExpr::Bytes(parse_bytes(inner)?)),
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

#[cfg(test)]
mod test {
    use pest::Parser;
    use super::super::HeaderParser;
    use super::*;

    #[test]
    fn parse_string_example() {
        for &(input, correct) in &[
            ("''", ""),
            (
                r"'he\qllo\th\03o\x1bw\
are\a\'y\u1234o\U00031234u'",
                "he\\qllo\th\x03o\x1bware\x07'y\u{1234}o\u{31234}u",
            ),
        ] {
            let mut parsed = HeaderParser::parse(Rule::string, input)
                .unwrap_or_else(|err| panic!("failed to parse: {}", err));
            let s = parse_string(parse_pairs_as!(parsed, (Rule::string,)).0).unwrap();
            assert_eq!(s, correct);
        }
    }

    #[test]
    fn parse_bytes_example() {
        for &(input, correct) in &[
            ("b''", &b""[..]),
            (
                r"b'he\qllo\th\03o\x1bw\
are\a\'y\u1234o\U00031234u'",
                &b"he\\qllo\th\x03o\x1bware\x07'y\\u1234o\\U00031234u"[..],
            ),
        ] {
            let mut parsed = HeaderParser::parse(Rule::bytes, input)
                .unwrap_or_else(|err| panic!("failed to parse: {}", err));
            let bytes = parse_bytes(parse_pairs_as!(parsed, (Rule::bytes,)).0).unwrap();
            assert_eq!(bytes, correct);
        }
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
