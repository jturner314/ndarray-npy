use ndarray::prelude::*;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::io::Cursor;

#[test]
fn write_f64_standard() {
    #[cfg(target_endian = "little")]
    let correct = include_bytes!("example_f64_little_endian_standard.npy");
    #[cfg(target_endian = "big")]
    let correct = include_bytes!("example_f64_big_endian_standard.npy");

    let mut writer = Vec::<u8>::new();
    let mut arr = Array3::<f64>::zeros((2, 3, 4));
    for (i, elem) in arr.iter_mut().enumerate() {
        *elem = i as f64;
    }
    arr.write_npy(&mut writer).unwrap();
    assert_eq!(&correct[..], &writer[..]);
}

#[test]
fn write_f64_fortran() {
    #[cfg(target_endian = "little")]
    let correct = include_bytes!("example_f64_little_endian_fortran.npy");
    #[cfg(target_endian = "big")]
    let correct = include_bytes!("example_f64_big_endian_fortran.npy");

    let mut writer = Vec::<u8>::new();
    let mut arr = Array3::<f64>::zeros((2, 3, 4).f());
    for (i, elem) in arr.iter_mut().enumerate() {
        *elem = i as f64;
    }
    arr.write_npy(&mut writer).unwrap();
    assert_eq!(&correct[..], &writer[..]);
}

#[test]
fn read_f64_standard() {
    let mut correct = Array3::<f64>::zeros((2, 3, 4));
    for (i, elem) in correct.iter_mut().enumerate() {
        *elem = i as f64;
    }
    for &bytes in &[
        &include_bytes!("example_f64_little_endian_standard.npy")[..],
        &include_bytes!("example_f64_big_endian_standard.npy")[..],
    ] {
        let reader = Cursor::new(bytes);
        let arr = Array3::<f64>::read_npy(reader).unwrap();
        assert_eq!(correct, arr);
        assert!(arr.is_standard_layout());
    }
}

#[test]
fn read_f64_fortran() {
    let mut correct = Array3::<f64>::zeros((2, 3, 4).f());
    for (i, elem) in correct.iter_mut().enumerate() {
        *elem = i as f64;
    }
    for &bytes in &[
        &include_bytes!("example_f64_little_endian_fortran.npy")[..],
        &include_bytes!("example_f64_big_endian_fortran.npy")[..],
    ] {
        let reader = Cursor::new(bytes);
        let arr = Array3::<f64>::read_npy(reader).unwrap();
        assert_eq!(correct, arr);
        assert!(arr.t().is_standard_layout());
    }
}

#[test]
fn read_bool() {
    let mut correct = Array3::from_elem((2, 3, 4), false);
    for (i, elem) in correct.iter_mut().enumerate() {
        *elem = (i % 5) % 2 == 0;
    }
    let reader = Cursor::new(&include_bytes!("example_bool_standard.npy")[..]);
    let arr = Array3::<bool>::read_npy(reader).unwrap();
    assert_eq!(correct, arr);
}

#[test]
fn read_bool_bad_value() {
    let reader = Cursor::new(&include_bytes!("example_bool_bad_value.npy")[..]);
    assert!(Array3::<bool>::read_npy(reader).is_err());
}
