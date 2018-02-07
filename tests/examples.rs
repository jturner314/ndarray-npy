extern crate ndarray;
extern crate ndarray_npy;

use ndarray::prelude::*;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::io::Cursor;

#[test]
fn write_f64() {
    let correct = include_bytes!("example_f64.npy");
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
    let correct = include_bytes!("example_f64_fortran.npy");
    let mut writer = Vec::<u8>::new();
    let mut arr = Array3::<f64>::zeros((2, 3, 4).f());
    for (i, elem) in arr.iter_mut().enumerate() {
        *elem = i as f64;
    }
    arr.write_npy(&mut writer).unwrap();
    assert_eq!(&correct[..], &writer[..]);
}

#[test]
fn read_f64() {
    let mut correct = Array3::<f64>::zeros((2, 3, 4));
    for (i, elem) in correct.iter_mut().enumerate() {
        *elem = i as f64;
    }
    let reader = Cursor::new(&include_bytes!("example_f64.npy")[..]);
    let arr = Array3::<f64>::read_npy(reader).unwrap();
    assert_eq!(correct, arr);
}

#[test]
fn read_f64_fortran() {
    let mut correct = Array3::<f64>::zeros((2, 3, 4));
    for (i, elem) in correct.iter_mut().enumerate() {
        *elem = i as f64;
    }
    let reader = Cursor::new(&include_bytes!("example_f64_fortran.npy")[..]);
    let arr = Array3::<f64>::read_npy(reader).unwrap();
    assert_eq!(correct, arr);
}
