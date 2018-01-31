extern crate ndarray;
extern crate ndarray_npy;

use ndarray::prelude::*;
use ndarray_npy::NpyExt;

#[test]
fn example_f64() {
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
fn example_f64_fortran() {
    let correct = include_bytes!("example_f64_fortran.npy");
    let mut writer = Vec::<u8>::new();
    let mut arr = Array3::<f64>::zeros((2, 3, 4).f());
    for (i, elem) in arr.iter_mut().enumerate() {
        *elem = i as f64;
    }
    arr.write_npy(&mut writer).unwrap();
    assert_eq!(&correct[..], &writer[..]);
}
