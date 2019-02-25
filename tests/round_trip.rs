extern crate ndarray;
extern crate ndarray_npy;

use ndarray::prelude::*;
use ndarray::{array, Data};
use ndarray_npy::{ReadNpyExt, ReadableElement, WritableElement, WriteNpyExt};
use std::fmt::Debug;

/// Tests that the writing the array as an `.npy` file and reading it
/// back again doesn't change the shape or data.
fn test_round_trip<A, S, D>(before: &ArrayBase<S, D>)
where
    A: Debug + PartialEq + ReadableElement + WritableElement,
    S: Data<Elem = A>,
    D: Dimension,
{
    let mut npy = Vec::<u8>::new();
    before.write_npy(&mut npy).unwrap();
    let after = Array::<A, D>::read_npy(&npy[..]).unwrap();
    assert_eq!(before, &after);
}

#[test]
fn round_trip_i32() {
    test_round_trip(&array![
        [[1i32, 8], [-3, 4], [2, 9]],
        [[-5, 0], [7, 38], [-4, 1]]
    ]);
}

#[test]
fn round_trip_f32() {
    test_round_trip(&array![
        [[3f32, -1.4], [-159., 26.], [5., -3.5]],
        [[-89.7, 93.], [2., 384.], [-626.4, 3.]],
    ]);
}

#[test]
fn round_trip_f64() {
    test_round_trip(&array![
        [2.7, -40.4, -23., 27.8, -49., -43.3],
        [-25.2, 11.8, -8.9, -17.8, 36.4, -25.6],
    ]);
}

#[test]
fn round_trip_bool() {
    test_round_trip(&array![
        [[true], [true], [false]],
        [[false], [true], [false]]
    ]);
}
