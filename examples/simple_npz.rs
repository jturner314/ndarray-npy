#[macro_use]
extern crate ndarray;
extern crate ndarray_npy;

use ndarray::prelude::*;
use ndarray_npy::{NpzReader, NpzWriter, ReadNpzError, WriteNpzError};
use std::fs::File;

fn write_example() -> Result<(), WriteNpzError> {
    let mut npz = NpzWriter::new(File::create("arrays.npz")?);
    let a: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];
    let b: Array1<i32> = array![7, 8, 9];
    npz.add_array("a", &a)?;
    npz.add_array("b", &b)?;
    Ok(())
}

fn read_example() -> Result<(), ReadNpzError> {
    let mut npz = NpzReader::new(File::open("arrays.npz")?)?;
    let a: Array2<i32> = npz.by_name("a")?;
    let b: Array1<i32> = npz.by_name("b")?;
    println!("a =\n{}", a);
    println!("b =\n{}", b);
    Ok(())
}

fn main() {
    write_example().expect("failure writing arrays to file");
    read_example().expect("failure reading arrays from file");
}
