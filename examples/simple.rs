#[macro_use]
extern crate ndarray;
extern crate ndarray_npy;

use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyExt};
use std::fs::File;

fn write_example() -> std::io::Result<()> {
    let arr: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];
    let writer = File::create("array.npy")?;
    arr.write_npy(writer)?;
    Ok(())
}

fn read_example() -> Result<(), ReadNpyError> {
    let reader = File::open("array.npy")?;
    let arr = Array2::<i32>::read_npy(reader)?;
    println!("arr =\n{}", arr);
    Ok(())
}

fn main() {
    write_example().expect("failure writing array to file");
    read_example().expect("failure reading array from file");
}
