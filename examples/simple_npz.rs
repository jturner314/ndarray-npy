#[macro_use]
extern crate ndarray;
extern crate ndarray_npy;

use ndarray_npy::{NpzError, NpzWriter};
use std::fs::File;

fn example() -> Result<(), NpzError> {
    let mut npz = NpzWriter::new(File::create("arrays.npz")?);
    npz.add_array("a", array![[1, 2, 3], [4, 5, 6]])?;
    npz.add_array("b", array![7, 8, 9])?;
    Ok(())
}

fn main () {
    example().expect("failure writing arrays to file");
}
