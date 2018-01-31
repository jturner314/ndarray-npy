#[macro_use]
extern crate ndarray;
extern crate ndarray_npy;

use ndarray_npy::NpyExt;
use std::fs::File;

fn example() -> std::io::Result<()> {
    let arr = array![[1, 2, 3], [4, 5, 6]];
    let mut file = File::create("array.npy")?;
    arr.write_npy(&mut file)?;
    Ok(())
}

fn main () {
    example().expect("failure writing array to file");
}
