//! Miscellaneous example cases.

use memmap2::{Mmap, MmapMut};
use ndarray::prelude::*;
use ndarray::Slice;
use ndarray_npy::{
    write_zeroed_npy, ReadNpyError, ReadNpyExt, ViewMutNpyExt, ViewNpyError, ViewNpyExt,
    WriteNpyExt,
};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem;

#[test]
fn write_f64_standard() {
    #[cfg(target_endian = "little")]
    let path = "resources/example_f64_little_endian_standard.npy";
    #[cfg(target_endian = "big")]
    let path = "resources/example_f64_big_endian_standard.npy";

    let correct = fs::read(path).unwrap();
    let mut writer = Vec::<u8>::new();
    let mut arr = Array3::<f64>::zeros((2, 3, 4));
    for (i, elem) in arr.iter_mut().enumerate() {
        *elem = i as f64;
    }
    arr.write_npy(&mut writer).unwrap();
    assert_eq!(&correct, &writer);
}

#[test]
fn write_f64_fortran() {
    #[cfg(target_endian = "little")]
    let path = "resources/example_f64_little_endian_fortran.npy";
    #[cfg(target_endian = "big")]
    let path = "resources/example_f64_big_endian_fortran.npy";

    let correct = fs::read(path).unwrap();
    let mut writer = Vec::<u8>::new();
    let mut arr = Array3::<f64>::zeros((2, 3, 4).f());
    for (i, elem) in arr.iter_mut().enumerate() {
        *elem = i as f64;
    }
    arr.write_npy(&mut writer).unwrap();
    assert_eq!(&correct[..], &writer[..]);
}

#[test]
fn write_f64_discontiguous() {
    #[cfg(target_endian = "little")]
    let path = "resources/example_f64_little_endian_standard.npy";
    #[cfg(target_endian = "big")]
    let path = "resources/example_f64_big_endian_standard.npy";

    let correct = fs::read(path).unwrap();
    let mut writer = Vec::<u8>::new();
    let mut arr = Array3::<f64>::zeros((3, 4, 4));
    arr.slice_axis_inplace(Axis(1), Slice::new(0, None, 2));
    arr.swap_axes(0, 1);
    for (i, elem) in arr.iter_mut().enumerate() {
        *elem = i as f64;
    }
    arr.write_npy(&mut writer).unwrap();
    assert_eq!(&correct, &writer);
}

#[test]
fn read_f64_standard() {
    let mut correct = Array3::<f64>::zeros((2, 3, 4));
    for (i, elem) in correct.iter_mut().enumerate() {
        *elem = i as f64;
    }
    for path in &[
        "resources/example_f64_little_endian_standard.npy",
        "resources/example_f64_big_endian_standard.npy",
    ] {
        let file = File::open(path).unwrap();
        let arr = Array3::<f64>::read_npy(file).unwrap();
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
    for path in &[
        "resources/example_f64_little_endian_fortran.npy",
        "resources/example_f64_big_endian_fortran.npy",
    ] {
        let file = File::open(path).unwrap();
        let arr = Array3::<f64>::read_npy(file).unwrap();
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
    let file = File::open("resources/example_bool_standard.npy").unwrap();
    let arr = Array3::<bool>::read_npy(file).unwrap();
    assert_eq!(correct, arr);
}

#[test]
fn read_bool_bad_value() {
    let file = File::open("resources/example_bool_bad_value.npy").unwrap();
    assert!(matches!(
        Array3::<bool>::read_npy(file),
        Err(ReadNpyError::ParseData(_))
    ));
}

#[test]
fn view_f64_standard() {
    #[cfg(target_endian = "little")]
    let path = "resources/example_f64_little_endian_standard.npy";
    #[cfg(target_endian = "big")]
    let path = "resources/example_f64_big_endian_standard.npy";

    let mut correct = Array3::<f64>::zeros((2, 3, 4));
    for (i, elem) in correct.iter_mut().enumerate() {
        *elem = i as f64;
    }
    let file = File::open(path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let view = ArrayView3::<f64>::view_npy(&mmap).unwrap();
    assert_eq!(correct, view);
    assert!(view.is_standard_layout());
}

#[test]
fn view_f64_fortran() {
    #[cfg(target_endian = "little")]
    let path = "resources/example_f64_little_endian_fortran.npy";
    #[cfg(target_endian = "big")]
    let path = "resources/example_f64_big_endian_fortran.npy";

    let mut correct = Array3::<f64>::zeros((2, 3, 4));
    for (i, elem) in correct.iter_mut().enumerate() {
        *elem = i as f64;
    }
    let file = File::open(path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let view = ArrayView3::<f64>::view_npy(&mmap).unwrap();
    assert_eq!(correct, view);
    assert!(view.t().is_standard_layout());
}

#[test]
fn view_bool() {
    let mut correct = Array3::from_elem((2, 3, 4), false);
    for (i, elem) in correct.iter_mut().enumerate() {
        *elem = (i % 5) % 2 == 0;
    }
    let file = File::open("resources/example_bool_standard.npy").unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let view = ArrayView3::<bool>::view_npy(&mmap).unwrap();
    assert_eq!(correct, view);
    assert!(view.is_standard_layout());
}

#[test]
fn view_bool_bad_value() {
    let file = File::open("resources/example_bool_bad_value.npy").unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    assert!(matches!(
        ArrayView3::<bool>::view_npy(&mmap),
        Err(ViewNpyError::InvalidData(_))
    ));
}

#[test]
fn view_mut_f64_standard() {
    #[cfg(target_endian = "little")]
    let path = "resources/example_f64_little_endian_standard.npy";
    #[cfg(target_endian = "big")]
    let path = "resources/example_f64_big_endian_standard.npy";

    let mut correct = Array3::<f64>::zeros((2, 3, 4));
    for (i, elem) in correct.iter_mut().enumerate() {
        *elem = i as f64;
    }
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
    let view = ArrayViewMut3::<f64>::view_mut_npy(&mut mmap).unwrap();
    assert_eq!(correct, view);
    assert!(view.is_standard_layout());
}

#[test]
fn view_mut_f64_fortran() {
    #[cfg(target_endian = "little")]
    let path = "resources/example_f64_little_endian_fortran.npy";
    #[cfg(target_endian = "big")]
    let path = "resources/example_f64_big_endian_fortran.npy";

    let mut correct = Array3::<f64>::zeros((2, 3, 4));
    for (i, elem) in correct.iter_mut().enumerate() {
        *elem = i as f64;
    }
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
    let view = ArrayViewMut3::<f64>::view_mut_npy(&mut mmap).unwrap();
    assert_eq!(correct, view);
    assert!(view.t().is_standard_layout());
}

#[test]
fn view_mut_bool() {
    let mut correct = Array3::from_elem((2, 3, 4), false);
    for (i, elem) in correct.iter_mut().enumerate() {
        *elem = (i % 5) % 2 == 0;
    }
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("resources/example_bool_standard.npy")
        .unwrap();
    let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
    let view = ArrayViewMut3::<bool>::view_mut_npy(&mut mmap).unwrap();
    assert_eq!(correct, view);
    assert!(view.is_standard_layout());
}

#[test]
fn view_mut_bool_bad_value() {
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("resources/example_bool_bad_value.npy")
        .unwrap();
    let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
    assert!(matches!(
        ArrayViewMut3::<bool>::view_mut_npy(&mut mmap),
        Err(ViewNpyError::InvalidData(_))
    ));
}

#[test]
fn misaligned() {
    let mut written = Vec::<u8>::new();
    let mut arr = Array3::<f64>::zeros((2, 3, 4));
    for (i, elem) in arr.iter_mut().enumerate() {
        *elem = i as f64;
    }
    arr.write_npy(&mut written).unwrap();

    const ADJUSTMENT: usize = 1;
    let mut maybe_misaligned = Vec::<u8>::with_capacity(written.len() + ADJUSTMENT);
    if maybe_misaligned.as_ptr() as usize % mem::align_of::<f64>() == 0 {
        maybe_misaligned.resize(ADJUSTMENT, 0);
    }
    let start = maybe_misaligned.len();
    maybe_misaligned.extend_from_slice(&written);
    let misaligned = &mut maybe_misaligned[start..start + written.len()];
    debug_assert_ne!(0, misaligned.as_ptr() as usize % mem::align_of::<f64>());

    assert!(matches!(
        ArrayView3::<f64>::view_npy(&misaligned[..]),
        Err(ViewNpyError::MisalignedData)
    ));
    assert!(matches!(
        ArrayViewMut3::<f64>::view_mut_npy(&mut misaligned[..]),
        Err(ViewNpyError::MisalignedData)
    ));
}

#[test]
fn zeroed() {
    const EXISTING_DATA: &[u8] = b"hello";
    const SHAPE: [usize; 3] = [3, 4, 5];

    // Create a file with some existing data.
    let mut file: File = tempfile::tempfile().unwrap();
    file.write_all(EXISTING_DATA).unwrap();

    // Write `.npy` file with zeroed data.
    write_zeroed_npy::<i32, _>(&file, SHAPE).unwrap();

    // Reset cursor and verify EXISTING_DATA is still there.
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut buf = [0; EXISTING_DATA.len()];
    file.read_exact(&mut buf).unwrap();
    assert_eq!(EXISTING_DATA, buf);

    // Read and verify the `.npy` file.
    let arr = Array3::<i32>::read_npy(file).unwrap();
    assert_eq!(arr, Array3::<i32>::zeros(SHAPE));
    assert!(arr.is_standard_layout());
}
