//! Regression tests for allocations driven by untrusted values in the file.
//!
//! The header of an `.npy` file declares both the header length and the array
//! shape. These values are controlled by whoever wrote the file, so a very
//! small file can ask for a very large amount of memory (CWE-770). When such an
//! allocation fails, the default Rust behavior is to abort the process, which a
//! caller has no way to intercept.
//!
//! This crate limits that in two different ways, and this module checks both:
//!
//! * The header itself is read with a bounded `Read::take`, so the allocation
//!   tracks the bytes actually present in the file rather than the declared
//!   header length. This is asserted directly by tracking allocation sizes.
//!
//! * The data allocation is made with `Vec::try_reserve_exact`, so an
//!   allocation the system cannot satisfy is returned to the caller as an error
//!   instead of aborting. A huge allocation is still *attempted* (this is
//!   deliberate — see the `# Panics` docs of `ReadNpyExt::read_npy`), so this
//!   is asserted as "returns `Err`" rather than as a bound on the requested
//!   size. Note that if this ever regresses to an abort, the whole test process
//!   dies, which is itself the signal that the fix has been lost.
//!
//! Note: all the checks live in a single `#[test]` function because the
//! allocation tracker is process-global, so the checks must not run
//! concurrently with each other.

use ndarray::prelude::*;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Records the size of the largest single allocation request so that the tests
/// can assert the allocation is bounded by the real input size.
struct TrackingAlloc;

static MAX_REQUEST: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        MAX_REQUEST.fetch_max(layout.size(), Ordering::SeqCst);
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        MAX_REQUEST.fetch_max(layout.size(), Ordering::SeqCst);
        System.alloc_zeroed(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        MAX_REQUEST.fetch_max(new_size, Ordering::SeqCst);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOC: TrackingAlloc = TrackingAlloc;

fn max_request() -> usize {
    MAX_REQUEST.load(Ordering::SeqCst)
}

fn reset_max_request() {
    MAX_REQUEST.store(0, Ordering::SeqCst);
}

/// Builds a version 2.0 file whose declared header length (which may be up to
/// 4 GiB) is far larger than the number of bytes actually present.
fn file_with_declared_header_len(declared_len: u32) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(b"\x93NUMPY");
    v.extend_from_slice(&[2, 0]);
    v.extend_from_slice(&declared_len.to_le_bytes());
    v.extend_from_slice(b"xxxxxxxx"); // far fewer bytes than declared
    v
}

/// Builds a version 1.0 file with a valid header whose declared shape implies
/// far more data than the file contains.
fn file_with_declared_shape(shape: &str) -> Vec<u8> {
    let header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': {}, }}",
        shape
    );
    // The header is padded so that the total length is a multiple of 64.
    let prefix_len = 6 + 2 + 2; // magic string + version + `u16` header length
    let unpadded = prefix_len + header.len() + 1;
    let pad = 64 - (unpadded % 64);
    let header_len = header.len() + pad + 1; // including the trailing newline

    let mut v = Vec::new();
    v.extend_from_slice(b"\x93NUMPY");
    v.extend_from_slice(&[1, 0]);
    v.extend_from_slice(&(header_len as u16).to_le_bytes());
    v.extend_from_slice(header.as_bytes());
    v.extend(std::iter::repeat(b' ').take(pad));
    v.push(b'\n');
    v
}

/// The declared header length must not drive allocation, and a declared shape
/// that cannot be allocated must be reported as an error rather than aborting
/// the process.
#[test]
fn untrusted_len_does_not_abort_the_process() {
    // A file declaring a 64 MiB header, but containing only 8 bytes of it.
    let file = file_with_declared_header_len(64 * 1024 * 1024);
    reset_max_request();
    let res = Array1::<f64>::read_npy(Cursor::new(&file));
    assert!(res.is_err(), "a truncated file should fail to parse");
    assert!(
        max_request() < 1024 * 1024,
        "the declared header length should not drive allocation: a file of \
         {} bytes caused a single allocation of {} bytes",
        file.len(),
        max_request()
    );

    // Files declaring shapes far larger than anything this machine can
    // allocate. Each must come back as an error. The sizes are chosen so that
    // they fail for a reason that does not depend on how much memory the test
    // machine happens to have: the first is well beyond any real address space,
    // the second overflows `usize` altogether once multiplied by the element
    // size.
    for shape in ["(1000000000000000,)", "(1000000000000000000000,)"] {
        let file = file_with_declared_shape(shape);
        reset_max_request();
        let res = Array1::<f64>::read_npy(Cursor::new(&file));
        assert!(
            res.is_err(),
            "a declared shape of {} should be reported as an error rather than \
             aborting the process",
            shape
        );
    }

    // Sanity check: a well-formed file still round-trips.
    let arr = Array1::<f64>::from_vec((0..1000).map(|i| i as f64).collect());
    let mut buf = Vec::new();
    arr.write_npy(&mut buf).unwrap();

    reset_max_request();
    let read: Array1<f64> = Array1::<f64>::read_npy(Cursor::new(&buf)).unwrap();
    assert_eq!(read, arr);
    // 1000 `f64`s are 8000 bytes; allow generous slack for the header and for
    // `Vec` growth.
    assert!(
        max_request() < 1024 * 1024,
        "unexpectedly large allocation of {} bytes for a {} byte file",
        max_request(),
        buf.len()
    );
}
