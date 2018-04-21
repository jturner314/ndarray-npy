# ndarray-npy

[![Build Status](https://travis-ci.org/jturner314/ndarray-npy.svg?branch=master)](https://travis-ci.org/jturner314/ndarray-npy)

This crate provides support for reading/writing [`ndarray`]'s `ArrayBase` type
from/to [`.npy`] and [`.npz`] files. See [`src/lib.rs`](src/lib.rs) for more
information.

[`ndarray`]: https://github.com/bluss/rust-ndarray
[`.npy`]: https://docs.scipy.org/doc/numpy/neps/npy-format.html
[`.npz`]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html

**This crate is a work-in-progress.** It currently supports only a subset of
`.npy` header descriptors and supports only primitive numeric types as the
array element type. You can implement `ReadableElement` and `WritableElement`
for your own types, but the next breaking release of this library will probably
change those traits.

Future plans include support for:

* Memory-mapped files.
* More element types (e.g. structs). If you need support for structs before
  this is implemented in `ndarray-npy`, check out the [`npy` crate].
* Possibly merging this with the [`npy` crate].

[`npy` crate]: https://crates.io/crates/npy

## Contributing

Please feel free to create issues and submit PRs. PRs adding more tests would
be especially appreciated.

## License

Copyright 2018 Jim Turner

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE) or the [MIT
license](LICENSE-MIT), at your option. You may not use this project except in
compliance with those terms.
