# ndarray-npy

[![Build status](https://travis-ci.org/jturner314/ndarray-npy.svg?branch=master)](https://travis-ci.org/jturner314/ndarray-npy)
[![Dependencies status](https://deps.rs/repo/github/jturner314/ndarray-npy/status.svg)](https://deps.rs/repo/github/jturner314/ndarray-npy)
[![Crate](https://img.shields.io/crates/v/ndarray-npy.svg)](https://crates.io/crates/ndarray-npy)
[![Documentation](https://docs.rs/ndarray-npy/badge.svg)](https://docs.rs/ndarray-npy)

This crate provides support for reading/writing [`ndarray`]'s `ArrayBase` type
from/to [`.npy`] and [`.npz`] files. See the
[documentation](https://docs.rs/ndarray-npy) for more information.

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

## Using with Cargo

To use with the default features:

```toml
[dependencies]
ndarray-npy = "0.1"
```

To use without the default features:

```toml
[dependencies]
ndarray-npy = { version = "0.1", default-features = false }
```

There are two optional features that are enabled by default:

* `npz` enables support for `.npz` files, which requires a dependency on the
  [`zip` crate].
* `compressed_npz` enables support for compressed `.npz` files, which requires
  a dependency on the [`zip` crate] and also pulls in the necessary
  dependencies for the `zip` crate's `deflate` feature.

For example, you can use just the `npz` feature:

```toml
[dependencies.ndarray-npy]
version = "0.1"
default-features = false
features = ["npz"]
```

[`zip` crate]: https://crates.io/crates/zip

## Contributing

Please feel free to create issues and submit PRs. PRs adding more tests would
be especially appreciated.

## License

Copyright 2018 Jim Turner

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE) or the [MIT
license](LICENSE-MIT), at your option. You may not use this project except in
compliance with those terms.
