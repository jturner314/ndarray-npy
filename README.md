# ndarray-npy

[![Build status](https://travis-ci.org/jturner314/ndarray-npy.svg?branch=master)](https://travis-ci.org/jturner314/ndarray-npy)
[![Coverage](https://codecov.io/gh/jturner314/ndarray-npy/branch/master/graph/badge.svg)](https://codecov.io/gh/jturner314/ndarray-npy)
[![Dependencies status](https://deps.rs/repo/github/jturner314/ndarray-npy/status.svg)](https://deps.rs/repo/github/jturner314/ndarray-npy)
[![Crate](https://img.shields.io/crates/v/ndarray-npy.svg)](https://crates.io/crates/ndarray-npy)
[![Documentation](https://docs.rs/ndarray-npy/badge.svg)](https://docs.rs/ndarray-npy)
[![Chat](https://img.shields.io/badge/chat-%23rust--sci%20on%20OFTC-lightgrey.svg)](https://webchat.oftc.net/?channels=rust-sci)

This crate provides support for reading/writing [`ndarray`]'s `ArrayBase` type
from/to [`.npy`] and [`.npz`] files. See the
[documentation](https://docs.rs/ndarray-npy) for more information.

[`ndarray`]: https://github.com/bluss/ndarray
[`.npy`]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html
[`.npz`]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html

**This crate is a work-in-progress.** It currently supports only a subset of
`.npy` header descriptors and supports only primitive fixed-size integer,
floating point, and `bool` types as the array element type. You can implement
`ReadableElement` and `WritableElement` for your own types, but the next
breaking release of this library will probably change those traits.

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
ndarray-npy = "0.5"
```

The `default` feature set includes the `compressed_npz` feature, which enables
support for uncompresssed and compressed `.npz` files. This requires a
dependency on the [`zip` crate] and a compression backend crate.

To use without the default features:

```toml
[dependencies]
ndarray-npy = { version = "0.5", default-features = false }
```

With `default-features = false`, `ndarray-npy` provides support only for `.npy`
files, not `.npz` files. If you want `.npz` file support, you can select
additional features:

* `npz` enables support for uncompressed `.npz` files. This requires a
  dependency on the [`zip` crate].
* `compressed_npz` enables support for uncompressed and compressed `.npz`
  files. This requires a dependency on the [`zip` crate] and a compression
  backend crate.

For example, you can use just the `npz` feature:

```toml
[dependencies.ndarray-npy]
version = "0.5"
default-features = false
features = ["npz"]
```

[`zip` crate]: https://crates.io/crates/zip

### Library authors

Library authors should specify their dependency on `ndarray-npy` like this:

```toml
[dependencies.ndarray-npy]
version = "0.5"
default-features = false
features = [FEATURES_LIST_HERE]
```

where the `features` list is one of the following:

* `[]` if your crate does not depend on `.npz` file support
* `["npz"]` if your crate depends on `.npz` file support but not compression
* `["compressed_npz"]` if your crate depends on `.npz` file support with compression

## Releases

* **0.5.0**

  * Updated to `ndarray` 0.13.
  * Updated to Rust 1.38.
  * Added `read_npy` and `write_npy` convenience functions.
  * Added support for `npy` format version 3.0.
  * Renamed `ReadableElement::read_vec` to `::read_to_end_exact_vec`.
  * Refactored the error types and variants, including removing the associated
    `Error` type from `Readable/WritableElement` and updating to the new style
    of `std::error::Error` implementation.
  * Updated the padding calculation to make the total header length be
    divisible by 64 instead of just 16 when writing files. (See
    [numpy/numpy#9025](https://github.com/numpy/numpy/pull/9025).)
  * Fixed determination of file format version when the addition of padding
    changes the required version when writing the file.
  * Fixed miscellaneous bugs related to overflow and error handling.

* **0.4.0**

  * Added support for reading/writing arrays of `bool`, by @tobni and
    @jturner314.
  * Updated to `zip` 0.5.
  * Updated to Rust 1.32.
  * Renamed the `compressed_npz_default` feature to `compressed_npz` because
    the `zip` crate no longer allows the user to select the compression
    backend.

* **0.3.0**

  * Updated to `ndarray` 0.12.
  * Updated to `num-traits` 0.2 (replacing dependency on `num`).
  * Updated to `py_literal` 0.2.

* **0.2.0**

  * Updated to `zip` 0.4.
  * Made the compression backend for compressed `.npz` files user-selectable.
  * Reworked how feature selection works. This should only affect you if you
    use `default-features = false, features = ["compressed_npz"]`.

* **0.1.1**

  * Improved crate documentation (no functional changes).

* **0.1.0**

  * Initial release.

## Contributing

Please feel free to create issues and submit PRs. PRs adding more tests would
be especially appreciated.

## License

Copyright 2018–2019 Jim Turner

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE), or the [MIT
license](LICENSE-MIT), at your option. You may not use this project except in
compliance with those terms.
