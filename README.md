# ndarray-npy

[![Build status](https://travis-ci.org/jturner314/ndarray-npy.svg?branch=master)](https://travis-ci.org/jturner314/ndarray-npy)
[![Coverage](https://codecov.io/gh/jturner314/ndarray-npy/branch/master/graph/badge.svg)](https://codecov.io/gh/jturner314/ndarray-npy)
[![Dependencies status](https://deps.rs/repo/github/jturner314/ndarray-npy/status.svg)](https://deps.rs/repo/github/jturner314/ndarray-npy)
[![Crate](https://img.shields.io/crates/v/ndarray-npy.svg)](https://crates.io/crates/ndarray-npy)
[![Documentation](https://docs.rs/ndarray-npy/badge.svg)](https://docs.rs/ndarray-npy)

This crate provides support for reading/writing [`ndarray`]'s `ArrayBase` type
from/to [`.npy`] and [`.npz`] files. See the
[documentation](https://docs.rs/ndarray-npy) for more information.

[`ndarray`]: https://github.com/bluss/ndarray
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
ndarray-npy = "0.2"
```

The `default` feature set includes the `compressed_npz_default` feature, which
enables support for uncompresssed and compressed `.npz` files with the default
compression backend. This requires a dependency on the [`zip` crate] and
[`flate2` crate].

To use without the default features:

```toml
[dependencies]
ndarray-npy = { version = "0.2", default-features = false }
```

With `default-features = false`, `ndarray-npy` provides support only for `.npy`
files, not `.npz` files. If you want `.npz` file support, you can select
additional features:

* `npz` enables support for uncompressed `.npz` files. This requires a
  dependency on the [`zip` crate].
* `compressed_npz` enables support for uncompressed and compressed `.npz` files
  without selecting a `flate2` backend. This requires a dependency on the
  [`zip` crate] and [`flate2` crate]. If you use `default-features = false` and
  enable the `compressed_npz` feature, you must select a `flate2` backend (see
  example below).

For example, you can use just the `npz` feature:

```toml
[dependencies.ndarray-npy]
version = "0.2"
default-features = false
features = ["npz"]
```

You can use the `npz_compressed` feature with a non-default `flate2` backend.
Note that the version of `flate2` must match the version of `flate2` used by
the `zip` crate for this to work. This example shows selecting the `zlib`
backend:

```toml
[dependencies.ndarray-npy]
version = "0.2"
default-features = false
features = ["compressed_npz"]

[dependencies.flate2]
version = "1.0"
default-features = false
features = ["zlib"]
```

[`zip` crate]: https://crates.io/crates/zip
[`flate2` crate]: https://crates.io/crates/flate2

### Library authors

Library authors should specify their dependency on `ndarray-npy` like this:

```toml
[dependencies.ndarray-npy]
version = "0.2"
default-features = false
features = [FEATURES_LIST_HERE]
```

where the `features` list is one of the following:

* `[]` if your crate does not depend on `.npz` file support
* `["npz"]` if your crate depends on `.npz` file support but not compression
* `["compressed_npz"]` if your crate depends on `.npz` file support with compression

Ideally, do not include a *required* dependency on the `default` feature set or
the `compressed_npz_default` feature, so that the user can select their desired
`flate2` backend.

If your crate depends on the `compressed_npz` feature, it may be a good idea to
simplify use with the following:

```toml
[features]
default = ["ndarray-npy/compressed_npz_default"]
```

so that the user does not have to manually select a `flate2` backend if they
use your crate's default feature set. This still enables the user to select a
backend if they use `default-features = false` with your crate.

## Releases

* **0.2.0** (not yet released)

  * Updated to `zip` 0.4.
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

Copyright 2018 Jim Turner

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE), or the [MIT
license](LICENSE-MIT), at your option. You may not use this project except in
compliance with those terms.
