# ndarray-npy

[![Continuous integration](https://github.com/jturner314/ndarray-npy/actions/workflows/ci.yml/badge.svg)](https://github.com/jturner314/ndarray-npy/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/jturner314/ndarray-npy/branch/master/graph/badge.svg)](https://codecov.io/gh/jturner314/ndarray-npy)
[![Dependencies status](https://deps.rs/repo/github/jturner314/ndarray-npy/status.svg)](https://deps.rs/repo/github/jturner314/ndarray-npy)
[![ndarray-npy at crates.io](https://img.shields.io/crates/v/ndarray-npy.svg)](https://crates.io/crates/ndarray-npy)
[![Documentation](https://docs.rs/ndarray-npy/badge.svg)](https://docs.rs/ndarray-npy)
[![Matrix chat at #rust-sci:matrix.org](https://img.shields.io/badge/Matrix-%23rust--sci%3Amatrix.org-lightgrey)](https://matrix.to/#/#rust-sci:matrix.org)
[![IRC at #rust-sci on OFTC](https://img.shields.io/badge/IRC-%23rust--sci%20on%20OFTC-lightgrey)](https://webchat.oftc.net/?channels=rust-sci)

This crate provides support for reading/writing [`ndarray`]'s `ArrayBase` type
from/to [`.npy`] and [`.npz`] files. See the
[documentation](https://docs.rs/ndarray-npy) for more information.

[`ndarray`]: https://github.com/bluss/ndarray
[`.npy`]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html
[`.npz`]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html

**This crate is a work-in-progress.** It currently supports only a subset of
`.npy` header descriptors and supports only primitive fixed-size integer,
floating point, and `bool` types as the array element type. You can implement
the `*Element` traits for your own types, but the next breaking release of this
library will probably change those traits.

Future plans include support for:

* Memory-mapped `.npz` files. (Memory-mapped `.npy` files are already
  supported.)
* More element types (e.g. structs). If you need support for structs before
  this is implemented in `ndarray-npy`, check out the [`npy` crate].
* Possibly merging this with the [`npy` crate].

[`npy` crate]: https://crates.io/crates/npy

## Using with Cargo

To use with the default features:

```toml
[dependencies]
ndarray-npy = "0.8"
```

The `default` feature set includes the `compressed_npz` feature, which enables
support for uncompresssed and compressed `.npz` files. This requires a
dependency on the [`zip` crate] and a compression backend crate.

To use without the default features:

```toml
[dependencies]
ndarray-npy = { version = "0.8", default-features = false }
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
version = "0.8"
default-features = false
features = ["npz"]
```

[`zip` crate]: https://crates.io/crates/zip

### Library authors

Library authors should specify their dependency on `ndarray-npy` like this:

```toml
[dependencies.ndarray-npy]
version = "0.8"
default-features = false
features = [FEATURES_LIST_HERE]
```

where the `features` list is one of the following:

* `[]` if your crate does not depend on `.npz` file support
* `["npz"]` if your crate depends on `.npz` file support but not compression
* `["compressed_npz"]` if your crate depends on `.npz` file support with compression

## Releases

See [RELEASES.md](RELEASES.md).

## Contributing

Please feel free to create issues and submit PRs. PRs adding more tests would
be especially appreciated.

## License

Copyright 2018–2021 Jim Turner and `ndarray-npy` developers

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE), or the [MIT
license](LICENSE-MIT), at your option. You may not use this project except in
compliance with those terms.
