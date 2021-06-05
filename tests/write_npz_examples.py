import os
import numpy as np

b8 = np.array([True, False], dtype='bool')
i8 = np.arange(10, dtype='int8')
u8 = np.arange(10, dtype='uint8')

np.savez("examples.npz",
    b8=b8,
    i8=i8,
    u8=u8,
)

le_i16 = np.arange(10, dtype='int16')
le_u16 = np.arange(10, dtype='uint16')
le_i32 = np.arange(10, dtype='int32')
le_u32 = np.arange(10, dtype='uint32')
le_i64 = np.arange(10, dtype='int64')
le_u64 = np.arange(10, dtype='uint64')
le_f32 = np.arange(10, dtype='float32')
le_f64 = np.arange(10, dtype='float64')

np.savez("examples_little_endian.npz",
    i16=le_i16,
    u16=le_u16,
    i32=le_i32,
    u32=le_u32,
    i64=le_i64,
    u64=le_u64,
    f32=le_f32,
    f64=le_f64,
)

be_i16 = le_i16.byteswap()
be_u16 = le_u16.byteswap()
be_i32 = le_i32.byteswap()
be_u32 = le_u32.byteswap()
be_i64 = le_i64.byteswap()
be_u64 = le_u64.byteswap()
be_f32 = le_f32.byteswap()
be_f64 = le_f64.byteswap()

np.savez("examples_big_endian.npz",
    i16=be_i16,
    u16=be_u16,
    i32=be_i32,
    u32=be_u32,
    i64=be_i64,
    u64=be_u64,
    f32=be_f32,
    f64=be_f64,
)

os.system('rezip -f examples.npz -o examples_64_byte_aligned.npz')
os.system('rezip -f examples_little_endian.npz -o examples_little_endian_64_byte_aligned.npz')
os.system('rezip -f examples_big_endian.npz -o examples_big_endian_64_byte_aligned.npz')

os.system('rm examples.npz')
os.system('rm examples_little_endian.npz')
os.system('rm examples_big_endian.npz')
