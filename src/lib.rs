#![doc = include_str!("../README.md")]
#![deny(warnings, missing_docs)]

/// An array layout allow N dimensions inlined.
pub struct ArrayLayout<const N: usize = 2> {
    ndim: usize,
    content: Union<N>,
}

union Union<const N: usize> {
    ptr: NonNull<usize>,
    _inlined: (isize, [usize; N], [isize; N]),
}

impl<const N: usize> Clone for ArrayLayout<N> {
    #[inline]
    fn clone(&self) -> Self {
        Self::new(self.shape(), self.strides(), self.offset())
    }
}

impl<const N: usize> PartialEq for ArrayLayout<N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ndim == other.ndim && self.content().as_slice() == other.content().as_slice()
    }
}

impl<const N: usize> Eq for ArrayLayout<N> {}

impl<const N: usize> Drop for ArrayLayout<N> {
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr_allocated() {
            unsafe { dealloc(ptr.cast().as_ptr(), layout(self.ndim)) }
        }
    }
}

/// 元信息存储顺序。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Endian {
    /// 大端序，范围更大的维度在元信息中更靠前的位置。
    BigEndian,
    /// 小端序，范围更小的维度在元信息中更靠前的位置。
    LittleEndian,
}

impl<const N: usize> ArrayLayout<N> {
    /// Creates a new Layout with the given shape, strides, and offset.
    ///
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<4>::new(&[2, 3, 4], &[12, -4, 1], 20);
    /// assert_eq!(layout.offset(), 20);
    /// assert_eq!(layout.shape(), &[2, 3, 4]);
    /// assert_eq!(layout.strides(), &[12, -4, 1]);
    /// ```
    pub fn new(shape: &[usize], strides: &[isize], offset: isize) -> Self {
        // check
        assert_eq!(
            shape.len(),
            strides.len(),
            "shape and strides must have the same length"
        );

        let mut ans = Self::with_ndim(shape.len());
        let mut content = ans.content_mut();
        content.set_offset(offset);
        content.copy_shape(shape);
        content.copy_strides(strides);
        ans
    }

    /// Creates a new contiguous Layout with the given shape.
    ///
    /// ```rust
    /// # use ndarray_layout::{Endian, ArrayLayout};
    /// let layout = ArrayLayout::<4>::new_contiguous(&[2, 3, 4], Endian::LittleEndian, 4);
    /// assert_eq!(layout.offset(), 0);
    /// assert_eq!(layout.shape(), &[2, 3, 4]);
    /// assert_eq!(layout.strides(), &[4, 8, 24]);
    /// ```
    pub fn new_contiguous(shape: &[usize], endian: Endian, element_size: usize) -> Self {
        let mut ans = Self::with_ndim(shape.len());
        let mut content = ans.content_mut();
        content.set_offset(0);
        content.copy_shape(shape);
        let mut mul = element_size as isize;
        let push = |i| {
            content.set_stride(i, mul);
            mul *= shape[i] as isize;
        };
        match endian {
            Endian::BigEndian => (0..shape.len()).rev().for_each(push),
            Endian::LittleEndian => (0..shape.len()).for_each(push),
        }
        ans
    }

    /// Gets offset.
    #[inline]
    pub const fn ndim(&self) -> usize {
        self.ndim
    }

    /// Gets offset.
    #[inline]
    pub fn offset(&self) -> isize {
        self.content().offset()
    }

    /// Gets shape.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.content().shape()
    }

    /// Gets strides.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.content().strides()
    }

    /// Calculate the range of data in bytes to determine the location of the memory area that the tensor needs to access.
    pub fn data_range(&self) -> RangeInclusive<isize> {
        let content = self.content();
        let mut start = content.offset();
        let mut end = content.offset();
        for (&d, s) in zip(content.shape(), content.strides()) {
            use std::cmp::Ordering::{Equal, Greater, Less};
            let i = d as isize - 1;
            match s.cmp(&0) {
                Equal => {}
                Less => start += s * i,
                Greater => end += s * i,
            }
        }
        start..=end
    }
}

mod transform;
pub use transform::{IndexArg, SliceArg, Split, TileArg};

use std::{
    alloc::{alloc, dealloc, Layout},
    isize,
    iter::zip,
    ops::RangeInclusive,
    ptr::{copy_nonoverlapping, NonNull},
    slice::from_raw_parts,
};

impl<const N: usize> ArrayLayout<N> {
    #[inline]
    fn ptr_allocated(&self) -> Option<NonNull<usize>> {
        const { assert!(N > 0) }
        if self.ndim > N {
            Some(unsafe { self.content.ptr })
        } else {
            None
        }
    }

    #[inline]
    fn content(&self) -> Content<false> {
        Content {
            ptr: self
                .ptr_allocated()
                .unwrap_or(unsafe { NonNull::new_unchecked(&self.content as *const _ as _) }),
            ndim: self.ndim,
        }
    }

    #[inline]
    fn content_mut(&mut self) -> Content<true> {
        Content {
            ptr: self
                .ptr_allocated()
                .unwrap_or(unsafe { NonNull::new_unchecked(&self.content as *const _ as _) }),
            ndim: self.ndim,
        }
    }

    /// Create a new ArrayLayout with the given dimensions.
    #[inline]
    fn with_ndim(ndim: usize) -> Self {
        Self {
            ndim,
            content: if ndim <= N {
                Union {
                    _inlined: (0, [0; N], [0; N]),
                }
            } else {
                Union {
                    ptr: unsafe { NonNull::new_unchecked(alloc(layout(ndim)).cast()) },
                }
            },
        }
    }
}

struct Content<const MUT: bool> {
    ptr: NonNull<usize>,
    ndim: usize,
}

impl Content<false> {
    #[inline]
    fn as_slice(&self) -> &[usize] {
        unsafe { from_raw_parts(self.ptr.as_ptr(), 1 + self.ndim * 2) }
    }

    #[inline]
    fn offset(&self) -> isize {
        unsafe { self.ptr.cast().read() }
    }

    #[inline]
    fn shape<'a>(&self) -> &'a [usize] {
        unsafe { from_raw_parts(self.ptr.add(1).as_ptr(), self.ndim) }
    }

    #[inline]
    fn strides<'a>(&self) -> &'a [isize] {
        unsafe { from_raw_parts(self.ptr.add(1 + self.ndim).cast().as_ptr(), self.ndim) }
    }
}

impl Content<true> {
    #[inline]
    fn set_offset(&mut self, val: isize) {
        unsafe { self.ptr.cast().write(val) }
    }

    #[inline]
    fn set_shape(&mut self, idx: usize, val: usize) {
        assert!(idx < self.ndim);
        unsafe { self.ptr.add(1 + idx).write(val) }
    }

    #[inline]
    fn set_stride(&mut self, idx: usize, val: isize) {
        assert!(idx < self.ndim);
        unsafe { self.ptr.add(1 + idx + self.ndim).cast().write(val) }
    }

    #[inline]
    fn copy_shape(&mut self, val: &[usize]) {
        assert!(val.len() == self.ndim);
        unsafe { copy_nonoverlapping(val.as_ptr(), self.ptr.add(1).as_ptr(), self.ndim) }
    }

    #[inline]
    fn copy_strides(&mut self, val: &[isize]) {
        assert!(val.len() == self.ndim);
        unsafe {
            copy_nonoverlapping(
                val.as_ptr(),
                self.ptr.add(1 + self.ndim).cast().as_ptr(),
                self.ndim,
            )
        }
    }
}

#[inline]
fn layout(ndim: usize) -> Layout {
    Layout::array::<usize>(1 + ndim * 2).unwrap()
}
