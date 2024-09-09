#![doc = include_str!("../README.md")]
#![deny(warnings, missing_docs)]

/// A tensor layout allow N dimensions inlined.
pub struct TensorLayout<const N: usize = 2> {
    order: usize,
    content: Union<N>,
}

union Union<const N: usize> {
    ptr: NonNull<usize>,
    _inlined: (usize, [usize; N], [isize; N]),
}

impl<const N: usize> Clone for TensorLayout<N> {
    #[inline]
    fn clone(&self) -> Self {
        Self::new(self.shape(), self.strides(), self.offset())
    }
}

impl<const N: usize> PartialEq for TensorLayout<N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order && self.content().as_slice() == other.content().as_slice()
    }
}

impl<const N: usize> Eq for TensorLayout<N> {}

impl<const N: usize> Drop for TensorLayout<N> {
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr_allocated() {
            unsafe { dealloc(ptr.cast().as_ptr(), layout(self.order)) }
        }
    }
}

impl<const N: usize> TensorLayout<N> {
    /// Create a new TensorLayout with the given shape, strides, and offset.
    ///
    /// ```rust
    /// # use tensor::TensorLayout;
    /// let layout = TensorLayout::<4>::new(&[2, 3, 4], &[12, -4, 1], 20);
    /// assert_eq!(layout.offset(), 20);
    /// assert_eq!(layout.shape(), &[2, 3, 4]);
    /// assert_eq!(layout.strides(), &[12, -4, 1]);
    /// ```
    pub fn new(shape: &[usize], strides: &[isize], offset: usize) -> Self {
        // check
        assert_eq!(
            shape.len(),
            strides.len(),
            "shape and strides must have the same length"
        );
        assert!(zip(shape, strides)
            .scan(offset as isize, |offset, (&d, &s)| {
                if s < 0 {
                    *offset += (d - 1) as isize * s;
                }
                Some(*offset)
            })
            .all(|off| off >= 0));

        let mut ans = Self::with_order(shape.len());
        let mut content = ans.content_mut();
        content.set_offset(offset);
        content.copy_shape(shape);
        content.copy_strides(strides);
        ans
    }

    /// Get offset of tensor.
    #[inline]
    pub fn offset(&self) -> usize {
        self.content().offset()
    }

    /// Get shape of tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.content().shape()
    }

    /// Get strides of tensor.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.content().strides()
    }
}

mod transform;
pub use transform::{IndexArg, SliceArg, Split, TileArg, TileOrder};

use std::{
    alloc::{alloc, dealloc, Layout},
    iter::zip,
    ptr::{copy_nonoverlapping, NonNull},
    slice::from_raw_parts,
};

impl<const N: usize> TensorLayout<N> {
    #[inline]
    fn ptr_allocated(&self) -> Option<NonNull<usize>> {
        const { assert!(N > 0) }
        if self.order > N {
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
            ord: self.order,
        }
    }

    #[inline]
    fn content_mut(&mut self) -> Content<true> {
        Content {
            ptr: self
                .ptr_allocated()
                .unwrap_or(unsafe { NonNull::new_unchecked(&self.content as *const _ as _) }),
            ord: self.order,
        }
    }

    /// Create a new TensorLayout with the given order.
    #[inline]
    fn with_order(order: usize) -> Self {
        Self {
            order,
            content: if order <= N {
                Union {
                    _inlined: (0, [0; N], [0; N]),
                }
            } else {
                Union {
                    ptr: unsafe { NonNull::new_unchecked(alloc(layout(order)).cast()) },
                }
            },
        }
    }
}

struct Content<const MUT: bool> {
    ptr: NonNull<usize>,
    ord: usize,
}

impl Content<false> {
    #[inline]
    fn as_slice(&self) -> &[usize] {
        unsafe { from_raw_parts(self.ptr.as_ptr(), 1 + self.ord * 2) }
    }

    #[inline]
    fn offset(&self) -> usize {
        unsafe { self.ptr.read() }
    }

    #[inline]
    fn shape<'a>(&self) -> &'a [usize] {
        unsafe { from_raw_parts(self.ptr.add(1).as_ptr(), self.ord) }
    }

    #[inline]
    fn strides<'a>(&self) -> &'a [isize] {
        unsafe { from_raw_parts(self.ptr.add(1 + self.ord).cast().as_ptr(), self.ord) }
    }
}

impl Content<true> {
    #[inline]
    fn set_offset(&mut self, val: usize) {
        unsafe { self.ptr.write(val) }
    }

    #[inline]
    fn set_shape(&mut self, idx: usize, val: usize) {
        assert!(idx < self.ord);
        unsafe { self.ptr.add(1 + idx).write(val) }
    }

    #[inline]
    fn set_stride(&mut self, idx: usize, val: isize) {
        assert!(idx < self.ord);
        unsafe { self.ptr.add(1 + idx + self.ord).cast().write(val) }
    }

    #[inline]
    fn copy_shape(&mut self, val: &[usize]) {
        assert!(val.len() == self.ord);
        unsafe { copy_nonoverlapping(val.as_ptr(), self.ptr.add(1).as_ptr(), self.ord) }
    }

    #[inline]
    fn copy_strides(&mut self, val: &[isize]) {
        assert!(val.len() == self.ord);
        unsafe {
            copy_nonoverlapping(
                val.as_ptr(),
                self.ptr.add(1 + self.ord).cast().as_ptr(),
                self.ord,
            )
        }
    }
}

#[inline]
fn layout(order: usize) -> Layout {
    Layout::array::<usize>(1 + order * 2).unwrap()
}
