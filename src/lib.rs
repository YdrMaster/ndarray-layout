mod transform;

use std::{
    alloc::{alloc_zeroed, dealloc, Layout},
    ptr::NonNull,
    slice::{from_raw_parts, from_raw_parts_mut},
};

pub struct TensorLayout<const N: usize = 2> {
    order: usize,
    offset: usize,
    shape_inline: [usize; N],
    strides_inline: [isize; N],
}

impl<const N: usize> Clone for TensorLayout<N> {
    #[inline]
    fn clone(&self) -> Self {
        Self::new(self.shape(), self.strides(), self.offset)
    }
}

impl<const N: usize> PartialEq for TensorLayout<N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
            && self.offset == other.offset
            && self.shape() == other.shape()
            && self.strides() == other.strides()
    }
}

impl<const N: usize> Eq for TensorLayout<N> {}

impl<const N: usize> Drop for TensorLayout<N> {
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr() {
            unsafe { dealloc(ptr.as_ptr(), layout(self.capacity())) }
        }
    }
}

impl<const N: usize> TensorLayout<N> {
    pub fn new(shape: &[usize], strides: &[isize], offset: usize) -> Self {
        assert_eq!(
            shape.len(),
            strides.len(),
            "shape and strides must have the same length"
        );
        let mut ans = Self::with_order(shape.len());
        let mut_ = ans.as_mut();
        *mut_.offset = offset;
        mut_.shape.copy_from_slice(shape);
        mut_.strides.copy_from_slice(strides);
        ans
    }

    #[inline]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        if let Some(ptr) = self.ptr() {
            let ptr = ptr.cast().as_ptr();
            unsafe { from_raw_parts(ptr, self.order) }
        } else {
            &self.shape_inline[..self.order]
        }
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        if let Some(ptr) = self.ptr() {
            let ptr = ptr.cast::<isize>().as_ptr();
            unsafe { from_raw_parts(ptr.add(self.capacity()), self.order) }
        } else {
            &self.strides_inline[..self.order]
        }
    }

    #[inline]
    fn ptr(&self) -> Option<NonNull<u8>> {
        const { assert!(N > 0) }
        if self.order > N {
            Some(NonNull::new(self.shape_inline[0] as _).unwrap())
        } else {
            None
        }
    }

    #[inline]
    const fn capacity(&self) -> usize {
        self.strides_inline[0] as _
    }

    #[inline]
    fn with_order(order: usize) -> Self {
        let mut ans = Self {
            order,
            offset: 0,
            shape_inline: [0; N],
            strides_inline: [0; N],
        };
        if order > N {
            let ptr = unsafe { alloc_zeroed(layout(order)) }.cast::<usize>();
            ans.shape_inline[0] = ptr as _;
            ans.strides_inline[0] = order as _;
        }
        ans
    }

    #[inline]
    fn as_mut(&mut self) -> Mut {
        if let Some(ptr) = self.ptr() {
            let ptr = ptr.cast().as_ptr();
            let capacity = self.capacity();
            Mut {
                offset: &mut self.offset,
                shape: unsafe { from_raw_parts_mut(ptr, self.order) },
                strides: unsafe { from_raw_parts_mut(ptr.add(capacity).cast(), self.order) },
            }
        } else {
            Mut {
                offset: &mut self.offset,
                shape: &mut self.shape_inline[..self.order],
                strides: &mut self.strides_inline[..self.order],
            }
        }
    }
}

#[inline]
fn layout(capacity: usize) -> Layout {
    Layout::array::<usize>(capacity * 2).unwrap()
}

struct Mut<'a> {
    offset: &'a mut usize,
    shape: &'a mut [usize],
    strides: &'a mut [isize],
}

#[test]
fn test() {
    let layout = TensorLayout::<4>::with_order(5);
    assert_eq!(layout.offset(), 0);
    assert_eq!(layout.shape(), &[0, 0, 0, 0, 0]);
    assert_eq!(layout.strides(), &[0, 0, 0, 0, 0]);
    assert!(layout.ptr().is_some());
    assert_eq!(layout.capacity(), 5);

    let layout = TensorLayout::<4>::new(&[2, 3, 4], &[12, -4, 1], 20);
    assert_eq!(layout.offset(), 20);
    assert_eq!(layout.shape(), &[2, 3, 4]);
    assert_eq!(layout.strides(), &[12, -4, 1]);
    assert!(layout.ptr().is_none());
}
