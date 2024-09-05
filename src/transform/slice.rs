﻿use crate::TensorLayout;
use std::iter::zip;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SliceArg {
    pub axis: usize,
    pub start: usize,
    pub step: isize,
    pub len: usize,
}

impl<const N: usize> TensorLayout<N> {
    /// 切片变换是裁剪张量指定阶上一组连续数据的变换。
    ///
    /// ```rust
    /// # use tensor::TensorLayout;
    /// // axis = 1, start = 1, step = -1, len = 2
    /// let layout = TensorLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).slice(1, 2, -1, 2);
    /// assert_eq!(layout.shape(), &[2, 2, 4]);
    /// assert_eq!(layout.strides(), &[12, -4, 1]);
    /// assert_eq!(layout.offset(), 8);
    /// ```
    pub fn slice(&self, axis: usize, start: usize, step: isize, len: usize) -> Self {
        self.slice_many(&[SliceArg {
            axis,
            start,
            step,
            len,
        }])
    }

    /// 一次对多个阶进行切片变换。
    pub fn slice_many(&self, mut args: &[SliceArg]) -> Self {
        let content = self.content();
        let mut offset = content.offset() as isize;
        let shape = content.shape();
        let strides = content.strides();

        let ans = Self::with_order(self.order);
        let content = ans.content();
        for (i, (&d, &s)) in zip(shape, strides).enumerate() {
            match args {
                [arg, tail @ ..] if arg.axis == i => {
                    let &SliceArg {
                        axis,
                        start,
                        step,
                        len,
                    } = arg;
                    use std::cmp::Ordering::*;
                    let len = match step.cmp(&0) {
                        Greater => {
                            assert!(start < d);
                            offset += start as isize * s;
                            (d - start).div_ceil(step as _).min(len)
                        }
                        Equal => {
                            assert!(start < d);
                            offset += start as isize * s;
                            len
                        }
                        Less => {
                            let start = start.min(d - 1);
                            offset += start as isize * s;
                            (start + 1).div_ceil((-step) as _).min(len)
                        }
                    };
                    content.set_shape(i, len);
                    content.set_stride(i, s * step);

                    if let [next, ..] = tail {
                        assert!(
                            axis < next.axis && next.axis < self.order,
                            "next.axis = {} ~in ({}, {})",
                            next.axis,
                            axis,
                            self.order,
                        );
                    }
                    args = tail;
                }
                [..] => {
                    content.set_shape(i, d);
                    content.set_stride(i, s);
                }
            }
        }
        content.set_offset(offset as _);
        ans
    }
}