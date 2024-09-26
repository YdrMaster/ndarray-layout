use crate::ArrayLayout;
use std::{iter::zip, ops::Range};

impl<const N: usize> ArrayLayout<N> {
    /// 合并变换是将多个连续维度划分合并的变换。
    ///
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0).merge(0..3).unwrap();
    /// assert_eq!(layout.shape(), &[24]);
    /// assert_eq!(layout.strides(), &[1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    #[inline]
    pub fn merge(&self, range: Range<usize>) -> Option<Self> {
        self.merge_many(&[range])
    }

    /// 一次对多个阶进行合并变换。
    pub fn merge_many(&self, args: &[Range<usize>]) -> Option<Self> {
        let content = self.content();
        let shape = content.shape();
        let strides = content.strides();

        let merged = args.iter().map(|range| range.len()).sum::<usize>();
        let mut ans = Self::with_ndim(self.ndim + args.len() - merged);

        let mut content = ans.content_mut();
        content.set_offset(self.offset());
        let mut i = 0;
        let mut push = |d, s| {
            content.set_shape(i, d);
            content.set_stride(i, s);
            i += 1;
        };

        let mut last_end = 0;
        for range in args {
            if range.is_empty() {
                continue;
            }

            assert!(range.start >= last_end);
            for j in last_end..range.start {
                push(shape[j], strides[j]);
            }

            let mut pairs = zip(&shape[range.clone()], &strides[range.clone()]).collect::<Vec<_>>();
            pairs.sort_unstable_by_key(|(_, &s)| s.unsigned_abs());

            let (&d, &s) = pairs[0];
            let mut d = d;

            for i in 1..pairs.len() {
                let (&l, &ls) = pairs[i - 1];
                let (&r, &rs) = pairs[i];
                if l == 1 || s == 1 || ls == rs * r as isize || rs == ls * l as isize {
                    d *= r;
                } else {
                    return None;
                }
            }

            push(d, s);
            last_end = range.end;
        }
        for j in last_end..shape.len() {
            push(shape[j], strides[j]);
        }

        Some(ans)
    }
}
