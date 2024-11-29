use crate::ArrayLayout;

/// 索引变换参数。
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BroadcastArg {
    /// 广播的轴。
    pub axis: usize,
    /// 广播次数。
    pub times: usize,
}

impl<const N: usize> ArrayLayout<N> {
    /// 广播变换将指定的长度为 1 的阶扩增指定的倍数，并将其步长固定为 0。
    ///
    /// ```rust
    /// # use ndarray_layout::ArrayLayout;
    /// let layout = ArrayLayout::<3>::new(&[1, 5, 2], &[10, 2, 1], 0).broadcast(0, 10);
    /// assert_eq!(layout.shape(), &[10, 5, 2]);
    /// assert_eq!(layout.strides(), &[0, 2, 1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    pub fn broadcast(&self, axis: usize, times: usize) -> Self {
        self.broadcast_many(&[BroadcastArg { axis, times }])
    }

    /// 一次对多个阶进行广播变换。
    pub fn broadcast_many(&self, args: &[BroadcastArg]) -> Self {
        let mut ans = self.clone();
        let mut content = ans.content_mut();
        for &BroadcastArg { axis, times } in args {
            assert!(content.shape()[axis] == 1 || content.strides()[axis] == 0);
            content.set_shape(axis, times);
            content.set_stride(axis, 0);
        }
        ans
    }
}
