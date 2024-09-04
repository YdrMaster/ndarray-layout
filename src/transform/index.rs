use crate::TensorLayout;
use std::iter::zip;

impl<const N: usize> TensorLayout<N> {
    pub fn index(&self, axis: usize, index: usize) -> Self {
        assert!(axis < self.order);
        assert!(index < self.shape()[axis]);

        let mut ans = Self::with_order(self.order - 1);
        let mut_ = ans.as_mut();

        let mut j = 0;
        for (i, (&d, &s)) in zip(self.shape(), self.strides()).enumerate() {
            if i != axis {
                mut_.shape[j] = d;
                mut_.strides[j] = s;
                j += 1;
            } else {
                let offset = self.offset as isize + index as isize * s;
                assert!(offset >= 0);
                *mut_.offset = offset as _;
            }
        }

        ans
    }
}

#[test]
fn test() {
    let layout = TensorLayout::<3>::new(&[2, 3, 4], &[12, 4, 1], 0);
    let layout = layout.index(1, 2);
    assert_eq!(layout.shape(), &[2, 4]);
    assert_eq!(layout.strides(), &[12, 1]);
    assert_eq!(layout.offset(), 8);

    let layout = TensorLayout::<1>::new(&[2, 3, 4], &[12, 4, 1], 0);
    let layout = layout.index(1, 2);
    assert_eq!(layout.shape(), &[2, 4]);
    assert_eq!(layout.strides(), &[12, 1]);
    assert_eq!(layout.offset(), 8);

    let layout = TensorLayout::<4>::new(&[2, 3, 4], &[12, -4, 1], 20);
    let layout = layout.index(1, 2);
    assert_eq!(layout.shape(), &[2, 4]);
    assert_eq!(layout.strides(), &[12, 1]);
    assert_eq!(layout.offset(), 12);
}
