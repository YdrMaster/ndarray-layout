use crate::TensorLayout;
use std::iter::zip;

/// 分块变换参数。
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TileArg<'a> {
    /// 分块的轴。
    pub axis: usize,
    /// 分块的顺序。
    pub order: TileOrder,
    /// 分块的大小。
    pub tiles: &'a [usize],
}

/// 分块顺序。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TileOrder {
    /// 大端分块，分块后范围更大的维度在形状中更靠前的位置。
    BigEndian,
    /// 小端分块，分块后范围更小的维度在形状中更靠前的位置。
    LittleEndian,
}

impl<const N: usize> TensorLayout<N> {
    /// 分块变换是将单个维度划分为多个分块的变换。
    /// 大端分块使得分块后范围更大的维度在形状中更靠前的位置。
    ///
    /// ```rust
    /// # use tensor::TensorLayout;
    /// let layout = TensorLayout::<3>::new(&[2, 3, 6], &[18, 6, 1], 0).tile_be(2, &[2, 3]);
    /// assert_eq!(layout.shape(), &[2, 3, 2, 3]);
    /// assert_eq!(layout.strides(), &[18, 6, 3, 1]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    #[inline]
    pub fn tile_be(&self, axis: usize, tiles: &[usize]) -> Self {
        self.tile_many(&[TileArg {
            axis,
            order: TileOrder::BigEndian,
            tiles,
        }])
    }

    /// 分块变换是将单个维度划分为多个分块的变换。
    /// 小端分块使得分块后范围更小的维度在形状中更靠前的位置。
    ///
    /// ```rust
    /// # use tensor::TensorLayout;
    /// let layout = TensorLayout::<3>::new(&[2, 3, 6], &[18, 6, 1], 0).tile_le(2, &[2, 3]);
    /// assert_eq!(layout.shape(), &[2, 3, 2, 3]);
    /// assert_eq!(layout.strides(), &[18, 6, 1, 2]);
    /// assert_eq!(layout.offset(), 0);
    /// ```
    #[inline]
    pub fn tile_le(&self, axis: usize, tiles: &[usize]) -> Self {
        self.tile_many(&[TileArg {
            axis,
            order: TileOrder::LittleEndian,
            tiles,
        }])
    }

    /// 一次对多个阶进行分块变换。
    pub fn tile_many(&self, mut args: &[TileArg]) -> Self {
        let content = self.content();
        let shape = content.shape();
        let iter = zip(shape, content.strides()).enumerate();

        let check = |&TileArg { axis, tiles, .. }| {
            shape
                .get(axis)
                .filter(|&&d| d == tiles.iter().product())
                .is_some()
        };

        let (mut new_orders, mut last_axis) = match args {
            [first, ..] => {
                assert!(check(first));
                (first.tiles.len(), first.axis)
            }
            [..] => return self.clone(),
        };
        for arg in &args[1..] {
            assert!(check(arg));
            assert!(arg.axis > last_axis);
            new_orders += arg.tiles.len();
            last_axis = arg.axis;
        }

        let mut ans = Self::with_order(self.order + new_orders - args.len());

        let mut content = ans.content_mut();
        content.set_offset(self.offset());
        let mut j = 0;
        let mut push = |t, s| {
            content.set_shape(j, t);
            content.set_stride(j, s);
            j += 1;
        };

        for (i, (&d, &s)) in iter {
            match *args {
                [TileArg { axis, order, tiles }, ref tail @ ..] if axis == i => {
                    match order {
                        TileOrder::BigEndian => {
                            // tile   : [a,         b    , c]
                            // strides: [s * c * b, s * c, s]
                            let mut s = s * d as isize;
                            for &t in tiles {
                                s /= t as isize;
                                push(t, s);
                            }
                        }
                        TileOrder::LittleEndian => {
                            // tile   : [a, b    , c        ]
                            // strides: [s, s * a, s * a * b]
                            let mut s = s;
                            for &t in tiles {
                                push(t, s);
                                s *= t as isize;
                            }
                        }
                    }
                    args = tail;
                }
                [..] => push(d, s),
            }
        }
        ans
    }
}
