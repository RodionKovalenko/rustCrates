#[cfg(test)]
mod tests {
    use num::Complex;
    use crate::wavelet_transform::cwt::{cwt_1d, cwt_2d};
    use crate::wavelet_transform::cwt_types::ContinuousWaletetType;
    use crate::wavelet_transform::fft::{fft_real1_d, fft_real2_d};

    #[test]
    fn test_cwt_1d() {
        let scales: Vec<f64> = (1..6).map(|x| x as f64).collect();
        let n: Vec<f64> = vec![1.0, 2.0, 3.0];

        let (transform_cwt, frequencies) = cwt_1d(&n, &scales, &ContinuousWaletetType::MEXH, &1.0);

        assert_eq!(5, transform_cwt.len());
        assert_eq!(transform_cwt[0], vec![-0.6741036718169315, 0.7022068718530412, 2.338346780786158]);
        assert_eq!(transform_cwt[1], vec![0.5041676154169978, 2.23192367774792, 2.9105329108437434]);
        assert_eq!(transform_cwt[2], vec![1.4601796060827077, 2.417532862790752, 2.6758970235153003]);
        assert_eq!(transform_cwt[3], vec![1.7607301542542242, 2.297982153373181, 2.4488656061501315]);
        assert_eq!(transform_cwt[4], vec![1.7858211385362848, 2.1782640485201874, 2.271167123147155]);

        assert_eq!(5, frequencies.len());
        assert_eq!(frequencies[0], 0.25);
        assert_eq!(frequencies[1], 0.125);
        assert_eq!(frequencies[2], 0.08333333333333333);
        assert_eq!(frequencies[3], 0.0625);
        assert_eq!(frequencies[4], 0.05);
    }

    #[test]
    fn test_cwt_2d() {
        let scales: Vec<f64> = (1..6).map(|x| x as f64).collect();
        let n: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let (transform_cwt, frequencies) = cwt_2d(&n, &scales, &ContinuousWaletetType::MEXH, &1.0);

        assert_eq!(5, transform_cwt.len());
        assert_eq!(transform_cwt[0], vec![[-0.67410367181693154, 0.7022068718530412, 2.338346780786158], [-0.5880689304740532, 2.9826221113324345, 4.618762020265538]]);
        assert_eq!(transform_cwt[1], vec![[0.5041676154169978, 2.23192367774792, 2.9105329108437434], [2.2939493779819182, 6.088766119191671, 6.767375352287462]]);
        assert_eq!(transform_cwt[2], vec![[1.4601796060827077, 2.417532862790752, 2.6758970235153003], [4.193716683741932, 6.237605277520296, 6.4959694382448205]]);
        assert_eq!(transform_cwt[3], vec![[1.7607301542542242, 2.297982153373181, 2.4488656061501315], [4.708172248389279, 5.858117973015669, 6.009001425792602]]);
        assert_eq!(transform_cwt[4], vec![[1.7858211385362848, 2.1782640485201874, 2.271167123147155], [4.684000069989407, 5.515337427270696, 5.60824050189765]]);

        assert_eq!(5, frequencies.len());
        assert_eq!(frequencies[0], 0.25);
        assert_eq!(frequencies[1], 0.125);
        assert_eq!(frequencies[2], 0.08333333333333333);
        assert_eq!(frequencies[3], 0.0625);
        assert_eq!(frequencies[4], 0.05);

        let scales: Vec<f64> = (2..5).map(|x| x as f64).collect();
        let n: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let (transform_cwt, frequencies) = cwt_2d(&n, &scales, &ContinuousWaletetType::MEXH, &1.0);

        assert_eq!(3, transform_cwt.len());
        assert_eq!(transform_cwt[0], vec![[-0.5581140923146637, 1.6424011241705805, 3.7198727962465195, 4.048766419597308], [0.7659798833735681, 6.195335159184911, 9.671669270240919, 8.601700454611564]]);
        assert_eq!(transform_cwt[1], vec![[1.1168209497544161, 2.838871815948713, 4.029221437984578, 4.054589233381288], [4.41817839697176, 8.353640655412732, 10.47597573875988, 9.569358072845233]]);
        assert_eq!(transform_cwt[2], vec![[2.039363009077822, 3.1639287410556776, 3.829970255621962, 3.8520641929515707], [6.247918656081495, 8.776723088261488, 9.957922664617087, 9.464858540157326]]);

        assert_eq!(3, frequencies.len());
        assert_eq!(frequencies[0], 0.125);
        assert_eq!(frequencies[1], 0.08333333333333333);
        assert_eq!(frequencies[2], 0.0625);
    }

    #[test]
    fn test_ff_1d_real() {
        let n: Vec<f64> = vec![1.0, 2.0, 3.0];
        let fft = fft_real1_d(&n);

        assert_eq!(3, fft.len());
        assert_eq!(fft[0], Complex::new(6.0, 0.0));
        assert_eq!(fft[1], Complex::new(-1.5000000000000009, 0.8660254037844382));
        assert_eq!(fft[2], Complex::new(-1.4999999999999987, -0.8660254037844404));
    }

    #[test]
    fn test_ff_2d_real() {
        let n: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let fft = fft_real2_d(&n);

        assert_eq!(2, fft.len());
        assert_eq!(fft[0], [Complex::new(21.0, 0.0), Complex::new(-3.0000000000000027, 1.7320508075688756), Complex::new(-2.9999999999999964, -1.7320508075688823)]);
        assert_eq!(fft[1], [Complex::new(-9.0, -1.83697019872103e-15), Complex::new(8.881784197001252e-16, 8.881784197001252e-16), Complex::new(-8.881784197001252e-16, 1.7763568394002505e-15)]);

        let n: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];
        let fft = fft_real2_d(&n);

        assert_eq!(3, fft.len());
        assert_eq!(fft[0], [Complex::new(45.0, 0.0), Complex::new(-4.500000000000004, 2.5980762113533125), Complex::new(-4.499999999999993, -2.5980762113533267)]);
        assert_eq!(fft[1], [Complex::new(-13.500000000000007, 7.7942286340599445), Complex::new(2.1094237467877974e-15, 1.1102230246251565e-15), Complex::new(8.881784197001252e-16, 4.6629367034256575e-15)]);
        assert_eq!(fft[2], [Complex::new(-13.49999999999999, -7.794228634059962), Complex::new(0.0, 2.220446049250313e-15), Complex::new(-4.884981308350689e-15, 2.4424906541753444e-15)]);
    }
}