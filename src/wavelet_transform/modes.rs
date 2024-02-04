
/*

    Padding in general:
                new_length = wavelet_filter_length - 2
               - adding new_length extra values (e.g. zeroes) before the data array
               - adding (wavelet_filter_length - ((new_length) % wavelet_filter_length)) (e.g. zeroes or other padding) after the data array

   Zero padding:
               adding zeroes before and after the array
               e.g.     0, 0 | 1, 2, 3, 4| 0, 0
   Constant padding:
               adding the first value as padding before the array the last value of the data array
                e.g.     1,  1 | 1, 2, 3, 4| 4, 4
    Symmetric padding:
                mirroring the values before and after
                e.g.     2,  1 | 1, 2, 3, 4| 4, 3
     Antisymmetric padding:
                adding additional values by mirroring and negating the values
                e.g.     -2,  -1 | 1, 2, 3, 4| -4, -3
     Reflect padding:
                adding the values by repeating the data first and last values in inverse order
                e.g.      3, 2 | 1, 2, 3, 4| 3, 2
     Antireflect padding:
                signal is extended by reflecting anti-symmetrically about the edge samples. This mode is also known as whole-sample anti-symmetric:
                ... (2*x1 - x3) (2*x1 - x2) | x1 x2 ... xn | (2*xn - xn-1) (2*xn - xn-2)
     Periodic padding:
                adding additional values in as module operator of the array length
                e.g.      3, 4 | 1, 2, 3, 4| 1, 2
      Smooth Padding:
                signal is extended according to the first derivatives calculated on the edges (straight line)

 */

pub enum WaveletMode {
    ZERO,
    CONSTANT,
    SYMMETRIC,
    ANTISYMMETRIC,
    REFLECT,
    ANTIREFLECT,
    PERIODIC,
    SMOOTH,
    PERIODIZATION
}