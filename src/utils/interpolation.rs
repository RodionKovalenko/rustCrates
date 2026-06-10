

pub fn interpolate_1d(x: f64, x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    if (x1 - x0).abs() <= std::f64::EPSILON * x0.abs().max(x1.abs()).max(1.0) {
        return (y0 + y1) / 2.0; // Avoid division by zero
    }
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

pub fn interpolate_2d(x: f64, y: f64, x0: f64, y0: f64, x1: f64, y1: f64, z00: f64, z01: f64, z10: f64, z11: f64) -> f64 {
    let zx0 = interpolate_1d(x, x0, z00, x1, z10);
    let zx1 = interpolate_1d(x, x0, z01, x1, z11);
    interpolate_1d(y, y0, zx0, y1, zx1)
}

pub fn interpolate_3d(x: f64, y: f64, z: f64, x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64, v000: f64, v001: f64, v010: f64, v011: f64, v100: f64, v101: f64, v110: f64, v111: f64) -> f64 {
    let vz0 = interpolate_2d(x, y, x0, y0, x1, y1, v000, v010, v100, v110);
    let vz1 = interpolate_2d(x, y, x0, y0, x1, y1, v001, v011, v101, v111);
    interpolate_1d(z, z0, vz0, z1, vz1)
}
