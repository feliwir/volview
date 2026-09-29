//! Image-based lighting: an equirectangular HDR environment ("light map"), the
//! tables to importance-sample it and its diffuse convolution.
//!
//! Directions map to the image as in shaders/pt.comp (`env_uv`): the image
//! centre lies ahead of the camera, its top above; columns advance with the
//! azimuth towards the camera's right.

use std::f32::consts::PI;

use rayon::prelude::*;

/// Linear-light RGB radiance, rows from the top (zenith) down.
pub struct EnvironmentMap {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<[f32; 3]>,
}

/// Edge length (columns) of the diffuse-convolved map; half as many rows.
pub(crate) const DIFFUSE_WIDTH: usize = 32;
/// Resolution the environment is reduced to before convolving it.
const DIFFUSE_SOURCE_WIDTH: usize = 128;

/// GPU-ready data derived from an [`EnvironmentMap`].
pub(crate) struct Prepared {
    /// Radiance normalised to a mean of 1 over the sphere, RGBA.
    pub radiance: Vec<[f32; 4]>,
    /// `(width + 1) x height`, RG: for columns `< width` the row's conditional
    /// CDF (inclusive) and the texel's probability; in the last column the
    /// marginal CDF over rows and the row's probability.
    pub table: Vec<[f32; 2]>,
    /// Cosine-weighted mean radiance around each direction (the Lambert
    /// response divided by albedo), `DIFFUSE_WIDTH x DIFFUSE_WIDTH / 2`, RGBA.
    pub diffuse: Vec<[f32; 4]>,
    /// Horizontal image coordinate (0-1) of the brightest region, used to turn
    /// the environment's main light towards the key light.
    pub peak_u: f32,
}

fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Unit direction in the environment's own frame (x right, y up, z ahead) of
/// image coordinates `(u, v)` in 0-1.
fn direction(u: f32, v: f32) -> [f32; 3] {
    let phi = (u - 0.5) * 2.0 * PI;
    let theta = v * PI;
    [
        theta.sin() * phi.sin(),
        theta.cos(),
        theta.sin() * phi.cos(),
    ]
}

impl EnvironmentMap {
    pub(crate) fn prepare(&self) -> Prepared {
        let (w, h) = (self.width, self.height);
        let sin_theta = |y: usize| (PI * (y as f32 + 0.5) / h as f32).sin();

        // Mean radiance over the sphere (texels weighted by their solid angle).
        let (sum, weight) = (0..h)
            .map(|y| {
                let s = sin_theta(y);
                let row = self.rgb[y * w..(y + 1) * w]
                    .iter()
                    .fold([0.0f64; 3], |a, c| [0, 1, 2].map(|i| a[i] + c[i] as f64));
                (row.map(|r| r * s as f64), s as f64 * w as f64)
            })
            .fold(([0.0f64; 3], 0.0), |(a, wa), (r, wr)| {
                ([0, 1, 2].map(|i| a[i] + r[i]), wa + wr)
            });
        let mean = luminance(sum.map(|s| (s / weight) as f32)).max(1e-6);
        let radiance: Vec<[f32; 4]> = self
            .rgb
            .iter()
            .map(|c| [c[0] / mean, c[1] / mean, c[2] / mean, 1.0])
            .collect();

        // Importance sampling proportional to luminance x solid angle.
        let f: Vec<f32> = (0..w * h)
            .map(|i| luminance(self.rgb[i]) * sin_theta(i / w))
            .collect();
        let row_sums: Vec<f64> = (0..h)
            .map(|y| f[y * w..(y + 1) * w].iter().map(|&v| v as f64).sum())
            .collect();
        let total: f64 = row_sums.iter().sum::<f64>().max(1e-12);
        let mut table = vec![[0.0f32; 2]; (w + 1) * h];
        let mut marginal = 0.0f64;
        for y in 0..h {
            let row = &mut table[y * (w + 1)..(y + 1) * (w + 1)];
            let mut acc = 0.0f64;
            for x in 0..w {
                acc += f[y * w + x] as f64;
                row[x] = [
                    if row_sums[y] > 0.0 {
                        (acc / row_sums[y]) as f32
                    } else {
                        (x + 1) as f32 / w as f32
                    },
                    (f[y * w + x] as f64 / total) as f32,
                ];
            }
            row[w - 1][0] = 1.0;
            marginal += row_sums[y];
            row[w] = [(marginal / total) as f32, (row_sums[y] / total) as f32];
        }
        table[(h - 1) * (w + 1) + w][0] = 1.0;

        // Brightest region: the azimuth of the power-weighted mean direction of
        // the brightest texels (squared luminance favours the light sources over
        // large dim areas).
        let (mut px, mut pz) = (0.0f64, 0.0f64);
        for y in 0..h {
            for x in 0..w {
                let l = f[y * w + x] as f64;
                let d = direction((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                px += l * l * d[0] as f64;
                pz += l * l * d[2] as f64;
            }
        }
        let peak_u = 0.5 + (px.atan2(pz) / (2.0 * std::f64::consts::PI)) as f32;

        Prepared {
            diffuse: self.diffuse(&radiance),
            radiance,
            table,
            peak_u,
        }
    }

    /// Convolves the (normalised) radiance with a clamped cosine lobe.
    fn diffuse(&self, radiance: &[[f32; 4]]) -> Vec<[f32; 4]> {
        let (w, h) = (self.width, self.height);
        // Box-filter down to a small map first: the lobe is very wide.
        let sw = DIFFUSE_SOURCE_WIDTH.min(w);
        let sh = (sw / 2).max(1);
        let small: Vec<([f32; 3], [f32; 3])> = (0..sw * sh)
            .map(|i| {
                let (sx, sy) = (i % sw, i / sw);
                let (x0, x1) = (sx * w / sw, ((sx + 1) * w / sw).max(sx * w / sw + 1));
                let (y0, y1) = (sy * h / sh, ((sy + 1) * h / sh).max(sy * h / sh + 1));
                let mut c = [0.0f32; 3];
                for y in y0..y1 {
                    for x in x0..x1 {
                        let r = radiance[y * w + x];
                        c = [c[0] + r[0], c[1] + r[1], c[2] + r[2]];
                    }
                }
                let n = ((y1 - y0) * (x1 - x0)) as f32;
                let u = (sx as f32 + 0.5) / sw as f32;
                let v = (sy as f32 + 0.5) / sh as f32;
                let solid_angle = (2.0 * PI / sw as f32) * (PI / sh as f32) * (PI * v).sin();
                (c.map(|v| v / n * solid_angle), direction(u, v))
            })
            .collect();
        let (dw, dh) = (DIFFUSE_WIDTH, DIFFUSE_WIDTH / 2);
        (0..dw * dh)
            .into_par_iter()
            .map(|i| {
                let n = direction(
                    ((i % dw) as f32 + 0.5) / dw as f32,
                    ((i / dw) as f32 + 0.5) / dh as f32,
                );
                let mut c = [0.0f32; 3];
                for (l, d) in &small {
                    let cos = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
                    if cos > 0.0 {
                        c = [0, 1, 2].map(|k| c[k] + l[k] * cos);
                    }
                }
                [c[0] / PI, c[1] / PI, c[2] / PI, 1.0]
            })
            .collect()
    }
}
