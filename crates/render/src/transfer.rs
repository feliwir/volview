/// HU range covered by the density texture and transfer-function LUT.
pub const HU_MIN: f32 = -1024.0;
pub const HU_MAX: f32 = 3071.0;
pub const LUT_SIZE: usize = 4096;

/// A transfer-function control point.
#[derive(Debug, Clone, Copy)]
pub struct ControlPoint {
    pub hu: f32,
    /// Linear RGB albedo.
    pub color: [f32; 3],
    /// Extinction coefficient in 1/mm.
    pub extinction: f32,
}

const fn cp(hu: f32, color: [f32; 3], extinction: f32) -> ControlPoint {
    ControlPoint {
        hu,
        color,
        extinction,
    }
}

/// A piecewise-linear 1D transfer function over HU.
#[derive(Debug, Clone)]
pub struct TransferFunction {
    pub name: &'static str,
    pub points: Vec<ControlPoint>,
}

impl TransferFunction {
    pub fn presets() -> Vec<TransferFunction> {
        vec![
            TransferFunction {
                name: "bone",
                points: vec![
                    cp(-1024.0, [0.0; 3], 0.0),
                    cp(180.0, [0.9, 0.6, 0.45], 0.0),
                    cp(300.0, [0.95, 0.85, 0.7], 0.6),
                    cp(1200.0, [1.0, 0.97, 0.9], 3.0),
                    cp(3071.0, [1.0, 1.0, 1.0], 4.0),
                ],
            },
            TransferFunction {
                name: "angio",
                points: vec![
                    cp(-1024.0, [0.0; 3], 0.0),
                    cp(120.0, [0.7, 0.1, 0.08], 0.0),
                    cp(200.0, [0.85, 0.15, 0.1], 0.5),
                    cp(450.0, [0.95, 0.3, 0.2], 1.5),
                    cp(600.0, [0.95, 0.88, 0.75], 2.5),
                    cp(3071.0, [1.0, 1.0, 1.0], 4.0),
                ],
            },
            TransferFunction {
                name: "soft tissue",
                points: vec![
                    cp(-1024.0, [0.0; 3], 0.0),
                    cp(-200.0, [0.9, 0.75, 0.5], 0.0),
                    cp(-80.0, [0.95, 0.8, 0.5], 0.004),
                    cp(-20.0, [0.8, 0.5, 0.4], 0.004),
                    cp(40.0, [0.75, 0.3, 0.25], 0.03),
                    cp(150.0, [0.85, 0.2, 0.15], 0.15),
                    cp(300.0, [0.95, 0.3, 0.2], 0.8),
                    cp(500.0, [0.95, 0.88, 0.75], 2.0),
                    cp(3071.0, [1.0, 1.0, 1.0], 4.0),
                ],
            },
        ]
    }

    pub fn evaluate(&self, hu: f32) -> [f32; 4] {
        let pts = &self.points;
        let rgba = |p: &ControlPoint| [p.color[0], p.color[1], p.color[2], p.extinction];
        if hu <= pts[0].hu {
            return rgba(&pts[0]);
        }
        for w in pts.windows(2) {
            if hu <= w[1].hu {
                let t = (hu - w[0].hu) / (w[1].hu - w[0].hu).max(1e-6);
                let (a, b) = (rgba(&w[0]), rgba(&w[1]));
                return std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t);
            }
        }
        rgba(pts.last().unwrap())
    }

    /// Samples the function at `LUT_SIZE` evenly spaced HU values over
    /// [`HU_MIN`, `HU_MAX`], matching the density texture's normalisation.
    pub fn to_lut(&self) -> Vec<[f32; 4]> {
        (0..LUT_SIZE)
            .map(|i| {
                let t = i as f32 / (LUT_SIZE - 1) as f32;
                self.evaluate(HU_MIN + t * (HU_MAX - HU_MIN))
            })
            .collect()
    }

    /// Cumulative integrals over normalised density of (albedo * extinction,
    /// extinction), sampled like [`TransferFunction::to_lut`], for pre-integrated
    /// rendering. Accumulated in f64 with the trapezoid rule, since the renderer
    /// takes differences of nearby entries.
    pub fn cumulative_integrals(&self) -> Vec<[f32; 4]> {
        let lut = self.to_lut();
        let weighted = |e: &[f32; 4]| [e[0] * e[3], e[1] * e[3], e[2] * e[3], e[3]].map(f64::from);
        let dx = 1.0 / (LUT_SIZE - 1) as f64;
        let mut sum = [0f64; 4];
        let mut out = vec![[0f32; 4]; LUT_SIZE];
        for i in 1..LUT_SIZE {
            let (a, b) = (weighted(&lut[i - 1]), weighted(&lut[i]));
            for c in 0..4 {
                sum[c] += 0.5 * (a[c] + b[c]) * dx;
            }
            out[i] = sum.map(|v| v as f32);
        }
        out
    }
}

/// Normalises HU to the [0, 1] density stored in the `R16_UNORM` volume.
pub fn hu_to_unorm16(hu: i16) -> u16 {
    let t = ((hu as f32 - HU_MIN) / (HU_MAX - HU_MIN)).clamp(0.0, 1.0);
    (t * 65535.0).round() as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_between_points() {
        let tf = TransferFunction {
            name: "t",
            points: vec![cp(0.0, [0.0; 3], 0.0), cp(100.0, [1.0; 3], 2.0)],
        };
        assert_eq!(tf.evaluate(-50.0), [0.0; 4]);
        assert_eq!(tf.evaluate(50.0), [0.5, 0.5, 0.5, 1.0]);
        assert_eq!(tf.evaluate(500.0), [1.0, 1.0, 1.0, 2.0]);
    }

    #[test]
    fn lut_matches_density_normalisation() {
        assert_eq!(hu_to_unorm16(-1024), 0);
        assert_eq!(hu_to_unorm16(3071), 65535);
        assert_eq!(hu_to_unorm16(-2000), 0);
        let tf = &TransferFunction::presets()[0];
        let lut = tf.to_lut();
        // LUT entry i corresponds to density i / (LUT_SIZE - 1), i.e. HU = i - 1024.
        assert_eq!(lut[1024 + 300], tf.evaluate(300.0));
    }
}
