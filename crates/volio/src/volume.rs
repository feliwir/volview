use glam::{DMat3, DMat4, DVec3, UVec3};

/// A scalar CT volume in Hounsfield units.
///
/// Voxel `(i, j, k)` is stored at `hu[i + dims.x * (j + dims.y * k)]`, where `i`
/// runs along the image rows (column index), `j` along the image columns (row
/// index) and `k` across slices.
#[derive(Debug, Clone)]
pub struct Volume {
    pub dims: UVec3,
    /// Voxel spacing in mm along `i`, `j`, `k`.
    pub spacing: DVec3,
    /// Patient-space (LPS, mm) position of voxel `(0, 0, 0)`'s centre.
    pub origin: DVec3,
    /// Columns are the unit patient-space directions of the `i`, `j`, `k` axes.
    pub direction: DMat3,
    pub hu: Vec<i16>,
    pub series_uid: String,
}

impl Volume {
    /// Maps continuous voxel indices to patient space (LPS, mm).
    pub fn index_to_world(&self) -> DMat4 {
        let m = self.direction * DMat3::from_diagonal(self.spacing);
        DMat4::from_cols(
            m.x_axis.extend(0.0),
            m.y_axis.extend(0.0),
            m.z_axis.extend(0.0),
            self.origin.extend(1.0),
        )
    }

    pub fn world_to_index(&self) -> DMat4 {
        self.index_to_world().inverse()
    }

    /// Physical extent covered by voxel centres, in mm.
    pub fn extent_mm(&self) -> DVec3 {
        (self.dims.as_dvec3() - DVec3::ONE).max(DVec3::ZERO) * self.spacing
    }

    pub fn voxel_count(&self) -> usize {
        self.dims.x as usize * self.dims.y as usize * self.dims.z as usize
    }

    #[inline]
    pub fn get(&self, i: u32, j: u32, k: u32) -> i16 {
        self.hu[(i + self.dims.x * (j + self.dims.y * k)) as usize]
    }

    pub fn hu_range(&self) -> (i16, i16) {
        self.hu
            .iter()
            .fold((i16::MAX, i16::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)))
    }
}
