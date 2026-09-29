use glam::Vec3;

/// Orbit camera in patient space (LPS, mm).
///
/// `yaw = 0, pitch = 0` is an anterior-posterior view: the camera sits anterior
/// of the patient (-y), looks along +y with superior (+z) up, so the patient's
/// left (+x) appears on the right of the screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// Rotation about the patient's superior axis, radians.
    pub yaw: f32,
    /// Elevation, radians; positive looks down from above the head.
    pub pitch: f32,
    pub fov_y: f32,
}

const MAX_PITCH: f32 = 1.55;

impl OrbitCamera {
    /// Frames a bounding box given by its centre and diagonal length.
    pub fn framing(center: Vec3, diagonal: f32) -> Self {
        let fov_y = 30f32.to_radians();
        OrbitCamera {
            target: center,
            distance: 0.55 * diagonal / (fov_y / 2.0).tan(),
            yaw: 0.0,
            pitch: 0.0,
            fov_y,
        }
    }

    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * sy, cp * cy, -sp)
    }

    pub fn eye(&self) -> Vec3 {
        self.target - self.forward() * self.distance
    }

    /// Orthonormal (right, up) basis of the image plane.
    pub fn basis(&self) -> (Vec3, Vec3) {
        let forward = self.forward();
        let right = forward.cross(Vec3::Z).normalize();
        (right, right.cross(forward))
    }

    /// Rotates by screen-space drag deltas in radians.
    pub fn orbit(&mut self, d_yaw: f32, d_pitch: f32) {
        self.yaw = (self.yaw + d_yaw).rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch + d_pitch).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Moves the target within the image plane; deltas are fractions of the view height.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let (right, up) = self.basis();
        let view_height = 2.0 * self.distance * (self.fov_y / 2.0).tan();
        self.target += (-right * dx + up * dy) * view_height;
    }

    pub fn zoom(&mut self, factor: f32) {
        self.distance = (self.distance * factor).max(1.0);
    }
}
