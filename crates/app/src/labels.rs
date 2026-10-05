//! Label volume with per-label display state for the viewer.

use std::collections::BTreeMap;

use rayon::prelude::*;
use render::transfer::{HU_MAX, HU_MIN};
use render::{LabelMode, LabelParams};

/// Depth of the skin shell below the body surface.
const SKIN_THICKNESS_MM: f32 = 3.0;
/// Blur width for unlabelled fat and soft tissue between organs (see
/// crate::context::add_context).
const CONTEXT_GAP_MM: f32 = 16.0;
/// Depth below the skin taken as body wall, left out of the context tissue.
const BODY_WALL_MM: f32 = 30.0;

pub struct LabelEntry {
    pub value: u8,
    pub name: String,
    pub voxels: usize,
    /// Linear RGB albedo.
    pub color: [f32; 3],
    pub roughness: f32,
    pub metalness: f32,
    /// Share of diffuse light scattered below the surface.
    pub subsurface: f32,
    /// Strength of the glossy film of wet tissue (clear coat).
    pub coat: f32,
    /// Procedural surface texture, see [`surface_texture`].
    pub surface: [f32; 4],
    pub style: Style,
    pub group: Group,
    /// Hybrid mode: normalised-density ramp (start, end), median and interquartile
    /// range of the voxels carrying this label.
    pub density: [f32; 4],
}

/// How a label is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    Shown,
    /// Translucent context: silhouettes stay, faces fade, so structures behind show.
    Ghost,
    Hidden,
}

impl Style {
    pub fn next(self) -> Self {
        match self {
            Style::Shown => Style::Ghost,
            Style::Ghost => Style::Hidden,
            Style::Hidden => Style::Shown,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Style::Shown => "shown",
            Style::Ghost => "ghost",
            Style::Hidden => "hidden",
        }
    }
}

/// Anatomical system of a TotalSegmentator label, for switching structures together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Group {
    Skin,
    /// Fat and connective tissue between the organs (derived from the CT).
    Context,
    Skeleton,
    Lungs,
    HeartVessels,
    Digestive,
    Urogenital,
    Muscles,
    Other,
}

impl Group {
    /// Parses a group name as used on the command line.
    pub fn parse(s: &str) -> Result<Group, String> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "skin" => Group::Skin,
            "fat" | "context" => Group::Context,
            "skeleton" | "bones" => Group::Skeleton,
            "lungs" | "airway" => Group::Lungs,
            "heart" | "vessels" => Group::HeartVessels,
            "digestive" => Group::Digestive,
            "urogenital" => Group::Urogenital,
            "muscles" => Group::Muscles,
            "other" => Group::Other,
            _ => {
                return Err(format!(
                    "unknown group {s:?}; use skin, fat, skeleton, lungs, heart, digestive, urogenital, muscles or other"
                ));
            }
        })
    }

    pub const ALL: [Group; 9] = [
        Group::Skin,
        Group::Context,
        Group::Skeleton,
        Group::Lungs,
        Group::HeartVessels,
        Group::Digestive,
        Group::Urogenital,
        Group::Muscles,
        Group::Other,
    ];

    /// Typical share of subsurface scattering: soft, water-rich organs scatter
    /// most, bone least. Hand-picked for appearance, not measured.
    pub fn subsurface(self) -> f32 {
        match self {
            Group::Skin => 0.5,
            Group::Context => 0.7,
            Group::Skeleton => 0.15,
            // Mostly air: light does not soften inside it as in solid organs.
            Group::Lungs => 0.15,
            Group::HeartVessels => 0.45,
            Group::Digestive | Group::Urogenital => 0.6,
            Group::Muscles => 0.5,
            Group::Other => 0.5,
        }
    }

    /// How wet the surface looks: serous membranes (pleura, peritoneum,
    /// pericardium) and vessel walls glisten; bone and cartilage are dry.
    pub fn coat(self) -> f32 {
        match self {
            Group::Skeleton => 0.0,
            // Sebum gives skin a faint sheen.
            Group::Skin => 0.2,
            Group::Context => 0.6,
            Group::Lungs => 0.3,
            Group::HeartVessels | Group::Digestive | Group::Urogenital => 0.8,
            Group::Muscles => 0.4,
            Group::Other => 0.5,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Group::Skin => "Skin",
            Group::Context => "Fat & connective tissue",
            Group::Skeleton => "Skeleton",
            Group::Lungs => "Lungs & airway",
            Group::HeartVessels => "Heart & vessels",
            Group::Digestive => "Digestive",
            Group::Urogenital => "Urogenital",
            Group::Muscles => "Muscles",
            Group::Other => "Other",
        }
    }

    fn of(name: &str) -> Group {
        let has = |keys: &[&str]| keys.iter().any(|k| name.contains(k));
        if name == "skin" {
            Group::Skin
        } else if name == "fat" || name == "soft tissue" {
            Group::Context
        } else if has(&[
            "vertebrae",
            "rib_",
            "sacrum",
            "skull",
            "sternum",
            "costal_cartilages",
            "clavicula",
            "scapula",
            "humerus",
            "femur",
            "hip_",
        ]) {
            Group::Skeleton
        } else if has(&["lung", "trachea"]) {
            Group::Lungs
        } else if has(&[
            "heart",
            "atrial_appendage",
            "aorta",
            "arter",
            "trunk",
            "vein",
            "vena",
        ]) {
            Group::HeartVessels
        } else if has(&[
            "liver",
            "spleen",
            "gallbladder",
            "pancreas",
            "stomach",
            "esophagus",
            "bowel",
            "duodenum",
            "colon",
        ]) {
            Group::Digestive
        } else if has(&["kidney", "adrenal", "bladder", "prostate"]) {
            Group::Urogenital
        } else if has(&["gluteus", "iliopsoas", "autochthon"]) {
            Group::Muscles
        } else {
            Group::Other
        }
    }
}

/// How default label materials are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Palette {
    /// Approximate in-situ tissue appearance.
    Anatomical,
    /// Well-separated hues, for telling neighbouring structures apart.
    Distinct,
}

pub struct LabelSet {
    /// One label per voxel, on the volume's grid.
    pub data: Vec<u8>,
    /// Labels that occur in `data`, by value.
    pub entries: Vec<LabelEntry>,
    /// The segmentation model's decision margins, when it came from our segmenter:
    /// used to place boundaries between CT voxels (see mask.comp).
    pub soft: Option<seg::SoftLabels>,
}

impl LabelSet {
    /// `hu` is the volume the labels belong to; per-label intensity statistics are
    /// taken from it.
    /// Adds a [`crate::skin`] shell to the unlabelled voxels at the body surface.
    pub fn new(mut data: Vec<u8>, names: &BTreeMap<u8, String>, vol: &volio::Volume) -> Self {
        crate::skin::add_skin(&mut data, vol, SKIN_THICKNESS_MM);
        // Fat and connective tissue around the organs. Not around bone, skin or
        // the brain, whose surroundings are body wall and meninges, nor the
        // lungs, which fill the chest wall to wall: the heart and great vessels
        // bring in the mediastinum.
        let anchor: [bool; 256] = std::array::from_fn(|v| {
            names.get(&(v as u8)).is_some_and(|name| {
                !matches!(
                    Group::of(name),
                    Group::Skeleton | Group::Skin | Group::Context | Group::Lungs | Group::Other
                )
            })
        });
        crate::context::add_context(
            &mut data,
            vol,
            &anchor,
            CONTEXT_GAP_MM,
            &[crate::skin::SKIN, crate::skin::SKIN_OUTER],
            BODY_WALL_MM,
        );
        let histograms = label_histograms(&data, &vol.hu);
        data.par_iter_mut()
            .filter(|l| **l == crate::skin::SKIN_OUTER)
            .for_each(|l| *l = crate::skin::SKIN);
        let counts: Vec<usize> = histograms
            .iter()
            .map(|h| h.iter().map(|&c| c as usize).sum())
            .collect();
        let entries = (1..=255u8)
            .filter(|&v| counts[v as usize] > 0 && v != crate::skin::SKIN_OUTER)
            .map(|value| {
                let name = if value == crate::skin::SKIN {
                    "skin".to_string()
                } else if value == crate::context::FAT {
                    "fat".to_string()
                } else if value == crate::context::SOFT_TISSUE {
                    "soft tissue".to_string()
                } else if value == crate::skin::DEVICES {
                    "table / devices".to_string()
                } else {
                    names
                        .get(&value)
                        .cloned()
                        .unwrap_or_else(|| format!("label {value}"))
                };
                let group = Group::of(&name);
                LabelEntry {
                    value,
                    name,
                    voxels: counts[value as usize],
                    color: [1.0; 3],
                    roughness: 0.5,
                    metalness: 0.0,
                    subsurface: 0.0,
                    coat: 0.0,
                    surface: [0.0; 4],
                    // The skeleton and skin enclose most organs: ghost them so they show.
                    // Table, head holder and the like only get in the way.
                    style: if value == crate::skin::DEVICES {
                        Style::Hidden
                    } else if matches!(group, Group::Skeleton | Group::Skin) {
                        Style::Ghost
                    } else {
                        Style::Shown
                    },
                    group,
                    density: {
                        let mut d = density_stats(&histograms[value as usize]);
                        // The skin's partial-volume voxels at the air boundary would
                        // mark it hollow, which drops the CT-placed surface.
                        if value == crate::skin::SKIN {
                            d[3] = d[3].abs();
                        }
                        d
                    },
                }
            })
            .collect();
        let mut set = Self {
            data,
            entries,
            soft: None,
        };
        set.apply_palette(Palette::Anatomical);
        set
    }

    pub fn set_group_style(&mut self, group: Group, style: Style) {
        self.entries
            .iter_mut()
            .filter(|e| e.group == group)
            .for_each(|e| e.style = style);
    }

    /// The soft labels in the renderer's terms.
    pub fn soft_grid(&self) -> Option<render::SoftLabelGrid<'_>> {
        self.soft.as_ref().map(|s| render::SoftLabelGrid {
            dims: s.dims,
            src_axis: s.src_axis,
            flip: s.flip,
            data: &s.data,
        })
    }

    /// Resets every label's colour and material to the palette's defaults.
    pub fn apply_palette(&mut self, palette: Palette) {
        for e in &mut self.entries {
            let (color, roughness) = match palette {
                Palette::Anatomical => anatomical_material(&e.name),
                Palette::Distinct => distinct_material(e.value, &e.name),
            };
            (e.color, e.roughness, e.metalness) = (color, roughness, 0.0);
            e.subsurface = e.group.subsurface();
            e.coat = e.group.coat();
            e.surface = match palette {
                Palette::Anatomical => surface_texture(&e.name, e.group),
                Palette::Distinct => [0.0; 4],
            };
        }
    }

    pub fn params(
        &self,
        mode: LabelMode,
        tint: f32,
        extinction: f32,
        ghost_opacity: f32,
        cut_margin_mm: f32,
        smoothing: u32,
    ) -> LabelParams {
        let mut params = LabelParams {
            mode,
            tint,
            extinction,
            ghost_opacity,
            cut_margin_mm,
            smoothing,
            ..Default::default()
        };
        for e in &self.entries {
            let [r, g, b] = e.color;
            let v = e.value as usize;
            params.colors[v] = [r, g, b, if e.style == Style::Hidden { 0.0 } else { 1.0 }];
            params.materials[v] = [e.roughness, e.metalness, e.subsurface, e.coat];
            params.surface[v] = e.surface;
            params.ghost[v] = e.style == Style::Ghost;
            params.density[v] = e.density;
        }
        params
    }
}

/// Histogram bins cover [`HU_MIN`, `HU_MAX`] one HU per bin, matching the
/// renderer's normalised density.
const BINS: usize = (HU_MAX - HU_MIN) as usize + 1;

/// HU histogram of every label value.
fn label_histograms(labels: &[u8], hu: &[i16]) -> Vec<Vec<u32>> {
    assert_eq!(labels.len(), hu.len(), "labels and volume differ in size");
    labels
        .par_chunks(1 << 20)
        .zip(hu.par_chunks(1 << 20))
        .fold(
            || vec![vec![0u32; BINS]; 256],
            |mut acc, (l, h)| {
                for (&l, &h) in l.iter().zip(h) {
                    let bin = (h as f32 - HU_MIN).clamp(0.0, (BINS - 1) as f32) as usize;
                    acc[l as usize][bin] += 1;
                }
                acc
            },
        )
        .reduce(
            || vec![vec![0u32; BINS]; 256],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    x.iter_mut().zip(y).for_each(|(x, y)| *x += y);
                }
                a
            },
        )
}

/// Opacity ramp and texture statistics for hybrid rendering, in normalised
/// density: the ramp runs from the 2nd to the 10th percentile, so tissue darker
/// than the structure (e.g. surrounding fat) is transparent at the real edge.
fn density_stats(histogram: &[u32]) -> [f32; 4] {
    let total: u64 = histogram.iter().map(|&c| c as u64).sum();
    let percentile = |p: f64| {
        let target = (p * total as f64).ceil() as u64;
        let mut seen = 0u64;
        let bin = histogram
            .iter()
            .position(|&c| {
                seen += c as u64;
                seen >= target.max(1)
            })
            .unwrap_or(BINS - 1);
        bin as f32 / (BINS - 1) as f32
    };
    let (p02, p10, p25, p50, p75) = (
        percentile(0.02),
        percentile(0.10),
        percentile(0.25),
        percentile(0.5),
        percentile(0.75),
    );
    // Air-containing structures (lungs, airway, bowel with gas) are labelled with
    // their lumen, so neither their CT edges nor their texture describe the organ
    // surface: they are drawn by the segmentation alone, flagged by a negative IQR.
    let hollow = p25 < (-300.0 - HU_MIN) / (BINS - 1) as f32;
    if hollow {
        return [0.0, 1e-3, p50, -1.0];
    }
    // Keep the ramp at least ~10 HU wide so CT noise does not make it a hard threshold.
    let min_width = 10.0 / (BINS - 1) as f32;
    [
        p02,
        p10.max(p02 + min_width),
        p50,
        (p75 - p25).max(min_width),
    ]
}

/// In-situ appearance of TotalSegmentator's structures: sRGB colour of the
/// tissue surface and roughness. Values follow anatomical atlases and surgical
/// photography; they are hand-picked, not measured reflectances. Wet mucosa and
/// organ capsules are glossy; bone has a satin sheen; lung and muscle are rougher.
const ANATOMICAL: &[(&[&str], [u8; 3], f32)] = &[
    (&["skin"], [0xe8, 0xb8, 0x9c], 0.45),
    // Visceral fat is yellow; connective tissue and thin muscle pinkish red.
    (&["fat"], [0xdc, 0xc4, 0x8c], 0.35),
    (&["soft tissue"], [0xa8, 0x55, 0x48], 0.4),
    (&["costal_cartilages", "trachea"], [0xd6, 0xd8, 0xcf], 0.35),
    (
        &[
            "vertebrae",
            "rib_",
            "sacrum",
            "skull",
            "sternum",
            "clavicula",
            "scapula",
            "humerus",
            "femur",
            "hip_",
        ],
        [0xec, 0xe6, 0xda],
        0.42,
    ),
    (&["liver"], [0x6b, 0x2a, 0x20], 0.25),
    (&["spleen"], [0x5e, 0x22, 0x33], 0.25),
    (&["kidney_cyst"], [0xd4, 0xd0, 0xa8], 0.15),
    (&["kidney"], [0x7a, 0x33, 0x28], 0.3),
    (&["adrenal"], [0xc9, 0xa4, 0x3d], 0.35),
    (&["gallbladder"], [0x4f, 0x6b, 0x2a], 0.2),
    (&["pancreas"], [0xd9, 0xb3, 0x8c], 0.35),
    (&["stomach"], [0xc8, 0x8b, 0x7a], 0.3),
    (&["esophagus"], [0xb8, 0x75, 0x6a], 0.3),
    (&["small_bowel", "duodenum"], [0xd1, 0x9a, 0x88], 0.25),
    (&["colon"], [0xc4, 0x9a, 0x7a], 0.3),
    (&["urinary_bladder"], [0xd8, 0xb9, 0xa6], 0.25),
    (&["prostate"], [0xb7, 0x7f, 0x6e], 0.3),
    (&["thyroid"], [0x8a, 0x3b, 0x2e], 0.3),
    // Healthy adult lung is a greyish pink, not the salmon of solid organs.
    (&["lung"], [0xc8, 0x9c, 0xa0], 0.5),
    // Chambers and myocardium are one label; epicardial fat is not segmented.
    (&["heart", "atrial_appendage"], [0x7a, 0x1e, 0x1e], 0.3),
    // Vessel labels cover the blood-filled lumen: bright arterial red, dark
    // bluish-purple deoxygenated blood (pulmonary veins carry arterial blood).
    (&["pulmonary_vein"], [0xa0, 0x2c, 0x28], 0.25),
    (&["aorta", "arter", "trunk"], [0xb2, 0x3a, 0x30], 0.25),
    (&["vein", "vena"], [0x3e, 0x2a, 0x4a], 0.25),
    (
        &["gluteus", "iliopsoas", "autochthon"],
        [0x8e, 0x2a, 0x22],
        0.4,
    ),
    (&["brain"], [0xd8, 0xb8, 0xb0], 0.3),
    (&["spinal_cord"], [0xe6, 0xd8, 0xc8], 0.3),
];

/// First matching entry of [`ANATOMICAL`], in linear RGB.
fn anatomical_material(name: &str) -> ([f32; 3], f32) {
    let (srgb, roughness) = ANATOMICAL
        .iter()
        .find(|(keys, _, _)| keys.iter().any(|k| name.contains(k)))
        .map_or(([0xb0, 0x70, 0x60], 0.35), |&(_, c, r)| (c, r));
    (srgb.map(|c| srgb_to_linear(c as f32 / 255.0)), roughness)
}

/// Procedural texture of a structure's surface in the anatomical palette:
/// pattern (0 none, 1 mottled, 2 lobular), colour variation (fraction of the
/// albedo), relief height (mm) and feature size (mm). Real organs are neither
/// evenly coloured nor perfectly smooth; hand-picked for appearance.
fn surface_texture(name: &str, group: Group) -> [f32; 4] {
    let has = |keys: &[&str]| keys.iter().any(|k| name.contains(k));
    if has(&["lung"]) {
        // Secondary pulmonary lobules (1-2.5 cm) outlined by interlobular
        // septa, which collect dark pigment (anthracosis) in adults.
        [2.0, 0.12, 0.25, 15.0]
    } else if has(&["pancreas", "thyroid", "adrenal"]) {
        // Glands with small lobules.
        [1.0, 0.12, 0.3, 3.0]
    } else {
        match group {
            // Their surfaces and texture come from the CT.
            Group::Skeleton | Group::Skin | Group::Context => [0.0; 4],
            Group::Lungs => [1.0, 0.05, 0.1, 4.0],
            Group::Muscles => [1.0, 0.08, 0.25, 3.0],
            Group::HeartVessels => [1.0, 0.06, 0.1, 6.0],
            Group::Digestive | Group::Urogenital => [1.0, 0.08, 0.15, 8.0],
            Group::Other => [1.0, 0.06, 0.15, 6.0],
        }
    }
}

/// Category colours with per-label variation, so neighbours (e.g. consecutive
/// vertebrae) stay distinguishable; organs get hues spread by the golden ratio.
fn distinct_material(value: u8, name: &str) -> ([f32; 3], f32) {
    let has = |keys: &[&str]| keys.iter().any(|k| name.contains(k));
    let (base, roughness) = if has(&[
        "vertebrae",
        "rib_",
        "sacrum",
        "skull",
        "sternum",
        "costal_cartilages",
        "clavicula",
        "scapula",
        "humerus",
        "femur",
        "hip_",
    ]) {
        ([0.80, 0.70, 0.52], 0.65)
    } else if has(&["heart"]) {
        ([0.45, 0.04, 0.06], 0.35)
    } else if has(&["aorta", "arter", "trunk"]) {
        ([0.65, 0.05, 0.04], 0.3)
    } else if has(&["vein", "vena"]) {
        ([0.05, 0.12, 0.60], 0.3)
    } else if has(&["lung"]) {
        ([0.75, 0.42, 0.42], 0.7)
    } else if has(&["gluteus", "iliopsoas", "autochthon"]) {
        ([0.45, 0.10, 0.07], 0.5)
    } else {
        return (
            hsv_to_srgb((value as f32 * 0.618_034).fract(), 0.65, 0.85).map(srgb_to_linear),
            0.4,
        );
    };
    let shade = 0.75 + 0.5 * (value as f32 * 0.618_034).fract();
    (base.map(|c| (c * shade).min(1.0)), roughness)
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn hsv_to_srgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s));
    match i as i32 % 6 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}
