//! Label volume with per-label display state for the viewer.

use std::collections::BTreeMap;

use rayon::prelude::*;
use render::transfer::{HU_MAX, HU_MIN};
use render::{LabelMode, LabelParams};
use serde::{Deserialize, Serialize};

/// Depth of the skin shell below the body surface.
const SKIN_THICKNESS_MM: f32 = 3.0;
/// Blur width for unlabelled fat and soft tissue between organs (see
/// crate::context::add_context).
const CONTEXT_GAP_MM: f32 = 16.0;
/// Depth below the skin taken as body wall, left out of the context tissue.
const BODY_WALL_MM: f32 = 30.0;
/// How far unlabelled bone may extend from the skull (see crate::bone::add_bone).
const BONE_REACH_MM: f32 = 8.0;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

/// Name fragments of the bones and cartilage (TotalSegmentator "total" and subtasks).
const BONES: &[&str] = &[
    "vertebrae",
    "rib_",
    "sacrum",
    "skull",
    "sternum",
    "cartilage",
    "clavicula",
    "scapula",
    "humerus",
    "femur",
    "hip_",
    "mandible",
    "teeth",
    "hyoid",
    "zygomatic_arch",
    "styloid_process",
    "hard_palate",
];

/// Name fragments of the skeletal muscles. Matched before [`BONES`] and the
/// organs: "thyrohyoid" and "sterno_thyroid" are muscles.
const MUSCLES: &[&str] = &[
    "gluteus",
    "iliopsoas",
    "autochthon",
    "masseter",
    "temporalis",
    "pterygoid",
    "tongue",
    "digastric",
    "sternocleidomastoid",
    "constrictor",
    "trapezius",
    "platysma",
    "levator_scapulae",
    "scalene",
    "sterno_thyroid",
    "thyrohyoid",
    "prevertebral",
    "pectoralis",
    "rectus_abdominis",
    "serratus",
    "latissimus",
    "oblique",
    "erector_spinae",
    "transversospinalis",
    "psoas",
    "quadratus_lumborum",
];

/// Air spaces of the head and neck, outside the lungs.
const AIR_SPACES: &[&str] = &[
    "larynx_air",
    "pharynx",
    "nasal_cavity",
    "sinus_",
    "auditory_canal",
];

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

    /// Measured reduced scattering coefficient of the tissue: μs′(λ) =
    /// a · (λ / 500 nm)^-b with `(a, b)`, a in 1/mm. Means per tissue type from
    /// Jacques, "Optical properties of biological tissues: a review", Phys. Med.
    /// Biol. 58 (2013), table 2. The review has no lung average; lung, organs,
    /// muscle and vessels use its "other soft tissues".
    fn scattering(self, name: &str) -> (f32, f32) {
        const SKIN: (f32, f32) = (4.60, 1.421);
        const BRAIN: (f32, f32) = (2.42, 1.611);
        const BONE: (f32, f32) = (2.29, 0.716);
        const FIBROUS: (f32, f32) = (2.71, 1.627);
        const FATTY: (f32, f32) = (1.84, 0.672);
        const SOFT: (f32, f32) = (1.89, 1.286);
        match self {
            Group::Skin => SKIN,
            Group::Skeleton if name.contains("cartilage") => FIBROUS,
            Group::Skeleton => BONE,
            Group::Context if name == "fat" => FATTY,
            Group::Other if name.contains("brain") => BRAIN,
            _ => SOFT,
        }
    }

    /// Reduced scattering coefficient (1/mm) at the wavelengths standing in for
    /// the red, green and blue channels; w = 1 marks it as measured.
    pub fn optics(self, name: &str) -> [f32; 4] {
        const CHANNEL_NM: [f32; 3] = [650.0, 550.0, 450.0];
        let (a, b) = self.scattering(name);
        let [r, g, bl] = CHANNEL_NM.map(|nm| a * (nm / 500.0).powf(-b));
        [r, g, bl, 1.0]
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
        } else if has(MUSCLES) {
            Group::Muscles
        } else if has(BONES) {
            Group::Skeleton
        } else if has(&[
            "heart",
            "atrial_appendage",
            "aorta",
            "arter",
            "trunk",
            "vein",
            "vena",
        ]) {
            // Before the lungs: lung_arteries and lung_veins are vessels.
            Group::HeartVessels
        } else if has(&["lung", "trachea"]) || has(AIR_SPACES) {
            Group::Lungs
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
            "soft_palate",
            "parotid_gland",
            "submandibular_gland",
        ]) {
            Group::Digestive
        } else if has(&["kidney", "adrenal", "bladder", "prostate"]) {
            Group::Urogenital
        } else {
            Group::Other
        }
    }
}

/// How default label materials are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Bone that the skull label misses ([`crate::bone::BONE`]): it has no entry
    /// of its own but looks like the skull's (by value), apart from its own
    /// density statistics, since thin plates are less dense than the skull.
    thin_bone: Option<(u8, [f32; 4])>,
}

impl LabelSet {
    /// `hu` is the volume the labels belong to; per-label intensity statistics are
    /// taken from it.
    /// Adds a [`crate::skin`] shell to the unlabelled voxels at the body surface.
    pub fn new(mut data: Vec<u8>, names: &BTreeMap<u8, String>, vol: &volio::Volume) -> Self {
        crate::skin::add_skin(&mut data, vol, SKIN_THICKNESS_MM);
        // Only the skull: the other bones are compact, and contrast-filled veins
        // run along the clavicles and first ribs.
        let skull_value = names
            .iter()
            .find_map(|(&v, name)| (name == "skull").then_some(v));
        let skull: [bool; 256] = std::array::from_fn(|v| Some(v as u8) == skull_value);
        crate::bone::add_bone(&mut data, vol, &skull, BONE_REACH_MM);
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
            .filter(|&v| {
                counts[v as usize] > 0 && v != crate::skin::SKIN_OUTER && v != crate::bone::BONE
            })
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
        let bone = crate::bone::BONE as usize;
        let thin_bone = skull_value
            .filter(|_| counts[bone] > 0)
            .map(|skull| (skull, density_stats(&histograms[bone])));
        let mut set = Self {
            data,
            entries,
            soft: None,
            thin_bone,
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
            params.optics[v] = e.group.optics(&e.name);
            params.surface[v] = e.surface;
            params.ghost[v] = e.style == Style::Ghost;
            params.density[v] = e.density;
        }
        if let Some((skull, density)) = self.thin_bone {
            let (from, to) = (skull as usize, crate::bone::BONE as usize);
            params.colors[to] = params.colors[from];
            params.materials[to] = params.materials[from];
            params.optics[to] = params.optics[from];
            params.surface[to] = params.surface[from];
            params.ghost[to] = params.ghost[from];
            params.density[to] = density;
        }
        params
    }

    /// The style of every label value, `None` for values without a label.
    pub fn styles(&self) -> [Option<Style>; 256] {
        let mut style = [None; 256];
        for e in &self.entries {
            style[e.value as usize] = Some(e.style);
        }
        if let Some((skull, _)) = self.thin_bone {
            style[crate::bone::BONE as usize] = style[skull as usize];
        }
        style
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
    // Before the organs and bones they are named after (thyrohyoid, sterno_thyroid).
    (MUSCLES, [0x8e, 0x2a, 0x22], 0.4),
    (
        &["cartilage", "trachea", "lung_airways"],
        [0xd6, 0xd8, 0xcf],
        0.35,
    ),
    // Enamel is glossier and whiter than bone.
    (&["teeth"], [0xf2, 0xee, 0xe0], 0.15),
    (BONES, [0xec, 0xe6, 0xda], 0.42),
    // Mucosa lining the air spaces and the soft palate.
    (AIR_SPACES, [0xd0, 0x8a, 0x84], 0.2),
    (&["soft_palate"], [0xc8, 0x78, 0x72], 0.25),
    (&["eye_lens"], [0xe8, 0xe0, 0xb8], 0.1),
    // Sclera: white, wet and glossy.
    (&["eye_"], [0xf0, 0xec, 0xe4], 0.15),
    (&["optic_nerve"], [0xe6, 0xd8, 0xc8], 0.3),
    // Salivary glands: lobular, yellowish pink.
    (
        &["parotid_gland", "submandibular_gland"],
        [0xd4, 0xa0, 0x80],
        0.35,
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
    // Vessel labels cover the blood-filled lumen (see below). Before "lung":
    // pulmonary arteries carry deoxygenated blood, pulmonary veins oxygenated.
    (&["pulmonary_vein", "lung_veins"], [0xa0, 0x2c, 0x28], 0.25),
    (&["lung_arteries"], [0x3e, 0x2a, 0x4a], 0.25),
    // Healthy adult lung is a greyish pink, not the salmon of solid organs.
    (&["lung"], [0xc8, 0x9c, 0xa0], 0.5),
    // Chambers and myocardium are one label; epicardial fat is not segmented.
    (&["heart", "atrial_appendage"], [0x7a, 0x1e, 0x1e], 0.3),
    // Bright arterial red, dark bluish-purple deoxygenated blood.
    (&["aorta", "arter", "trunk"], [0xb2, 0x3a, 0x30], 0.25),
    (&["vein", "vena"], [0x3e, 0x2a, 0x4a], 0.25),
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
    } else if has(&[
        "pancreas",
        "thyroid_gland",
        "adrenal",
        "parotid",
        "submandibular",
    ]) {
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
    let (base, roughness) = if has(MUSCLES) {
        ([0.45, 0.10, 0.07], 0.5)
    } else if has(BONES) {
        ([0.80, 0.70, 0.52], 0.65)
    } else if has(&["heart"]) {
        ([0.45, 0.04, 0.06], 0.35)
    } else if has(&["aorta", "arter", "trunk"]) {
        ([0.65, 0.05, 0.04], 0.3)
    } else if has(&["vein", "vena"]) {
        ([0.05, 0.12, 0.60], 0.3)
    } else if has(&["lung"]) {
        ([0.75, 0.42, 0.42], 0.7)
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
