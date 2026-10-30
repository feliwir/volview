//! Scenes: a dataset with the viewer's settings and label looks, saved as JSON
//! so a view can be reopened (`--scene`) or rendered again with `--screenshot`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::labels::{LabelSet, Style};
use crate::viewer::Settings;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub dataset: Dataset,
    /// Light map for the `map` environment.
    pub env_map: Option<PathBuf>,
    /// Size in pixels of the image as shown in the viewer; the default
    /// `--screenshot` size, so screenshots keep the framing.
    pub image_size: Option<[u32; 2]>,
    pub settings: Settings,
    pub labels: Vec<LabelLook>,
}

/// Where the volume and its labels come from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dataset {
    pub dicom_dir: PathBuf,
    pub series: Option<String>,
    /// Label map given with `--labels`; otherwise the segmentation cache is used.
    pub labels: Option<PathBuf>,
}

/// Display state of one label, matched by name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelLook {
    pub name: String,
    pub style: Style,
    pub color: [f32; 3],
    pub roughness: f32,
    pub metalness: f32,
    pub subsurface: f32,
    pub coat: f32,
}

impl Scene {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing scene {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
    }
}

pub fn label_looks(labels: &LabelSet) -> Vec<LabelLook> {
    labels
        .entries
        .iter()
        .map(|e| LabelLook {
            name: e.name.clone(),
            style: e.style,
            color: e.color,
            roughness: e.roughness,
            metalness: e.metalness,
            subsurface: e.subsurface,
            coat: e.coat,
        })
        .collect()
}

/// Restores saved looks; labels missing from `looks` keep theirs.
pub fn apply_label_looks(labels: &mut LabelSet, looks: &[LabelLook]) {
    for e in &mut labels.entries {
        if let Some(l) = looks.iter().find(|l| l.name == e.name) {
            e.style = l.style;
            e.color = l.color;
            e.roughness = l.roughness;
            e.metalness = l.metalness;
            e.subsurface = l.subsurface;
            e.coat = l.coat;
        }
    }
}
