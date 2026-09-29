use super::*;
use dicom_core::{DataElement, PrimitiveValue, VR};
use dicom_object::{FileMetaTableBuilder, InMemDicomObject};

const CT_IMAGE_STORAGE: &str = "1.2.840.10008.5.1.4.1.1.2";
const EXPLICIT_VR_LE: &str = "1.2.840.10008.1.2.1";

struct Synth {
    series_uid: &'static str,
    cols: u32,
    rows: u32,
    pixel_spacing: (f64, f64),
    iop: [f64; 6],
    origin: DVec3,
    slice_spacing: f64,
}

impl Default for Synth {
    fn default() -> Self {
        Synth {
            series_uid: "1.2.3.4",
            cols: 4,
            rows: 3,
            pixel_spacing: (0.8, 0.7),
            iop: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            origin: DVec3::new(-10.0, 20.0, 30.0),
            slice_spacing: 2.5,
        }
    }
}

/// Stored value for voxel (i, j, k); HU = stored - 1024.
fn stored(i: u32, j: u32, k: u32) -> i16 {
    (i + 10 * j + 100 * k) as i16
}

fn write_slice(dir: &Path, s: &Synth, k: u32, file_name: &str) {
    let row = DVec3::new(s.iop[0], s.iop[1], s.iop[2]);
    let col = DVec3::new(s.iop[3], s.iop[4], s.iop[5]);
    let pos = s.origin + row.cross(col) * s.slice_spacing * k as f64;
    let sop_uid = format!("{}.{}", s.series_uid, k + 1);
    let ds = |v: String| PrimitiveValue::from(v);
    let join = |v: &[f64]| {
        v.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join("\\")
    };

    let mut pixels = Vec::with_capacity((s.rows * s.cols * 2) as usize);
    for j in 0..s.rows {
        for i in 0..s.cols {
            pixels.extend_from_slice(&stored(i, j, k).to_le_bytes());
        }
    }

    let obj = InMemDicomObject::from_element_iter([
        DataElement::new(
            tags::SOP_CLASS_UID,
            VR::UI,
            PrimitiveValue::from(CT_IMAGE_STORAGE),
        ),
        DataElement::new(
            tags::SOP_INSTANCE_UID,
            VR::UI,
            PrimitiveValue::from(sop_uid.as_str()),
        ),
        DataElement::new(tags::MODALITY, VR::CS, PrimitiveValue::from("CT")),
        DataElement::new(
            tags::SERIES_INSTANCE_UID,
            VR::UI,
            PrimitiveValue::from(s.series_uid),
        ),
        DataElement::new(
            tags::IMAGE_POSITION_PATIENT,
            VR::DS,
            ds(join(&pos.to_array())),
        ),
        DataElement::new(tags::IMAGE_ORIENTATION_PATIENT, VR::DS, ds(join(&s.iop))),
        DataElement::new(
            tags::PIXEL_SPACING,
            VR::DS,
            ds(join(&[s.pixel_spacing.0, s.pixel_spacing.1])),
        ),
        DataElement::new(tags::SAMPLES_PER_PIXEL, VR::US, PrimitiveValue::from(1u16)),
        DataElement::new(
            tags::PHOTOMETRIC_INTERPRETATION,
            VR::CS,
            PrimitiveValue::from("MONOCHROME2"),
        ),
        DataElement::new(tags::ROWS, VR::US, PrimitiveValue::from(s.rows as u16)),
        DataElement::new(tags::COLUMNS, VR::US, PrimitiveValue::from(s.cols as u16)),
        DataElement::new(tags::BITS_ALLOCATED, VR::US, PrimitiveValue::from(16u16)),
        DataElement::new(tags::BITS_STORED, VR::US, PrimitiveValue::from(16u16)),
        DataElement::new(tags::HIGH_BIT, VR::US, PrimitiveValue::from(15u16)),
        DataElement::new(
            tags::PIXEL_REPRESENTATION,
            VR::US,
            PrimitiveValue::from(1u16),
        ),
        DataElement::new(tags::RESCALE_INTERCEPT, VR::DS, ds("-1024".into())),
        DataElement::new(tags::RESCALE_SLOPE, VR::DS, ds("1".into())),
        DataElement::new(tags::PIXEL_DATA, VR::OW, PrimitiveValue::from(pixels)),
    ]);
    obj.with_meta(FileMetaTableBuilder::new().transfer_syntax(EXPLICIT_VR_LE))
        .unwrap()
        .write_to_file(dir.join(file_name))
        .unwrap();
}

fn write_series(dir: &Path, s: &Synth, n: u32) {
    // Name files in reverse so directory order differs from spatial order.
    for k in 0..n {
        write_slice(dir, s, k, &format!("img{:03}.dcm", n - k));
    }
}

#[test]
fn loads_and_sorts_shuffled_series() {
    let dir = tempfile::tempdir().unwrap();
    let s = Synth::default();
    write_series(dir.path(), &s, 5);
    std::fs::write(dir.path().join("README.txt"), "not dicom").unwrap();

    let vol = load_series(dir.path(), &LoadOptions::default()).unwrap();
    assert_eq!(vol.dims, UVec3::new(4, 3, 5));
    assert!((vol.spacing - DVec3::new(0.7, 0.8, 2.5)).length() < 1e-9);
    assert!((vol.origin - s.origin).length() < 1e-9);
    for k in 0..5 {
        for j in 0..3 {
            for i in 0..4 {
                assert_eq!(
                    vol.get(i, j, k),
                    stored(i, j, k) - 1024,
                    "voxel {i},{j},{k}"
                );
            }
        }
    }

    // Voxel (1, 2, 3) sits 1*0.7 mm along x, 2*0.8 mm along y, 3*2.5 mm along z.
    let w = vol
        .index_to_world()
        .transform_point3(DVec3::new(1.0, 2.0, 3.0));
    assert!((w - (s.origin + DVec3::new(0.7, 1.6, 7.5))).length() < 1e-9);
}

#[test]
fn oblique_orientation_geometry() {
    let dir = tempfile::tempdir().unwrap();
    let c = std::f64::consts::FRAC_1_SQRT_2;
    // Coronal-ish: rows along +x, columns along -z rotated 45° about x.
    let s = Synth {
        iop: [1.0, 0.0, 0.0, 0.0, c, -c],
        ..Default::default()
    };
    write_series(dir.path(), &s, 3);
    let vol = load_series(dir.path(), &LoadOptions::default()).unwrap();
    let normal = DVec3::new(0.0, c, c);
    assert!((vol.direction.z_axis - normal).length() < 1e-9);
    let last = vol
        .index_to_world()
        .transform_point3(DVec3::new(0.0, 0.0, 2.0));
    assert!((last - (s.origin + normal * 5.0)).length() < 1e-9);
}

#[test]
fn picks_largest_series_and_honours_uid() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a");
    let b = dir.path().join("b");
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&b).unwrap();
    write_series(
        &a,
        &Synth {
            series_uid: "1.1",
            ..Default::default()
        },
        3,
    );
    write_series(
        &b,
        &Synth {
            series_uid: "2.2",
            ..Default::default()
        },
        6,
    );

    let listed = list_series(dir.path()).unwrap();
    assert_eq!(listed.len(), 2);
    assert_eq!(listed[0].uid, "2.2");

    assert_eq!(
        load_series(dir.path(), &LoadOptions::default())
            .unwrap()
            .dims
            .z,
        6
    );
    let opts = LoadOptions {
        series_uid: Some("1.1".into()),
    };
    assert_eq!(load_series(dir.path(), &opts).unwrap().dims.z, 3);
}

#[test]
fn ignores_duplicate_copies() {
    let dir = tempfile::tempdir().unwrap();
    let s = Synth::default();
    for copy in ["a", "a (Copy)", "b"] {
        let d = dir.path().join(copy);
        std::fs::create_dir_all(&d).unwrap();
        write_series(&d, &s, 4);
    }
    let listed = list_series(dir.path()).unwrap();
    assert_eq!(listed[0].slices, 4);
    assert_eq!(
        load_series(dir.path(), &LoadOptions::default())
            .unwrap()
            .dims
            .z,
        4
    );
}

#[test]
fn rejects_missing_slice() {
    let dir = tempfile::tempdir().unwrap();
    let s = Synth::default();
    for k in [0, 1, 2, 4, 5] {
        write_slice(dir.path(), &s, k, &format!("{k}.dcm"));
    }
    let err = load_series(dir.path(), &LoadOptions::default()).unwrap_err();
    assert!(matches!(err, Error::Inconsistent(_)), "{err}");
}

#[test]
fn empty_folder_errors() {
    let dir = tempfile::tempdir().unwrap();
    assert!(matches!(
        load_series(dir.path(), &LoadOptions::default()),
        Err(Error::NoImages(_))
    ));
}
