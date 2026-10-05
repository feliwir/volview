// Shared declarations for the volume rendering compute shaders.
// Must match `Uniforms` and the descriptor layout in crates/render/src/renderer.rs.

layout(set = 0, binding = 0, std140) uniform Uniforms {
    mat4 world_to_index;  // patient space (mm) -> continuous voxel index
    vec4 eye;             // xyz: camera position (mm)
    vec4 forward;         // xyz: view direction
    vec4 right;           // xyz: image-plane right, scaled by tan(fov/2) * aspect
    vec4 up;              // xyz: image-plane up, scaled by tan(fov/2)
    vec4 dims;            // xyz: volume dimensions in voxels, w: ray step (mm)
    uvec4 flags;          // x: frame index, y: 1 = sRGB-encode output, zw: output size
    vec4 labels;          // x: mode (0 off, 1 tint, 2 isolate, 3 hybrid), y: tint amount, z: label extinction (1/mm), w: ghost opacity
    vec4 light_dir;       // xyz: unit direction towards the key light (world), w: irradiance
    vec4 ambient;         // rgb: ambient radiance, w: exposure multiplier
    vec4 material;        // x: roughness, y: metalness of unlabelled tissue, z: tone mapper (0 none, 1 ACES, 2 AgX), w: key light angular radius
    uvec4 lighting;       // x: shadows on, y: ambient occlusion on, z: pre-integrated transfer function, w: environment (0 dome, 1 studio, 2 light map)
    vec4 light_cells;     // x: shadow, y: ambient-occlusion volume cell size (mm, largest axis)
    uvec4 clip;           // x: number of clip planes, y: 1 = cut only where all planes cut
    vec4 clip_planes[3];  // index space: dot(xyz, idx) - w = signed distance (mm) into the cut side
    vec4 clip_normals[3]; // xyz: world normal pointing into the cut side
    vec4 subsurface;      // x: amount for unlabelled tissue, y: indirect extinction scale, z: blur radius (shadow voxels), w: lookup depth (mm)
    uvec4 path;           // path tracing: x paths accumulated so far, y max bounces, z paths per frame; w: 1 = skip empty space
    vec4 env;             // x: light map rotation (radians), y: 1 = cubic B-spline sampling, z: phase anisotropy g, w: chromatic extinction strength
    vec4 lens;            // path tracing: x lens radius (mm, 0 = pinhole), y focus distance along the view axis (mm)
    vec4 finish;          // camera finishing: x bloom, y vignette, z grain strength, w: 1 = auto exposure
    vec4 tf_bucket_max[16];  // largest transfer-function extinction in each 1/64 of the density range
    vec4 label_colors[256];     // per label: rgb albedo, a: 0 hidden, 1 shown, 2 ghosted
    vec4 label_materials[256];  // per label: x roughness, y metalness, z clear coat (wetness), w subsurface amount
    vec4 label_density[256];    // per label (hybrid): opacity ramp start/end, median, IQR (normalised density; IQR < 0: hollow)
    vec4 label_surface[256];    // per label: surface texture pattern (0 none, 1 mottled, 2 lobular), colour variation, relief (mm), feature size (mm)
} u;

layout(set = 0, binding = 1) uniform sampler3D density_tex;  // HU normalised to [0, 1]
layout(set = 0, binding = 2) uniform sampler1D tf_tex;       // rgb albedo, a extinction (1/mm)
layout(set = 0, binding = 3, rgba8) uniform writeonly image2D out_image;
layout(set = 0, binding = 4) uniform usampler3D label_tex;   // label per voxel (same grid as density)
layout(set = 0, binding = 5) uniform sampler3D normal_tex;   // xyz: world normal, w: surfaceness
layout(set = 0, binding = 6) uniform sampler3D shadow_tex;   // light volume: rgb diffused (subsurface) light, a direct transmittance
layout(set = 0, binding = 7) uniform sampler3D ao_tex;       // ambient visibility
layout(set = 0, binding = 8, rgba16f) uniform image2D hdr_image;  // rgb radiance, a transmittance
layout(set = 0, binding = 9, rgba8_snorm) uniform writeonly image3D normal_image;
layout(set = 0, binding = 10, rgba16f) uniform writeonly image3D shadow_image;
layout(set = 0, binding = 11, r16f) uniform writeonly image3D ao_image;
layout(set = 0, binding = 12) uniform sampler1D preint_tex;  // cumulative integrals of the transfer function
layout(set = 0, binding = 13) uniform sampler3D mask_tex;    // smoothed masks (label modes): r solid, g ghosted
layout(set = 0, binding = 22) uniform sampler3D majorant_tex;
layout(set = 0, binding = 25) uniform sampler2D env_tex;          // light map radiance (mean 1), equirectangular
layout(set = 0, binding = 26) uniform sampler2D env_table_tex;    // light map sampling: see crates/render/src/environment.rs
layout(set = 0, binding = 27) uniform sampler2D env_diffuse_tex;  // light map convolved with a cosine lobe  // extinction bound per brick (see majorant.comp)
// 16 bits: the masks' gradients are surface normals, and 8 bits would band them.
layout(set = 0, binding = 14, rg16) uniform writeonly image3D mask_image;

const float LUT_SIZE = 4096.0;
// Edge length of the bricks of the majorant grid, in voxels.
const int BRICK_SIZE = 8;
const float PI = 3.14159265358979;
// Label of the skin shell (crates/app/src/skin.rs), derived from the CT: its
// outer surface is the CT's air boundary, so it is shaded by the CT's normals.
const uint SKIN_LABEL = 255u;
// Labels from 251 up are derived from the CT after segmenting (251 soft
// tissue, 252 fat: crates/app/src/context.rs; 253 devices, 255 skin:
// crates/app/src/skin.rs): the model's margins know nothing of them.
const uint FIRST_CT_LABEL = 251u;

vec3 index_to_tex(vec3 idx) {
    return (idx + 0.5) / u.dims.xyz;
}

vec3 tex_to_index(vec3 tex) {
    return tex * u.dims.xyz - 0.5;
}

float density_at_index(vec3 idx) {
    return textureLod(density_tex, index_to_tex(idx), 0.0).r;
}

// Cubic B-spline filtering from 8 trilinear fetches (Sigg and Hadwiger, "Fast
// third-order texture filtering", GPU Gems 2, 2005). Per axis, the four B-spline
// weights of a sample are folded into two linear fetches at `h0`/`h1` weighted
// `g0`/`g1`; `dg0`/`dh0`... are the same for the first derivative.
struct CubicTaps {
    vec3 h0, h1, g0, g1;  // value
    vec3 dh0, dh1, dg0, dg1;  // derivative
};

CubicTaps cubic_taps(vec3 idx) {
    vec3 i = floor(idx);
    vec3 f = idx - i;
    vec3 f2 = f * f;
    vec3 f3 = f2 * f;
    vec3 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    vec3 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    vec3 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    vec3 w3 = f3 / 6.0;
    vec3 d0 = -0.5 * (1.0 - f) * (1.0 - f);
    vec3 d1 = 0.5 * f * (3.0 * f - 4.0);
    vec3 d2 = 0.5 * (1.0 - f) * (3.0 * f + 1.0);
    vec3 d3 = 0.5 * f2;
    CubicTaps t;
    t.g0 = w0 + w1;
    t.g1 = w2 + w3;
    t.h0 = i - 1.0 + w1 / t.g0;
    t.h1 = i + 1.0 + w3 / t.g1;
    // d0, d1 <= 0 and d2, d3 >= 0, so each pair folds into one fetch as well.
    t.dg0 = d0 + d1;
    t.dg1 = d2 + d3;
    t.dh0 = i - 1.0 + d1 / min(t.dg0, -1e-7);
    t.dh1 = i + 1.0 + d3 / max(t.dg1, 1e-7);
    return t;
}

// The 8-fetch tensor product with per-axis tap positions and weights.
float cubic_fetch(vec3 h0, vec3 h1, vec3 g0, vec3 g1) {
    float y0z0 = g0.x * density_at_index(vec3(h0.x, h0.y, h0.z)) + g1.x * density_at_index(vec3(h1.x, h0.y, h0.z));
    float y1z0 = g0.x * density_at_index(vec3(h0.x, h1.y, h0.z)) + g1.x * density_at_index(vec3(h1.x, h1.y, h0.z));
    float y0z1 = g0.x * density_at_index(vec3(h0.x, h0.y, h1.z)) + g1.x * density_at_index(vec3(h1.x, h0.y, h1.z));
    float y1z1 = g0.x * density_at_index(vec3(h0.x, h1.y, h1.z)) + g1.x * density_at_index(vec3(h1.x, h1.y, h1.z));
    return g0.z * (g0.y * y0z0 + g1.y * y1z0) + g1.z * (g0.y * y0z1 + g1.y * y1z1);
}

float density_cubic(vec3 idx) {
    CubicTaps t = cubic_taps(idx);
    return cubic_fetch(t.h0, t.h1, t.g0, t.g1);
}

// Gradient (per index unit) of the B-spline filtered density.
vec3 density_gradient_cubic(vec3 idx) {
    CubicTaps t = cubic_taps(idx);
    return vec3(
        cubic_fetch(vec3(t.dh0.x, t.h0.yz), vec3(t.dh1.x, t.h1.yz), vec3(t.dg0.x, t.g0.yz), vec3(t.dg1.x, t.g1.yz)),
        cubic_fetch(vec3(t.h0.x, t.dh0.y, t.h0.z), vec3(t.h1.x, t.dh1.y, t.h1.z), vec3(t.g0.x, t.dg0.y, t.g0.z), vec3(t.g1.x, t.dg1.y, t.g1.z)),
        cubic_fetch(vec3(t.h0.xy, t.dh0.z), vec3(t.h1.xy, t.dh1.z), vec3(t.g0.xy, t.dg0.z), vec3(t.g1.xy, t.dg1.z)));
}

// Cubic B-spline filtered lookup of a 3D texture at texture coordinates `tc`
// (8 linear fetches). Unlike trilinear interpolation it has no kinks at cell
// faces, which show as contour bands where coarse lighting volumes change fast.
vec4 texture_cubic(sampler3D tex, vec3 tc) {
    vec3 size = vec3(textureSize(tex, 0));
    CubicTaps t = cubic_taps(tc * size - 0.5);
    vec3 a = (t.h0 + 0.5) / size;
    vec3 b = (t.h1 + 0.5) / size;
    vec4 z0 = t.g0.y * (t.g0.x * textureLod(tex, vec3(a.x, a.y, a.z), 0.0) + t.g1.x * textureLod(tex, vec3(b.x, a.y, a.z), 0.0))
        + t.g1.y * (t.g0.x * textureLod(tex, vec3(a.x, b.y, a.z), 0.0) + t.g1.x * textureLod(tex, vec3(b.x, b.y, a.z), 0.0));
    vec4 z1 = t.g0.y * (t.g0.x * textureLod(tex, vec3(a.x, a.y, b.z), 0.0) + t.g1.x * textureLod(tex, vec3(b.x, a.y, b.z), 0.0))
        + t.g1.y * (t.g0.x * textureLod(tex, vec3(a.x, b.y, b.z), 0.0) + t.g1.x * textureLod(tex, vec3(b.x, b.y, b.z), 0.0));
    return t.g0.z * z0 + t.g1.z * z1;
}

// Lighting-volume lookup, B-spline filtered with smooth sampling.
vec4 lighting_lookup(sampler3D tex, vec3 tc) {
    return u.env.y > 0.5 ? texture_cubic(tex, tc) : textureLod(tex, tc, 0.0);
}

// Density where surfaces are found and shaded (primary rays): B-spline filtered
// if enabled. Lighting passes keep the cheaper trilinear density_at_index.
float density_view(vec3 idx) {
    return u.env.y > 0.5 ? density_cubic(idx) : density_at_index(idx);
}

// Label of the voxel nearest to continuous index `idx`.
uint label_at_index(vec3 idx) {
    ivec3 i = clamp(ivec3(floor(idx + 0.5)), ivec3(0), ivec3(u.dims.xyz) - 1);
    return texelFetch(label_tex, i, 0).r;
}

vec4 transfer(float density) {
    // LUT entry i holds density i / (LUT_SIZE - 1); address texel centres.
    return textureLod(tf_tex, (density * (LUT_SIZE - 1.0) + 0.5) / LUT_SIZE, 0.0);
}

struct Sample {
    vec3 albedo;
    float extinction;  // 1/mm
    float roughness;
    float metalness;
    uint label;
    bool ghost;     // translucent context structure (see dvr.comp)
    float density;  // normalised density at the sample
    float subsurface;  // share of diffuse light that is scattered below the surface
    float coat;        // strength of a thin glossy film on top (wet tissue)
};

// Transfer-function properties at one density value.
Sample tf_point(float density) {
    vec4 tf = transfer(density);
    return Sample(tf.rgb, tf.a, u.material.x, u.material.y, 0u, false, density, u.subsurface.x, 0.0);
}

// Cumulative integrals (rgb: albedo * extinction, a: extinction) from density 0 to
// `density`, interpolated manually because 32-bit float filtering is optional.
vec4 preint_at(float density) {
    float x = clamp(density, 0.0, 1.0) * (LUT_SIZE - 1.0);
    int i = min(int(x), int(LUT_SIZE) - 2);
    return mix(texelFetch(preint_tex, i, 0), texelFetch(preint_tex, i + 1, 0), x - float(i));
}

// Pre-integrated transfer function for a ray segment along which the density goes
// linearly from `front` to `back` (Engel et al. 2001, in the 1D-table form of Lum
// et al. 2004): the mean extinction is exact, so thin high-opacity ranges of the
// transfer function are not missed between samples; the albedo is the
// extinction-weighted mean over the segment.
Sample tf_segment(float front, float back) {
    if (abs(back - front) < 1e-4) {
        return tf_point(0.5 * (front + back));
    }
    vec4 mean = (preint_at(back) - preint_at(front)) / (back - front);
    vec3 albedo = mean.a > 1e-6 ? mean.rgb / mean.a : transfer(back).rgb;
    return Sample(albedo, max(mean.a, 0.0), u.material.x, u.material.y, 0u, false, back, u.subsurface.x, 0.0);
}

// Fraction of the way from `front` to `back` density at which half of the
// segment's optical depth has accumulated: where an (almost) opaque segment
// actually becomes opaque, found by bisecting the cumulative extinction.
float preint_half_depth(float front, float back) {
    float target = 0.5 * (preint_at(front).a + preint_at(back).a);
    float lo = 0.0;
    float hi = 1.0;
    for (int i = 0; i < 6; i++) {
        float mid = 0.5 * (lo + hi);
        // Cumulative extinction is monotonic in density, so compare by direction.
        bool before = (preint_at(mix(front, back, mid)).a < target) == (back > front);
        lo = before ? mid : lo;
        hi = before ? hi : mid;
    }
    return 0.5 * (lo + hi);
}

bool label_shown(uint label) {
    return u.label_colors[label].a > 0.5;
}

bool label_ghosted(uint label) {
    return label != 0u && u.label_colors[label].a > 1.5;
}

// Whether voxels with `label` are drawn as solid in label `mode` (1 tint, 2 isolate).
// Ghosted labels are not solid: they are drawn translucently outside the mask.
bool label_solid(uint label, uint mode) {
    if (label_ghosted(label)) {
        return false;
    }
    return mode == 1u ? (label == 0u || label_shown(label)) : (label != 0u && label_shown(label));
}

// Median density (normalised, ~300 HU) above which a label is treated as bone.
const float BONE_DENSITY = (300.0 + 1024.0) / 4095.0;

// `s` with the colour and material of `label`.
Sample with_label(Sample s, uint label, uint mode, bool ghost) {
    s.label = label;
    s.ghost = ghost;
    if (label == 0u) {
        return s;  // unlabelled tissue in tint mode
    }
    vec4 color = u.label_colors[label];
    vec4 mat = u.label_materials[label];
    if (mode == 1u) {
        s.albedo = mix(s.albedo, color.rgb, u.labels.y);
    } else {
        s.albedo = color.rgb;
        s.extinction = u.labels.z;
        if (mode == 3u && !ghost && u.label_density[label].w > 0.0) {
            // Hybrid: tissue darker than the structure (e.g. surrounding fat) fades
            // out, so the surface follows the CT edge rather than the coarser
            // segmentation boundary; brightness varies with the CT around the
            // structure's median, which brings back vessels and texture.
            vec4 d = u.label_density[label];
            s.extinction *= smoothstep(d.x, d.y, s.density);
            float t = clamp((s.density - d.z) / d.w, -1.5, 1.5);
            s.albedo = clamp(s.albedo * (1.0 + 0.2 * t), 0.0, 1.0);
            if (d.z > BONE_DENSITY) {
                // Bone: thin plates and partial-volume rims, less dense than the
                // cortex, let the marrow and periosteum show through as a warm,
                // reddish tint; dense cortex stays ivory (as in the HU-coloured
                // bone of cinematic rendering).
                float thin = 1.0 - smoothstep(-1.5, -0.5, t);
                s.albedo *= mix(vec3(1.0), vec3(0.92, 0.6, 0.5), thin);
            }
        }
    }
    s.roughness = mat.x;
    s.metalness = mat.y;
    s.subsurface = mat.w;
    s.coat = mat.z;
    return s;
}

// Applies the label volume to a transfer-function sample. Whether a point is drawn
// solid comes from the smoothed mask (thresholded at 0.5, so boundaries are smooth
// surfaces rather than voxel steps); its label comes from the nearest voxel, or
// near boundaries from a solid corner of the surrounding cell. Outside the mask,
// ghosted labels are drawn by their nearest voxel.
// A label near `idx` that is drawn solid (`ghost` false) or ghosted (`ghost`
// true): the nearest voxel's, else one of the surrounding cell's corners, else
// one up to two voxels away along an axis. The smoothed masks decide the shape
// and may extend past the voxels carrying a label; this only supplies the
// label (colour, material) there, so it must not cut the shape back.
bool drawn_label_near(vec3 idx, uint mode, bool ghost, out uint label) {
    ivec3 hi = ivec3(u.dims.xyz) - 1;
    ivec3 nearest = clamp(ivec3(floor(idx + 0.5)), ivec3(0), hi);
    ivec3 base = ivec3(floor(idx));
    for (int i = 0; i < 21; i++) {
        ivec3 c;
        if (i == 0) {
            c = nearest;
        } else if (i < 9) {
            c = base + ivec3((i - 1) & 1, ((i - 1) >> 1) & 1, (i - 1) >> 2);
        } else {
            int k = i - 9;  // 12 axis neighbours: 3 axes x (+1, -1, +2, -2)
            ivec3 d = ivec3(0);
            d[k % 3] = (k / 3 % 2 == 0 ? 1 : -1) * (k < 6 ? 1 : 2);
            c = nearest + d;
        }
        label = texelFetch(label_tex, clamp(c, ivec3(0), hi), 0).r;
        if (ghost ? label_ghosted(label) : label_solid(label, mode)) {
            return true;
        }
    }
    return false;
}

Sample apply_labels(Sample s, vec3 idx) {
    uint mode = uint(u.labels.x);
    if (mode == 0u) {
        return s;
    }
    vec2 mask = textureLod(mask_tex, index_to_tex(idx), 0.0).rg;
    uint label;
    // Hybrid lets the region reach about a voxel beyond the segmentation, so the CT
    // edge (see with_label) can define the surface there.
    if (mask.r < (mode == 3u ? 0.25 : 0.5)) {
        // Ghosted labels have their own smoothed mask, so their shells are smooth too.
        if (mask.g >= 0.5 && drawn_label_near(idx, mode, true, label)) {
            return with_label(s, label, mode, true);
        }
        s.extinction = 0.0;
        return s;
    }
    if (drawn_label_near(idx, mode, false, label)) {
        return with_label(s, label, mode, false);
    }
    // Only possible far outside any label (the mask is a few voxels wide at most).
    if (mode >= 2u) {
        s.extinction = 0.0;
    }
    return s;
}

// How far `idx` lies inside the cut-away region (mm; > 0 is cut) and which plane
// bounds it there. With "any" planes, a point is cut if it is on the cut side of
// any plane (half-spaces, slabs); with "all", only if it is on the cut side of
// every plane (corner cutaways).
float clip_depth(vec3 idx, out int plane) {
    plane = -1;
    uint count = u.clip.x;
    if (count == 0u) {
        return -1e30;
    }
    bool all_planes = u.clip.y != 0u;
    float depth = all_planes ? 1e30 : -1e30;
    for (uint i = 0u; i < count; i++) {
        float d = dot(u.clip_planes[i].xyz, idx) - u.clip_planes[i].w;
        if (all_planes ? d < depth : d > depth) {
            depth = d;
            plane = int(i);
        }
    }
    return depth;
}

bool is_clipped(vec3 idx) {
    int plane;
    return clip_depth(idx, plane) > 0.0;
}

// Optical and material properties at a point. Shared by rendering, shadows and
// ambient occlusion so they agree on what is opaque.
// Ghosted structures cast no shadows or occlusion, so they do not darken what
// they are meant to reveal.
// Cut-away tissue casts none either.
Sample sample_point(vec3 idx) {
    Sample s = apply_labels(tf_point(density_at_index(idx)), idx);
    if (s.ghost || is_clipped(idx)) {
        s.extinction = 0.0;
    }
    return s;
}

// World-space gradient of a smoothed label mask (0 solid, 1 ghosted); points
// into the masked region.
vec3 mask_gradient(vec3 idx, int channel) {
    vec3 g;
    for (int a = 0; a < 3; a++) {
        vec3 d = vec3(0.0);
        d[a] = 1.0;
        g[a] = textureLod(mask_tex, index_to_tex(idx + d), 0.0)[channel]
             - textureLod(mask_tex, index_to_tex(idx - d), 0.0)[channel];
    }
    return transpose(mat3(u.world_to_index)) * (0.5 * g);
}

// Intersects a ray given in voxel-index space with the volume's bounds.
// Returns (t_near, t_far); empty when t_near >= t_far.
vec2 intersect_volume(vec3 origin, vec3 dir) {
    vec3 lo = vec3(-0.5);
    vec3 hi = u.dims.xyz - 0.5;
    vec3 inv = 1.0 / dir;
    vec3 t0 = (lo - origin) * inv;
    vec3 t1 = (hi - origin) * inv;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    return vec2(max(max(tmin.x, tmin.y), max(tmin.z, 0.0)), min(min(tmax.x, tmax.y), tmax.z));
}

// Extinction bound of the brick containing continuous index `idx`; with empty-
// space skipping off, a bound that never skips.
float brick_majorant(vec3 idx) {
    if (u.path.w == 0u) {
        return 1e30;
    }
    ivec3 b = ivec3(floor((idx + 0.5) / float(BRICK_SIZE)));
    return texelFetch(majorant_tex, clamp(b, ivec3(0), textureSize(majorant_tex, 0) - 1), 0).r;
}

// Ray parameter distance from `idx` to where the ray leaves its brick.
float brick_exit(vec3 idx, vec3 d_idx) {
    vec3 lo = floor((idx + 0.5) / float(BRICK_SIZE)) * float(BRICK_SIZE) - 0.5;
    vec3 hi = lo + float(BRICK_SIZE);
    vec3 d = mix(d_idx, vec3(1e-9), equal(d_idx, vec3(0.0)));
    vec3 t_far = max((lo - idx) / d, (hi - idx) / d);
    return max(min(t_far.x, min(t_far.y, t_far.z)), 0.0);
}

// Primary ray direction (world space, normalised) through pixel `px`.
vec3 camera_ray(vec2 px) {
    vec2 ndc = vec2(px.x / float(u.flags.z), px.y / float(u.flags.w)) * 2.0 - 1.0;
    return normalize(u.forward.xyz + ndc.x * u.right.xyz - ndc.y * u.up.xyz);
}

vec3 linear_to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

vec3 background(vec3 dir) {
    if (u.lighting.w >= 1u) {
        return vec3(0.0);
    }
    float t = 0.5 + 0.5 * -dir.z;
    return mix(vec3(0.02, 0.022, 0.028), vec3(0.08, 0.085, 0.1), t);
}

// Light map coordinates of world direction `d`. The map is fixed to the camera
// (like the key light): its centre column lies ahead, its top row above.
vec2 env_uv(vec3 d) {
    vec3 f = normalize(u.forward.xyz);
    vec3 r = normalize(u.right.xyz);
    vec3 up = normalize(u.up.xyz);
    float phi = atan(dot(d, r), dot(d, f));
    float theta = acos(clamp(dot(d, up), -1.0, 1.0));
    return vec2(fract(0.5 + (phi + u.env.x) / (2.0 * PI)), theta / PI);
}

vec3 env_direction(vec2 uv) {
    float phi = (uv.x - 0.5) * 2.0 * PI - u.env.x;
    float theta = uv.y * PI;
    vec3 f = normalize(u.forward.xyz);
    vec3 r = normalize(u.right.xyz);
    vec3 up = normalize(u.up.xyz);
    return sin(theta) * (sin(phi) * r + cos(phi) * f) + cos(theta) * up;
}

// Light map radiance towards `d` (mean 1 over the sphere, scaled by the ambient).
vec3 env_map_radiance(vec3 d) {
    return u.ambient.rgb * textureLod(env_tex, env_uv(d), 0.0).rgb;
}

// Cosine-weighted mean light map radiance around normal `n`.
vec3 env_map_diffuse(vec3 n) {
    return u.ambient.rgb * textureLod(env_diffuse_tex, env_uv(n), 0.0).rgb;
}
