// Surface description and reflectance shared by the ray marcher (dvr.comp) and
// the path tracer (pt.comp). Requires common.glsl.

float ggx_distribution(float nh, float a2) {
    float denom = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Lambert diffuse (returned in `diffuse`) and GGX specular (Smith-Schlick
// geometry, Schlick Fresnel; the return value), not yet multiplied by n.l.
vec3 brdf(vec3 n, vec3 v, vec3 l, vec3 albedo, float roughness, float metalness, out vec3 diffuse) {
    vec3 h = normalize(v + l);
    float nv = max(dot(n, v), 1e-4);
    float nl = max(dot(n, l), 1e-4);
    float nh = max(dot(n, h), 0.0);
    float vh = max(dot(v, h), 0.0);
    float a = max(roughness * roughness, 2e-3);
    float d = ggx_distribution(nh, a * a);
    float k = a * 0.5;
    float g = (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));
    // Tissue is water-like (n ~ 1.4): F0 = ((n - 1) / (n + 1))^2 ~ 0.028.
    vec3 f0 = mix(vec3(0.028), albedo, metalness);
    vec3 f = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
    vec3 specular = d * g * f / (4.0 * nv * nl);
    diffuse = (1.0 - f) * (1.0 - metalness) * albedo / PI;
    return specular;
}

// Roughness of the wet film of clear-coated tissue.
const float COAT_ROUGHNESS = 0.1;

// Smith-Schlick geometry term.
float smith(float nv, float nl, float a) {
    float k = a * 0.5;
    return (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));
}

// brdf() of sample `s` under its clear coat: a thin water film (F0 = 0.02,
// smooth) reflects sharp highlights and passes the rest to the tissue below
// (layered like Filament's clear coat).
vec3 brdf_coated(vec3 n, vec3 v, vec3 l, Sample s, out vec3 diffuse) {
    vec3 specular = brdf(n, v, l, s.albedo, s.roughness, s.metalness, diffuse);
    if (s.coat <= 0.0) {
        return specular;
    }
    vec3 h = normalize(v + l);
    float nv = max(dot(n, v), 1e-4);
    float nl = max(dot(n, l), 1e-4);
    float a = COAT_ROUGHNESS * COAT_ROUGHNESS;
    float f = s.coat * (0.02 + 0.98 * pow(1.0 - max(dot(v, h), 0.0), 5.0));
    float coat = ggx_distribution(max(dot(n, h), 0.0), a * a) * smith(nv, nl, a) * f / (4.0 * nv * nl);
    diffuse *= 1.0 - f;
    return specular * (1.0 - f) + vec3(coat);
}

// Interleaved gradient noise, to jitter the ray start and hide step banding.
float ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// Outward normal (from denser towards less dense material; zero if undefined)
// and surfaceness (0 in homogeneous material, 1 on a clear surface) of the
// sample `s` at `idx`, seen from direction `v` (towards the viewer). Returns
// the factor to apply to the sample's extinction: 1, except for ghosted
// structures, which keep only their shell.
//
// `cap_plane`/`cut` come from clip_depth(); samples within `cap_depth` mm of a
// clip plane are shaded as the flat cut face.
float surface_frame(
    vec3 idx, Sample s, vec3 v, int cap_plane, float cut, float cap_depth,
    out vec3 n, out float surfaceness
) {
    uint label_mode = uint(u.labels.x);
    vec3 tex = index_to_tex(idx);
    vec4 nw = textureLod(normal_tex, tex, 0.0);
    float n_len = length(nw.xyz);
    n = n_len > 1e-3 ? nw.xyz / n_len : vec3(0.0);
    surfaceness = nw.w;
    // Skin comes from the CT's air boundary, a real edge: shaded by the CT
    // normal rather than its voxel mask, whose steps would show.
    bool skin = s.label == SKIN_LABEL;
    // (Not for other ghosts: they are shaded by their mask's normal below.)
    if (u.env.y > 0.5 && n_len > 1e-3 && (!s.ghost || skin)) {
        // The precomputed normals are interpolated between voxels and show their
        // grid (slice terraces); the B-spline gradient is smooth.
        vec3 gw = transpose(mat3(u.world_to_index)) * density_gradient_cubic(idx);
        float len = length(gw);
        n = len > 1e-7 ? -gw / len : n;
    }
    // Label boundaries are shaded by the smoothed mask: always when isolating
    // (density edges inside an organ would otherwise dominate), and on the cut
    // faces that hidden labels leave in tint mode.
    // Ghosted samples lie outside the mask and keep their density normal.
    bool hollow = u.label_density[s.label].w < 0.0;
    if (skin) {
        // Keep the CT normal.
    } else if (!s.ghost && label_mode == 3u && !hollow) {
        // Hybrid: the CT gradient where it is strong (real tissue edges and
        // texture), the segmentation's shape elsewhere.
        vec3 g = mask_gradient(idx, 0);
        float g_len = length(g);
        vec3 mask_n = g_len > 1e-5 ? -g / g_len : n;
        vec3 blended = mix(mask_n, n, surfaceness);
        n = length(blended) > 1e-3 ? normalize(blended) : mask_n;
        surfaceness = max(surfaceness, clamp(g_len * 4.0, 0.0, 1.0));
    } else if (!s.ghost
        && (label_mode >= 2u || (label_mode == 1u && textureLod(mask_tex, tex, 0.0).r < 0.98))) {
        vec3 g = mask_gradient(idx, 0);
        float g_len = length(g);
        float mask_surface = clamp(g_len * 4.0, 0.0, 1.0);
        if (label_mode >= 2u || mask_surface > surfaceness) {
            n = g_len > 1e-5 ? -g / g_len : n;
            surfaceness = label_mode >= 2u ? mask_surface : max(surfaceness, mask_surface);
        }
    }

    // Capping: tissue exposed by a clip plane is shaded as a flat cut face.
    if (cap_plane >= 0 && !s.ghost && -cut < cap_depth) {
        n = u.clip_normals[cap_plane].xyz;
        surfaceness = 1.0;
    }

    // Ghosted structures keep their silhouettes (where the surface is seen
    // edge-on) and fade elsewhere, so they give context without hiding what
    // lies behind (after Bruckner et al., "Illustrative context-preserving
    // exploration of volume data", 2006).
    if (s.ghost) {
        vec3 g = mask_gradient(idx, 1);
        float g_len = length(g);
        if (!skin || n_len <= 1e-3) {
            n = g_len > 1e-5 ? -g / g_len : n;
        }
        surfaceness = clamp(g_len * 4.0, 0.0, 1.0);
        // Only the shell (where the ghost mask changes) is drawn, so thick
        // structures such as the lungs do not add up to opaque.
        float edge_on = pow(1.0 - abs(dot(n, v)), 3.0);
        return surfaceness * mix(u.labels.w, 1.0, edge_on);
    }
    return 1.0;
}
