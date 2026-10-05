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

// Roughness of the wet film of clear-coated tissue: a thin film follows the
// tissue's microrelief, so rougher tissue (lung, muscle) glints less sharply
// than smooth capsules and vessel walls.
float coat_roughness(Sample s) {
    return clamp(0.5 * s.roughness, 0.1, 0.25);
}

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
    float a = coat_roughness(s) * coat_roughness(s);
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

// Procedural surface texture of labelled tissue (u.label_surface): solid
// noise in patient space (mm), so it sticks to the anatomy as the camera moves.

// Integer hash (pcg3d; Jarzynski & Olano, "Hash functions for GPU rendering",
// 2020), in [0, 1).
vec3 hash3(ivec3 c) {
    uvec3 v = uvec3(c) * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return vec3(v) * (1.0 / 4294967296.0);
}

float gradient_at(ivec3 corner, vec3 offset) {
    return dot(hash3(corner) * 2.0 - 1.0, offset);
}

// Gradient noise (Perlin, quintic fade), about -1..1 with unit feature size.
float gradient_noise(vec3 p) {
    ivec3 i = ivec3(floor(p));
    vec3 f = fract(p);
    vec3 w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    const ivec3 X = ivec3(1, 0, 0), Y = ivec3(0, 1, 0), Z = ivec3(0, 0, 1);
    float x00 = mix(gradient_at(i, f), gradient_at(i + X, f - vec3(X)), w.x);
    float x10 = mix(gradient_at(i + Y, f - vec3(Y)), gradient_at(i + X + Y, f - vec3(X + Y)), w.x);
    float x01 = mix(gradient_at(i + Z, f - vec3(Z)), gradient_at(i + X + Z, f - vec3(X + Z)), w.x);
    float x11 = mix(gradient_at(i + Y + Z, f - vec3(Y + Z)), gradient_at(i + X + Y + Z, f - vec3(X + Y + Z)), w.x);
    return 1.7 * mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}

// Three octaves of gradient noise, about -1..1.
float fbm(vec3 p) {
    return (gradient_noise(p) + 0.5 * gradient_noise(p * 2.03 + 17.1) + 0.25 * gradient_noise(p * 4.07 + 31.7))
        / 1.75;
}

// Cellular noise (Worley, "A cellular texture basis function", 1996), one
// feature point per unit cell: distance to the nearest point, distance to the
// border with the second nearest cell, and the nearest point's cell.
void cellular(vec3 p, out float nearest, out float border, out ivec3 cell) {
    ivec3 i = ivec3(floor(p));
    vec3 f = fract(p);
    vec3 d1 = vec3(8.0);
    vec3 d2 = vec3(8.0);
    cell = i;
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                ivec3 c = i + ivec3(x, y, z);
                vec3 d = vec3(x, y, z) + hash3(c) - f;
                if (dot(d, d) < dot(d1, d1)) {
                    d2 = d1;
                    d1 = d;
                    cell = c;
                } else if (dot(d, d) < dot(d2, d2)) {
                    d2 = d;
                }
            }
        }
    }
    nearest = length(d1);
    // Distance to the bisecting plane of the two nearest points.
    border = dot(0.5 * (d1 + d2), normalize(d2 - d1));
}

// Pattern 2 (lung): lobules, a few centimetres wide, outlined by septa.
// Returns the distance to the nearest septum (in lobules) and the lobule's cell.
// The cells are warped so the septa wander instead of running dead straight.
float lobule_border(vec3 q, out ivec3 lobule) {
    vec3 warp = vec3(gradient_noise(q * 1.7), gradient_noise(q * 1.7 + 5.2), gradient_noise(q * 1.7 + 9.4));
    float nearest;
    float border;
    cellular(q + 0.2 * warp, nearest, border, lobule);
    return border;
}

// Relief of the texture at `p` (mm) in units of the label's relief height:
// lobules bulge between sunken septa; mottled tissue has gentle fine bumps.
float surface_height(vec3 p, vec4 params) {
    vec3 q = p / params.w;
    if (params.x > 1.5) {
        // Each lobule bulges gently between its septa.
        ivec3 lobule;
        float bulge = smoothstep(0.0, 0.35, lobule_border(q, lobule));
        return 0.7 * bulge + 0.3 * fbm(q * 3.0);
    }
    return 0.6 * fbm(q) + 0.4 * gradient_noise(q * 4.0);
}

vec3 index_to_world(vec3 idx) {
    return inverse(mat3(u.world_to_index)) * (idx - u.world_to_index[3].xyz);
}

bool textured(Sample s) {
    return s.label != 0u && !s.ghost && u.label_surface[s.label].x > 0.5;
}

// Varies the albedo, roughness and wetness of labelled tissue `s` at `idx`
// with its procedural texture.
void apply_surface_texture(vec3 idx, inout Sample s) {
    if (!textured(s)) {
        return;
    }
    vec4 params = u.label_surface[s.label];
    float amount = params.y;
    vec3 p = index_to_world(idx);
    vec3 q = p / params.w;
    vec3 tint;
    float wet;
    float rough;
    if (params.x > 1.5) {
        // Lung: lobules differ slightly in shade, with greyish-mauve patches
        // across them; the septa between them are greyer (stronger in some
        // stretches than others), and dark specks of pigment gather mostly
        // along them.
        float strength = min(1.0, 5.0 * amount);
        ivec3 lobule;
        float border = lobule_border(q, lobule);
        float septum = (1.0 - smoothstep(0.0, 0.06, border))
            * smoothstep(-0.7, 0.3, gradient_noise(q * 6.0));
        float near_septum = 1.0 - smoothstep(0.0, 0.2, border);
        float patches = fbm(q * 1.5);
        float mottle = fbm(q * 4.0);
        tint = vec3(1.0 + amount * (hash3(lobule).x * 2.0 - 1.0 + 0.5 * mottle));
        tint *= mix(vec3(1.0), vec3(0.84, 0.78, 0.88), strength * smoothstep(-0.1, 0.4, -patches));
        tint *= mix(vec3(1.0), vec3(0.6, 0.57, 0.66), strength * septum);
        // Specks, about 0.5-1 mm: one candidate per 3 mm cell, most near a septum.
        float nearest;
        float unused;
        ivec3 speck;
        cellular(p / 3.0, nearest, unused, speck);
        vec3 h = hash3(speck + 7919);
        float speck_mask = (h.x < 0.03 + 0.4 * near_septum ? 1.0 : 0.0)
            * (1.0 - smoothstep(0.12, 0.2, nearest / (0.6 + 0.8 * h.y)));
        tint *= mix(1.0, 0.2, strength * speck_mask);
        wet = 1.0 - 0.3 * near_septum;
        rough = 1.0 + 0.2 * mottle;
    } else {
        // Mottled: patches of slightly different perfusion and fine grain;
        // darker patches are a little more purple.
        float patches = fbm(q);
        float grain = gradient_noise(q * 4.0);
        tint = vec3(1.0 + amount * (0.7 * patches + 0.3 * grain));
        tint *= mix(vec3(1.0), vec3(0.97, 0.94, 1.0), clamp(-patches, 0.0, 1.0) * min(1.0, 5.0 * amount));
        wet = clamp(0.8 + 0.4 * patches, 0.4, 1.2);
        rough = 1.0 + 0.3 * grain;
    }
    s.albedo = clamp(s.albedo * tint, 0.0, 1.0);
    s.roughness = clamp(s.roughness * rough, 0.02, 1.0);
    s.coat = clamp(s.coat * wet, 0.0, 1.0);
}

// Outward normal `n` of labelled tissue `s` at `idx`, tilted by the relief of
// its texture (bump mapping of the height field).
vec3 bump_normal(vec3 idx, Sample s, vec3 n) {
    if (!textured(s) || u.label_surface[s.label].z <= 0.0 || dot(n, n) < 0.5) {
        return n;
    }
    vec4 params = u.label_surface[s.label];
    vec3 p = index_to_world(idx);
    float e = 0.02 * params.w;  // finite-difference step (mm), finer than the septa
    float h = surface_height(p, params);
    vec3 g = vec3(
        surface_height(p + vec3(e, 0.0, 0.0), params) - h,
        surface_height(p + vec3(0.0, e, 0.0), params) - h,
        surface_height(p + vec3(0.0, 0.0, e), params) - h
    ) * (params.z / e);
    return normalize(n - (g - dot(g, n) * n));
}
