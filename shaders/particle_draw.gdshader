shader_type canvas_item;

// Distance (as a fraction of the quad's half-width) where the solid core ends and the glow begins.
uniform float core_ratio = 0.75;
uniform float glow_alpha = 0.25;
// Half-width of the particle quad in local units.
uniform float quad_radius = 6.0;
// Smallest on-screen half-width in pixels. Particles that would be smaller (on a scaled-down
// plane) are drawn at this size but fainter, so they don't flicker in and out of existence.
uniform float min_screen_radius = 1.25;

varying float coverage;

void vertex() {
	float pixels_per_unit = length((CANVAS_MATRIX * MODEL_MATRIX * vec4(1.0, 0.0, 0.0, 0.0)).xy);
	float grow = max(1.0, min_screen_radius / (quad_radius * pixels_per_unit));
	VERTEX *= grow;
	coverage = 1.0 / (grow * grow);
}

void fragment() {
	float dist = length(UV - vec2(0.5)) * 2.0;
	float edge = fwidth(dist);
	float core = 1.0 - smoothstep(core_ratio - edge, core_ratio, dist);
	float glow = (1.0 - smoothstep(1.0 - edge, 1.0, dist)) * glow_alpha;
	COLOR.a *= max(core, glow) * coverage;
}
