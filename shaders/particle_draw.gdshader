shader_type canvas_item;

// Distance (as a fraction of the quad's half-width) where the solid core ends and the glow begins.
uniform float core_ratio = 0.75;
uniform float glow_alpha = 0.25;

void fragment() {
	float dist = length(UV - vec2(0.5)) * 2.0;
	float edge = fwidth(dist);
	float core = 1.0 - smoothstep(core_ratio - edge, core_ratio, dist);
	float glow = (1.0 - smoothstep(1.0 - edge, 1.0, dist)) * glow_alpha;
	COLOR.a *= max(core, glow);
}
