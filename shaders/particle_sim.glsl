#[versions]

spawn = "#define PASS_SPAWN";
count = "#define PASS_COUNT";
scan_blocks = "#define PASS_SCAN_BLOCKS";
scan_totals = "#define PASS_SCAN_TOTALS";
reorder = "#define PASS_REORDER";
forces = "#define PASS_FORCES";
collide = "#define PASS_COLLIDE";

#[compute]
#version 450

VERSION_DEFINES

#define WORKGROUP_SIZE 256
// Grid cells prefix-summed by one scan_blocks workgroup. Must match ParticleCompute.SCAN_BLOCK.
#define SCAN_BLOCK 1024
#define SCAN_ITEMS_PER_THREAD (SCAN_BLOCK / WORKGROUP_SIZE)
// Must match ParticleCompute.MAX_SPECIES.
#define MAX_SPECIES 16
// Largest params.cell_reach. Must match ParticleCompute.MAX_CELL_REACH.
#define MAX_CELL_REACH 2
// Each row of the neighborhood is one span of cells, or two where it wraps around.
#define MAX_RANGES (2 * (2 * MAX_CELL_REACH + 1))
// Bits of params.flags. Must match ParticleCompute.FLAG_*.
#define FLAG_WRITE_INSTANCES 1u
#define FLAG_WRAP 2u

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

// Must match the byte layout built in ParticleCompute._push_constant().
layout(push_constant, std430) uniform Params {
	uint particle_count;
	uint grid_cols;
	uint grid_rows;
	uint species_count;
	uint flags;
	float interaction_radius;
	float repulsion_zone;
	float force_strength;
	float damping;
	float max_speed;
	float delta;
	float edge_margin;
	vec2 plane_size;
	float contact_distance;
	float quadratic_drag;
	float surface_friction;
	float contact_friction;
	uint seed;
	// Fraction of interaction_radius where color forces end, never below repulsion_zone.
	float force_cutoff;
	// Cells the neighbor search reaches out on each side of a particle's cell.
	uint cell_reach;
	float relative_damping;
	uint padding1;
	uint padding2;
} params;

bool wrapping() {
	return (params.flags & FLAG_WRAP) != 0u;
}

// The grid's cells exactly tile the plane, so cells along a wrapped edge line up.
ivec2 cell_coord(vec2 pos) {
	vec2 grid = vec2(params.grid_cols, params.grid_rows);
	ivec2 cell = ivec2(floor(pos / params.plane_size * grid));
	return clamp(cell, ivec2(0), ivec2(grid) - 1);
}

uint cell_index(int x, int y) {
	return uint(x) + uint(y) * params.grid_cols;
}

// Index of the first sorted particle in cell c. cell_start is relative to the cell's scan
// block and block_offset holds where each block starts. c may be one past the last cell.
#define CELL_BEGIN(c) (cell_start[c] + block_offset[(c) / SCAN_BLOCK])

// Keeps a particle on the plane: carries it to the opposite edge when wrapping,
// otherwise bounces its velocity off the edges.
void confine(inout vec2 pos, inout vec2 vel) {
	if (wrapping()) {
		pos = mod(pos, params.plane_size);
		return;
	}
	vec2 lo = vec2(params.edge_margin);
	vec2 hi = max(lo, params.plane_size - vec2(params.edge_margin));
	if (pos.x < lo.x) {
		pos.x = lo.x;
		vel.x = abs(vel.x);
	} else if (pos.x > hi.x) {
		pos.x = hi.x;
		vel.x = -abs(vel.x);
	}
	if (pos.y < lo.y) {
		pos.y = lo.y;
		vel.y = abs(vel.y);
	} else if (pos.y > hi.y) {
		pos.y = hi.y;
		vel.y = -abs(vel.y);
	}
}

uint hash(uint x) {
	x ^= x >> 16u;
	x *= 0x7feb352du;
	x ^= x >> 15u;
	x *= 0x846ca68bu;
	x ^= x >> 16u;
	return x;
}

#ifdef PASS_SPAWN

// Scatters particle_count particles uniformly inside the edge margin, at rest, with a
// species picked uniformly from [0, species_count).

layout(set = 0, binding = 0, std430) restrict writeonly buffer StateOut { vec4 state_out[]; };
layout(set = 0, binding = 1, std430) restrict writeonly buffer SpeciesOut { uint species_out[]; };

float next_random(inout uint rng) {
	rng = hash(rng);
	return float(rng >> 8u) * (1.0 / 16777216.0);
}

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= params.particle_count) {
		return;
	}
	uint rng = hash(i * 0x9e3779b9u + hash(params.seed));
	vec2 lo = vec2(params.edge_margin);
	vec2 hi = max(lo, params.plane_size - lo);
	vec2 pos = mix(lo, hi, vec2(next_random(rng), next_random(rng)));
	state_out[i] = vec4(pos, 0.0, 0.0);
	species_out[i] = hash(rng) % max(params.species_count, 1u);
}

#endif

#ifdef PASS_COUNT

layout(set = 0, binding = 0, std430) restrict readonly buffer StateIn { vec4 state_in[]; };
layout(set = 0, binding = 1, std430) restrict buffer CellCount { uint cell_count[]; };
layout(set = 0, binding = 2, std430) restrict writeonly buffer ParticleCell { uint particle_cell[]; };
layout(set = 0, binding = 3, std430) restrict writeonly buffer ParticleSlot { uint particle_slot[]; };

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= params.particle_count) {
		return;
	}
	ivec2 cell = cell_coord(state_in[i].xy);
	uint c = cell_index(cell.x, cell.y);
	particle_cell[i] = c;
	particle_slot[i] = atomicAdd(cell_count[c], 1u);
}

#endif

#ifdef PASS_SCAN_BLOCKS

// First half of the exclusive prefix sum of cell_count: each workgroup scans SCAN_BLOCK cells
// into cell_start (relative to the block) and writes the block's total to block_offset.
// Covers cells + 1 entries so the end sentinel cell_start[cells] exists; it counts as 0.
// Also zeroes cell_count, leaving it ready for the next grid build without a clear.

layout(set = 0, binding = 0, std430) restrict buffer CellCount { uint cell_count[]; };
layout(set = 0, binding = 1, std430) restrict writeonly buffer CellStart { uint cell_start[]; };
layout(set = 0, binding = 2, std430) restrict writeonly buffer BlockOffset { uint block_offset[]; };

shared uint thread_total[WORKGROUP_SIZE];

void main() {
	uint tid = gl_LocalInvocationID.x;
	uint cells = params.grid_cols * params.grid_rows;
	uint base = gl_WorkGroupID.x * SCAN_BLOCK + tid * SCAN_ITEMS_PER_THREAD;

	uint counts[SCAN_ITEMS_PER_THREAD];
	uint sum = 0u;
	for (uint n = 0u; n < SCAN_ITEMS_PER_THREAD; n++) {
		uint c = base + n;
		uint value = 0u;
		if (c < cells) {
			value = cell_count[c];
			cell_count[c] = 0u;
		}
		counts[n] = value;
		sum += value;
	}

	// Inclusive Hillis-Steele scan of the per-thread totals.
	thread_total[tid] = sum;
	barrier();
	for (uint offset = 1u; offset < WORKGROUP_SIZE; offset <<= 1u) {
		uint add = tid >= offset ? thread_total[tid - offset] : 0u;
		barrier();
		thread_total[tid] += add;
		barrier();
	}

	uint running = thread_total[tid] - sum;
	for (uint n = 0u; n < SCAN_ITEMS_PER_THREAD; n++) {
		uint c = base + n;
		if (c <= cells) {
			cell_start[c] = running;
		}
		running += counts[n];
	}
	if (tid == WORKGROUP_SIZE - 1u) {
		block_offset[gl_WorkGroupID.x] = thread_total[tid];
	}
}

#endif

#ifdef PASS_SCAN_TOTALS

// Second half of the prefix sum, run as a single workgroup: turns the block totals in
// block_offset into each block's exclusive starting offset, in place.

layout(set = 0, binding = 0, std430) restrict buffer BlockOffset { uint block_offset[]; };

shared uint chunk_offset[WORKGROUP_SIZE];

void main() {
	uint tid = gl_LocalInvocationID.x;
	uint blocks = (params.grid_cols * params.grid_rows) / SCAN_BLOCK + 1u;
	uint per_thread = (blocks + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
	uint begin = min(tid * per_thread, blocks);
	uint end = min(begin + per_thread, blocks);

	uint sum = 0u;
	for (uint b = begin; b < end; b++) {
		sum += block_offset[b];
	}
	chunk_offset[tid] = sum;
	barrier();
	for (uint offset = 1u; offset < WORKGROUP_SIZE; offset <<= 1u) {
		uint add = tid >= offset ? chunk_offset[tid - offset] : 0u;
		barrier();
		chunk_offset[tid] += add;
		barrier();
	}

	uint running = chunk_offset[tid] - sum;
	for (uint b = begin; b < end; b++) {
		uint value = block_offset[b];
		block_offset[b] = running;
		running += value;
	}
}

#endif

#ifdef PASS_REORDER

// Copies every particle into cell order, so each cell's particles (and each row of
// neighboring cells) are contiguous in memory for the passes that follow.

layout(set = 0, binding = 0, std430) restrict readonly buffer StateIn { vec4 state_in[]; };
layout(set = 0, binding = 1, std430) restrict readonly buffer SpeciesIn { uint species_in[]; };
layout(set = 0, binding = 2, std430) restrict readonly buffer ParticleCell { uint particle_cell[]; };
layout(set = 0, binding = 3, std430) restrict readonly buffer ParticleSlot { uint particle_slot[]; };
layout(set = 0, binding = 4, std430) restrict readonly buffer CellStart { uint cell_start[]; };
layout(set = 0, binding = 5, std430) restrict readonly buffer BlockOffset { uint block_offset[]; };
layout(set = 0, binding = 6, std430) restrict writeonly buffer StateOut { vec4 state_out[]; };
layout(set = 0, binding = 7, std430) restrict writeonly buffer SpeciesOut { uint species_out[]; };

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= params.particle_count) {
		return;
	}
	uint k = CELL_BEGIN(particle_cell[i]) + particle_slot[i];
	state_out[k] = state_in[i];
	species_out[k] = species_in[i];
}

#endif

#if defined(PASS_FORCES) || defined(PASS_COLLIDE)

// Both passes read particles sorted by the grid in params and write them back in the same order.
layout(set = 0, binding = 0, std430) restrict readonly buffer StateIn { vec4 state_in[]; };
layout(set = 0, binding = 1, std430) restrict writeonly buffer StateOut { vec4 state_out[]; };
layout(set = 0, binding = 2, std430) restrict readonly buffer Species { uint species[]; };
layout(set = 0, binding = 3, std430) restrict readonly buffer CellStart { uint cell_start[]; };
layout(set = 0, binding = 4, std430) restrict readonly buffer BlockOffset { uint block_offset[]; };
layout(set = 0, binding = 5, std430) restrict readonly buffer Palette { vec4 palette[]; };
// MultiMesh instance data: the two rows of the 2D transform, then the color.
layout(set = 0, binding = 6, std430) restrict writeonly buffer Instances { vec4 instances[]; };

void write_instance(uint i, vec2 pos) {
	if ((params.flags & FLAG_WRITE_INSTANCES) == 0u) {
		return;
	}
	instances[i * 3u] = vec4(1.0, 0.0, 0.0, pos.x);
	instances[i * 3u + 1u] = vec4(0.0, 1.0, 0.0, pos.y);
	instances[i * 3u + 2u] = palette[species[i]];
}

// When wrapping, the offset to the nearest copy of the other particle across the edges.
vec2 wrap_offset(vec2 offset) {
	if (wrapping()) {
		offset -= params.plane_size * round(offset / params.plane_size);
	}
	return offset;
}

// Fills ranges with the [begin, end) sorted-index spans holding the particles of the cells
// within params.cell_reach of cell, and returns how many there are. Neighboring cells in one
// row are consecutive, so each row is one span, or two where it wraps around the plane's edge.
// Grids too small to fit the whole neighborhood visit each row or column only once.
int neighbor_ranges(ivec2 cell, out uvec2 ranges[MAX_RANGES]) {
	int cols = int(params.grid_cols);
	int rows = int(params.grid_rows);
	int reach = int(params.cell_reach);
	int width = 2 * reach + 1;
	ivec2 lo = cell - reach;
	ivec2 hi = cell + reach;
	if (!wrapping() || cols < width) {
		lo.x = max(lo.x, 0);
		hi.x = min(hi.x, cols - 1);
	}
	if (!wrapping() || rows < width) {
		lo.y = max(lo.y, 0);
		hi.y = min(hi.y, rows - 1);
	}

	ivec2 spans[2];
	int span_count = 1;
	if (lo.x < 0) {
		spans[0] = ivec2(0, hi.x);
		spans[1] = ivec2(lo.x + cols, cols - 1);
		span_count = 2;
	} else if (hi.x >= cols) {
		spans[0] = ivec2(lo.x, cols - 1);
		spans[1] = ivec2(0, hi.x - cols);
		span_count = 2;
	} else {
		spans[0] = ivec2(lo.x, hi.x);
	}

	int count = 0;
	for (int y = lo.y; y <= hi.y; y++) {
		int row = (y + rows) % rows;
		for (int s = 0; s < span_count; s++) {
			uint first = cell_index(spans[s].x, row);
			uint last = cell_index(spans[s].y, row);
			ranges[count] = uvec2(CELL_BEGIN(first), CELL_BEGIN(last + 1u));
			count++;
		}
	}
	return count;
}

#endif

#ifdef PASS_FORCES

layout(set = 0, binding = 7, std430) restrict readonly buffer Rules { float rules[]; };

shared float rule_table[MAX_SPECIES * MAX_SPECIES];

void main() {
	// The whole workgroup loads the rule table before any thread is allowed to exit.
	for (uint r = gl_LocalInvocationID.x; r < MAX_SPECIES * MAX_SPECIES; r += WORKGROUP_SIZE) {
		rule_table[r] = rules[r];
	}
	barrier();

	uint i = gl_GlobalInvocationID.x;
	if (i >= params.particle_count) {
		return;
	}

	vec4 state = state_in[i];
	vec2 pos = state.xy;
	vec2 vel = state.zw;
	uint rule_row = species[i] * params.species_count;

	float r_max = params.interaction_radius;
	float inv_r_max = 1.0 / r_max;
	float beta = params.repulsion_zone;
	float cutoff = params.force_cutoff;
	float reach = r_max * cutoff;
	float reach_sq = reach * reach;
	float inv_beta = 1.0 / beta;
	float inv_falloff = 1.0 / max(cutoff - beta, 1e-6);
	float inv_reach = 1.0 / reach;
	vec2 acc = vec2(0.0);
	vec2 neighbor_vel = vec2(0.0);
	float neighbor_weight = 0.0;

	uvec2 ranges[MAX_RANGES];
	int range_count = neighbor_ranges(cell_coord(pos), ranges);
	for (int n = 0; n < range_count; n++) {
		for (uint j = ranges[n].x; j < ranges[n].y; j++) {
			vec4 other = state_in[j];
			vec2 offset = wrap_offset(other.xy - pos);
			float dist_sq = dot(offset, offset);
			// Also skips the particle itself, which sits at distance 0.
			if (dist_sq >= reach_sq || dist_sq < 1e-8) {
				continue;
			}
			float inv_dist = inversesqrt(dist_sq);
			float dist = dist_sq * inv_dist;
			float r = dist * inv_r_max;

			// Nearer neighbors drag harder on the particle's velocity.
			float weight = 1.0 - dist * inv_reach;
			neighbor_vel += other.zw * weight;
			neighbor_weight += weight;

			// Close range always repels; beyond it the color rule peaks midway to the cutoff
			// and fades out by it.
			float f;
			if (r < beta) {
				f = r * inv_beta - 1.0;
			} else {
				float attraction = rule_table[rule_row + species[j]];
				f = attraction * max(1.0 - abs(2.0 * r - cutoff - beta) * inv_falloff, 0.0);
			}
			acc += offset * (f * inv_dist);
		}
	}

	if (neighbor_weight > 0.0) {
		// Exact solution of dv/dt = k * sum(weight * (v_j - v)) with the neighbors' velocities
		// held fixed, so strong damping only brings the particle to their weighted mean velocity.
		vec2 mean_vel = neighbor_vel / neighbor_weight;
		vel = mix(vel, mean_vel, 1.0 - exp(-params.relative_damping * neighbor_weight * params.delta));
	}
	vel += acc * params.force_strength * params.delta;
	vel *= params.damping;
	float speed = length(vel);
	if (speed > 0.0) {
		// Exact solution of dv/dt = -k v^2 over the step, so high drag can't flip the velocity.
		float new_speed = speed / (1.0 + params.quadratic_drag * speed * params.delta);
		// Constant deceleration that stops a particle rather than reversing it, so weak forces
		// can't overcome it and the particle stays at rest.
		new_speed = max(new_speed - params.surface_friction * params.delta, 0.0);
		new_speed = min(new_speed, params.max_speed);
		vel *= new_speed / speed;
	}
	pos += vel * params.delta;
	confine(pos, vel);

	state_out[i] = vec4(pos, vel);
	write_instance(i, pos);
}

#endif

#ifdef PASS_COLLIDE

// One relaxation iteration of hard-disc collisions: overlapping pairs are pushed apart
// along their contact normal and stop moving into each other.

// Separation direction for two particles at exactly the same spot. Pseudo-random per pair,
// and opposite for i and j so they split apart instead of moving together.
vec2 coincident_normal(uint i, uint j) {
	uint h = hash((min(i, j) * 73856093u) ^ (max(i, j) * 19349663u));
	float angle = float(h) * (6.28318530718 / 4294967296.0);
	vec2 dir = vec2(cos(angle), sin(angle));
	return i < j ? dir : -dir;
}

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= params.particle_count) {
		return;
	}

	vec4 state = state_in[i];
	vec2 pos = state.xy;
	vec2 vel = state.zw;
	float contact = params.contact_distance;
	float contact_sq = contact * contact;
	vec2 push = vec2(0.0);
	vec2 vel_fix = vec2(0.0);

	uvec2 ranges[MAX_RANGES];
	int range_count = neighbor_ranges(cell_coord(pos), ranges);
	for (int n = 0; n < range_count; n++) {
		for (uint j = ranges[n].x; j < ranges[n].y; j++) {
			if (j == i) {
				continue;
			}
			vec4 other = state_in[j];
			vec2 offset = wrap_offset(other.xy - pos);
			float dist_sq = dot(offset, offset);
			if (dist_sq >= contact_sq) {
				continue;
			}
			float dist = sqrt(dist_sq);
			vec2 normal = dist > 1e-4 ? offset / dist : coincident_normal(i, j);
			// Each particle of the pair resolves half the overlap.
			push -= normal * ((contact - dist) * 0.5);
			// Cancel the closing speed along the normal (perfectly inelastic, split evenly).
			vec2 rel_vel = other.zw - vel;
			float closing = dot(rel_vel, normal);
			if (closing < 0.0) {
				vel_fix += normal * (closing * 0.5);
				// Coulomb friction: the sliding speed between the pair is reduced by at most
				// contact_friction times the normal impulse, never past matching speeds.
				vec2 slip = rel_vel - normal * closing;
				float slip_speed = length(slip);
				if (slip_speed > 1e-6) {
					float max_change = params.contact_friction * (-closing * 0.5);
					vel_fix += slip * min(0.5, max_change / slip_speed);
				}
			}
		}
	}

	pos += push;
	vel += vel_fix;
	confine(pos, vel);

	state_out[i] = vec4(pos, vel);
	write_instance(i, pos);
}

#endif
