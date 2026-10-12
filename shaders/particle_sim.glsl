#[versions]

count = "#define PASS_COUNT";
scan = "#define PASS_SCAN";
scatter = "#define PASS_SCATTER";
forces = "#define PASS_FORCES";
collide = "#define PASS_COLLIDE";

#[compute]
#version 450

VERSION_DEFINES

#define WORKGROUP_SIZE 256

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

// Must match the byte layout built in ParticleCompute._push_constant().
layout(push_constant, std430) uniform Params {
	uint particle_count;
	uint grid_cols;
	uint grid_rows;
	uint species_count;
	float cell_size;
	float interaction_radius;
	float repulsion_zone;
	float force_strength;
	float damping;
	float max_speed;
	float delta;
	float edge_margin;
	vec2 plane_size;
	float contact_distance;
	float padding;
} params;

ivec2 cell_coord(vec2 pos) {
	ivec2 cell = ivec2(floor(pos / params.cell_size));
	return clamp(cell, ivec2(0), ivec2(int(params.grid_cols) - 1, int(params.grid_rows) - 1));
}

// Keeps a particle inside the plane, bouncing its velocity off the edges.
void confine(inout vec2 pos, inout vec2 vel) {
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
	uint c = uint(cell.x) + uint(cell.y) * params.grid_cols;
	particle_cell[i] = c;
	particle_slot[i] = atomicAdd(cell_count[c], 1u);
}

#endif

#ifdef PASS_SCAN

// Exclusive prefix sum of cell_count into cell_start, run as a single workgroup.
// Each thread sums a contiguous chunk of cells, the chunk totals are scanned, then
// each thread writes its chunk's offsets.

layout(set = 0, binding = 0, std430) restrict readonly buffer CellCount { uint cell_count[]; };
layout(set = 0, binding = 1, std430) restrict writeonly buffer CellStart { uint cell_start[]; };

shared uint chunk_offset[WORKGROUP_SIZE];

void main() {
	uint tid = gl_LocalInvocationID.x;
	uint cells = params.grid_cols * params.grid_rows;
	uint per_thread = (cells + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
	uint begin = min(tid * per_thread, cells);
	uint end = min(begin + per_thread, cells);

	uint sum = 0u;
	for (uint c = begin; c < end; c++) {
		sum += cell_count[c];
	}
	chunk_offset[tid] = sum;
	barrier();

	if (tid == 0u) {
		uint running = 0u;
		for (uint t = 0u; t < WORKGROUP_SIZE; t++) {
			uint value = chunk_offset[t];
			chunk_offset[t] = running;
			running += value;
		}
	}
	barrier();

	uint running = chunk_offset[tid];
	for (uint c = begin; c < end; c++) {
		cell_start[c] = running;
		running += cell_count[c];
	}
	if (tid == WORKGROUP_SIZE - 1u) {
		cell_start[cells] = running;
	}
}

#endif

#ifdef PASS_SCATTER

layout(set = 0, binding = 0, std430) restrict readonly buffer ParticleCell { uint particle_cell[]; };
layout(set = 0, binding = 1, std430) restrict readonly buffer ParticleSlot { uint particle_slot[]; };
layout(set = 0, binding = 2, std430) restrict readonly buffer CellStart { uint cell_start[]; };
layout(set = 0, binding = 3, std430) restrict writeonly buffer Sorted { uint sorted_index[]; };

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= params.particle_count) {
		return;
	}
	sorted_index[cell_start[particle_cell[i]] + particle_slot[i]] = i;
}

#endif

#ifdef PASS_FORCES

layout(set = 0, binding = 0, std430) restrict readonly buffer StateIn { vec4 state_in[]; };
layout(set = 0, binding = 1, std430) restrict writeonly buffer StateOut { vec4 state_out[]; };
layout(set = 0, binding = 2, std430) restrict readonly buffer Species { uint species[]; };
layout(set = 0, binding = 3, std430) restrict readonly buffer Rules { float rules[]; };
layout(set = 0, binding = 4, std430) restrict readonly buffer CellStart { uint cell_start[]; };
layout(set = 0, binding = 5, std430) restrict readonly buffer Sorted { uint sorted_index[]; };

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= params.particle_count) {
		return;
	}

	vec4 state = state_in[i];
	vec2 pos = state.xy;
	vec2 vel = state.zw;
	uint rule_row = species[i] * params.species_count;

	float r_max = params.interaction_radius;
	float r_max_sq = r_max * r_max;
	float beta = params.repulsion_zone;
	ivec2 cell = cell_coord(pos);
	int max_col = int(params.grid_cols) - 1;
	int max_row = int(params.grid_rows) - 1;
	vec2 acc = vec2(0.0);

	for (int ny = max(cell.y - 1, 0); ny <= min(cell.y + 1, max_row); ny++) {
		for (int nx = max(cell.x - 1, 0); nx <= min(cell.x + 1, max_col); nx++) {
			uint c = uint(nx) + uint(ny) * params.grid_cols;
			uint end = cell_start[c + 1u];
			for (uint k = cell_start[c]; k < end; k++) {
				uint j = sorted_index[k];
				if (j == i) {
					continue;
				}
				vec2 offset = state_in[j].xy - pos;
				float dist_sq = dot(offset, offset);
				if (dist_sq >= r_max_sq || dist_sq < 1e-8) {
					continue;
				}
				float dist = sqrt(dist_sq);
				float r = dist / r_max;
				// Close range always repels; beyond it the color rule peaks mid-range and fades out.
				float f;
				if (r < beta) {
					f = r / beta - 1.0;
				} else {
					float attraction = rules[rule_row + species[j]];
					f = attraction * (1.0 - abs(2.0 * r - 1.0 - beta) / (1.0 - beta));
				}
				acc += offset * (f / dist);
			}
		}
	}

	vel += acc * params.force_strength * params.delta;
	vel *= params.damping;
	float speed = length(vel);
	if (speed > params.max_speed) {
		vel *= params.max_speed / speed;
	}
	pos += vel * params.delta;
	confine(pos, vel);

	state_out[i] = vec4(pos, vel);
}

#endif

#ifdef PASS_COLLIDE

// One relaxation iteration of hard-disc collisions: overlapping pairs are pushed apart
// along their contact normal and stop moving into each other. Also writes the MultiMesh,
// since this is always the last pass of a step.

layout(set = 0, binding = 0, std430) restrict readonly buffer StateIn { vec4 state_in[]; };
layout(set = 0, binding = 1, std430) restrict writeonly buffer StateOut { vec4 state_out[]; };
layout(set = 0, binding = 2, std430) restrict readonly buffer Species { uint species[]; };
layout(set = 0, binding = 3, std430) restrict readonly buffer CellStart { uint cell_start[]; };
layout(set = 0, binding = 4, std430) restrict readonly buffer Sorted { uint sorted_index[]; };
layout(set = 0, binding = 5, std430) restrict readonly buffer Palette { vec4 palette[]; };
// MultiMesh instance data: 8 floats of 2D transform followed by 4 floats of color.
layout(set = 0, binding = 6, std430) restrict writeonly buffer Instances { float instances[]; };

// Separation direction for two particles at exactly the same spot. Pseudo-random per pair,
// and opposite for i and j so they split apart instead of moving together.
vec2 coincident_normal(uint i, uint j) {
	uint h = (min(i, j) * 73856093u) ^ (max(i, j) * 19349663u);
	h ^= h >> 16u;
	h *= 0x7feb352du;
	h ^= h >> 15u;
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

	if (contact > 0.0) {
		float contact_sq = contact * contact;
		ivec2 cell = cell_coord(pos);
		int max_col = int(params.grid_cols) - 1;
		int max_row = int(params.grid_rows) - 1;
		vec2 push = vec2(0.0);
		vec2 vel_fix = vec2(0.0);

		for (int ny = max(cell.y - 1, 0); ny <= min(cell.y + 1, max_row); ny++) {
			for (int nx = max(cell.x - 1, 0); nx <= min(cell.x + 1, max_col); nx++) {
				uint c = uint(nx) + uint(ny) * params.grid_cols;
				uint end = cell_start[c + 1u];
				for (uint k = cell_start[c]; k < end; k++) {
					uint j = sorted_index[k];
					if (j == i) {
						continue;
					}
					vec4 other = state_in[j];
					vec2 offset = other.xy - pos;
					float dist_sq = dot(offset, offset);
					if (dist_sq >= contact_sq) {
						continue;
					}
					float dist = sqrt(dist_sq);
					vec2 normal = dist > 1e-4 ? offset / dist : coincident_normal(i, j);
					// Each particle of the pair resolves half the overlap.
					push -= normal * ((contact - dist) * 0.5);
					// Cancel the closing speed along the normal (perfectly inelastic, split evenly).
					float closing = dot(other.zw - vel, normal);
					if (closing < 0.0) {
						vel_fix += normal * (closing * 0.5);
					}
				}
			}
		}

		pos += push;
		vel += vel_fix;
		confine(pos, vel);
	}

	state_out[i] = vec4(pos, vel);

	uint base = i * 12u;
	instances[base + 0u] = 1.0;
	instances[base + 1u] = 0.0;
	instances[base + 2u] = 0.0;
	instances[base + 3u] = pos.x;
	instances[base + 4u] = 0.0;
	instances[base + 5u] = 1.0;
	instances[base + 6u] = 0.0;
	instances[base + 7u] = pos.y;
	vec4 color = palette[species[i]];
	instances[base + 8u] = color.r;
	instances[base + 9u] = color.g;
	instances[base + 10u] = color.b;
	instances[base + 11u] = color.a;
}

#endif
