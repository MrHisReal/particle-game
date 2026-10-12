class_name ParticleCompute
extends RefCounted

## Runs the particle force simulation on the GPU.
##
## Particle state lives in GPU buffers, and every step writes straight into the
## MultiMesh used for drawing, so nothing is copied back to the CPU per frame.
## Neighbor lookups use a uniform grid built by four passes:
##   count       - assign each particle to a cell and count particles per cell
##   scan_blocks - prefix-sum the counts within blocks of cells
##   scan_totals - prefix-sum the block totals into each block's offset
##   reorder     - copy the particles into cell order, so neighbors are contiguous in memory
## Each step builds a grid with cells 1 / MAX_CELL_REACH of the force reach (interaction
## radius x cutoff) across and runs:
##   forces  - sum color forces from the neighboring cells within the reach, damp velocity
##             relative to those neighbors, apply drag and surface friction, and integrate
## then builds a grid with cell size = contact distance and, collision_iterations times, runs:
##   collide - push overlapping particles apart and apply contact friction
## The collision grid isn't rebuilt between iterations, so pairs that drift into contact from
## beyond the neighboring cells during them are only caught on the next step.
## The last pass of a step also writes the MultiMesh.

const SHADER_FILE: RDShaderFile = preload("res://shaders/particle_sim.glsl")
const PASSES: Array[StringName] = [
	&"spawn", &"count", &"scan_blocks", &"scan_totals", &"reorder", &"forces", &"collide",
]
const MAX_SPECIES := 16
const WORKGROUP_SIZE := 256
## Grid cells prefix-summed per scan_blocks workgroup. Must match SCAN_BLOCK in particle_sim.glsl.
const SCAN_BLOCK := 1024
## Most cells the neighbor search reaches out on each side, which is also how many cells
## the force reach is split into. Must match MAX_CELL_REACH in particle_sim.glsl.
const MAX_CELL_REACH := 2
## Byte size of the Params push constant block in particle_sim.glsl.
const PUSH_CONSTANT_SIZE := 96
## Bits of the flags push constant. Must match FLAG_* in particle_sim.glsl.
const FLAG_WRITE_INSTANCES := 1
const FLAG_WRAP := 2

var plane_size: Vector2 = Vector2.ONE
var interaction_radius: float = 80.0
var repulsion_zone: float = 0.3
## Fraction of interaction_radius where color forces end. Values below repulsion_zone leave
## only the close-range repulsion.
var force_cutoff: float = 1.0
var force_strength: float = 400.0
var linear_drag: float = 3.0
## How fast a particle's velocity relaxes toward its neighbors' (1 / second per neighbor,
## weighted by closeness).
var relative_damping: float = 0.0
var quadratic_drag: float = 0.0
var surface_friction: float = 0.0
var contact_friction: float = 0.0
var max_speed: float = 300.0
var edge_margin: float = 0.0
var species_count: int = 1
## Closest two particle centers may get (one particle diameter). 0 disables collisions.
var contact_distance: float = 0.0
var collision_iterations: int = 4
## Toroidal plane: particles leaving one edge re-enter on the opposite one, and forces and
## collisions act across the edges.
var wrap_edges: bool = false

var particle_count: int = 0

var _rd: RenderingDevice
var _shaders: Dictionary[StringName, RID] = {}
var _pipelines: Dictionary[StringName, RID] = {}
var _uniform_sets: Dictionary[String, RID] = {}

var _rules_buffer: RID
var _palette_buffer: RID
## [0] holds the particles between steps; [1] receives them in cell order during a grid build.
var _state_buffers: Array[RID] = []
## Species are reordered along with the state, so they ping-pong between two buffers.
var _species_buffers: Array[RID] = []
## Which species buffer lines up with _state_buffers[0]. Flips on every grid build.
var _species_index: int = 0
var _particle_cell: RID
var _particle_slot: RID
var _cell_count: RID
var _cell_start: RID
var _block_offset: RID
var _instance_buffer: RID
## Number of cells the cell buffers can hold; shared by the force and collision grids.
var _cell_capacity: int = 0
## Dispatches recorded in the current compute list, so barriers go only between them.
var _list_dispatches: int = 0


## Compiles the compute passes. Returns false if the renderer has no RenderingDevice.
func setup() -> bool:
	_rd = RenderingServer.get_rendering_device()
	if _rd == null:
		push_error("GPU particle simulation requires the Forward+ or Mobile renderer.")
		return false

	for pass_name in PASSES:
		var spirv: RDShaderSPIRV = SHADER_FILE.get_spirv(pass_name)
		if not spirv.compile_error_compute.is_empty():
			push_error("Particle shader pass '%s' failed to compile:\n%s" % [pass_name, spirv.compile_error_compute])
			return false
		var shader: RID = _rd.shader_create_from_spirv(spirv, "Particle %s" % pass_name)
		_shaders[pass_name] = shader
		_pipelines[pass_name] = _rd.compute_pipeline_create(shader)

	_rules_buffer = _rd.storage_buffer_create(MAX_SPECIES * MAX_SPECIES * 4)
	_palette_buffer = _rd.storage_buffer_create(MAX_SPECIES * 16)
	return true


## Uploads the rule table, flattened row-major with species_count entries per row.
func set_rules(rules: PackedFloat32Array) -> void:
	var bytes := rules.to_byte_array()
	_rd.buffer_update(_rules_buffer, 0, mini(bytes.size(), MAX_SPECIES * MAX_SPECIES * 4), bytes)


func set_palette(colors: PackedColorArray) -> void:
	var floats := PackedFloat32Array()
	for col in colors.slice(0, MAX_SPECIES):
		floats.append_array([col.r, col.g, col.b, col.a])
	if not floats.is_empty():
		_rd.buffer_update(_palette_buffer, 0, floats.size() * 4, floats.to_byte_array())


## Replaces all particles. states holds 4 floats per particle (position.xy, velocity.xy).
## instance_buffer is the MultiMesh storage buffer the simulation writes into.
func upload(states: PackedFloat32Array, species: PackedInt32Array, instance_buffer: RID) -> void:
	_allocate(species.size(), states.to_byte_array(), species.to_byte_array(), instance_buffer)


## Replaces all particles with count new ones generated on the GPU: at rest, spread uniformly
## at least margin from the plane edges, with species picked uniformly from [0, species_choices).
func spawn(count: int, species_choices: int, margin: float, rng_seed: int, instance_buffer: RID) -> void:
	_allocate(count, PackedByteArray(), PackedByteArray(), instance_buffer)
	if count == 0:
		return
	var push: PackedByteArray = _push_constant(0.0, Vector2i.ONE, 1, false)
	push.encode_u32(12, species_choices)
	push.encode_float(44, margin)
	push.encode_u32(72, rng_seed)
	var spawn_set: RID = _make_set(&"spawn", [_state_buffers[0], _species_buffers[0]])
	var list: int = _begin_list()
	_dispatch(list, &"spawn", spawn_set, push, _particle_groups())
	_rd.compute_list_end()
	_rd.free_rid(spawn_set)


## Reads the current particle states back from the GPU (blocks until the GPU catches up).
func read_states() -> PackedFloat32Array:
	if particle_count == 0:
		return PackedFloat32Array()
	return _rd.buffer_get_data(_state_buffers[0]).to_float32_array()


## Reads each particle's species back from the GPU, in the same order as read_states().
func read_species() -> PackedInt32Array:
	if particle_count == 0:
		return PackedInt32Array()
	return _rd.buffer_get_data(_species_buffers[_species_index]).to_int32_array()


## Advances the simulation by delta seconds. A delta of 0 only resolves overlaps and
## refreshes the MultiMesh.
func step(delta: float) -> void:
	if particle_count == 0:
		return

	var collisions: bool = contact_distance > 0.0
	var reach: float = interaction_radius * _cutoff()
	var force_cell: float = maxf(maxf(reach / MAX_CELL_REACH, contact_distance), 1.0)
	var force_grid: Vector2i = _grid_for(force_cell)
	var collide_grid: Vector2i = _grid_for(contact_distance) if collisions else Vector2i.ONE
	_ensure_cell_capacity(maxi(force_grid.x * force_grid.y, collide_grid.x * collide_grid.y))

	var iterations: int = maxi(collision_iterations, 1) if collisions else 0
	var groups: int = _particle_groups()
	var list: int = _begin_list()

	var force_push: PackedByteArray = _push_constant(delta, force_grid, _cells_spanned(reach, force_grid), iterations == 0)
	_build_grid(list, force_push, force_grid, groups)
	_dispatch(list, &"forces", _uniform_sets["forces%d" % _species_index], force_push, groups)

	if iterations > 0:
		var collide_push: PackedByteArray = _push_constant(delta, collide_grid, 1, false)
		_build_grid(list, collide_push, collide_grid, groups)
		for iteration in range(iterations):
			if iteration == iterations - 1:
				collide_push = _push_constant(delta, collide_grid, 1, true)
			# Iterations ping-pong between the two state buffers, keeping the sorted order.
			var direction: String = "collide" if iteration % 2 == 0 else "collide_back"
			_dispatch(list, &"collide", _uniform_sets["%s%d" % [direction, _species_index]], collide_push, groups)

	_rd.compute_list_end()
	# After an even number of iterations the particles end up in the sorted buffer.
	if iterations > 0 and iterations % 2 == 0:
		_rd.buffer_copy(_state_buffers[1], _state_buffers[0], 0, 0, particle_count * 16)


func free_resources() -> void:
	if _rd == null:
		return
	_free_particle_buffers()
	for rid in [_rules_buffer, _palette_buffer]:
		if rid.is_valid():
			_rd.free_rid(rid)
	for pass_name in _pipelines:
		_rd.free_rid(_pipelines[pass_name])
	for pass_name in _shaders:
		_rd.free_rid(_shaders[pass_name])
	_pipelines.clear()
	_shaders.clear()
	_rd = null


## Recreates the per-particle buffers for count particles, optionally filled with data.
func _allocate(count: int, state_bytes: PackedByteArray, species_bytes: PackedByteArray, instance_buffer: RID) -> void:
	_free_particle_buffers()
	if count == 0:
		return
	particle_count = count
	_species_index = 0
	_state_buffers = [_rd.storage_buffer_create(count * 16, state_bytes), _rd.storage_buffer_create(count * 16)]
	_species_buffers = [_rd.storage_buffer_create(count * 4, species_bytes), _rd.storage_buffer_create(count * 4)]
	_particle_cell = _rd.storage_buffer_create(count * 4)
	_particle_slot = _rd.storage_buffer_create(count * 4)
	_instance_buffer = instance_buffer


func _particle_groups() -> int:
	return ceili(particle_count / float(WORKGROUP_SIZE))


func _begin_list() -> int:
	_list_dispatches = 0
	return _rd.compute_list_begin()


func _dispatch(list: int, pass_name: StringName, uniform_set: RID, push: PackedByteArray, groups: int) -> void:
	if _list_dispatches > 0:
		_rd.compute_list_add_barrier(list)
	_list_dispatches += 1
	_rd.compute_list_bind_compute_pipeline(list, _pipelines[pass_name])
	_rd.compute_list_bind_uniform_set(list, uniform_set, 0)
	_rd.compute_list_set_push_constant(list, push, push.size())
	_rd.compute_list_dispatch(list, groups, 1, 1)


## Buckets the particles into the given grid and leaves them in cell order in
## _state_buffers[1], ready for a forces or collide pass to write back to _state_buffers[0].
func _build_grid(list: int, push: PackedByteArray, grid: Vector2i, groups: int) -> void:
	var scan_groups: int = (grid.x * grid.y) / SCAN_BLOCK + 1
	_dispatch(list, &"count", _uniform_sets["count"], push, groups)
	_dispatch(list, &"scan_blocks", _uniform_sets["scan_blocks"], push, scan_groups)
	_dispatch(list, &"scan_totals", _uniform_sets["scan_totals"], push, 1)
	_dispatch(list, &"reorder", _uniform_sets["reorder%d" % _species_index], push, groups)
	_species_index = 1 - _species_index


## Grid whose cells exactly tile the plane and are each at least min_cell_size across, so the
## 3x3 cells around a particle cover everything within min_cell_size, even across a wrapped edge.
func _grid_for(min_cell_size: float) -> Vector2i:
	return Vector2i(maxi(1, floori(plane_size.x / min_cell_size)), maxi(1, floori(plane_size.y / min_cell_size)))


## How many cells of grid the neighbor search must reach out on each side to cover distance.
## The slack keeps rounding from adding a whole ring of cells when distance fits exactly.
func _cells_spanned(distance: float, grid: Vector2i) -> int:
	var cell: float = minf(plane_size.x / grid.x, plane_size.y / grid.y)
	return clampi(ceili(distance / cell - 1e-6), 1, MAX_CELL_REACH)


func _push_constant(delta: float, grid: Vector2i, cell_reach: int, write_instances: bool) -> PackedByteArray:
	var flags: int = 0
	if write_instances:
		flags |= FLAG_WRITE_INSTANCES
	if wrap_edges:
		flags |= FLAG_WRAP

	var push := PackedByteArray()
	push.resize(PUSH_CONSTANT_SIZE)
	push.fill(0)
	push.encode_u32(0, particle_count)
	push.encode_u32(4, grid.x)
	push.encode_u32(8, grid.y)
	push.encode_u32(12, species_count)
	push.encode_u32(16, flags)
	push.encode_float(20, interaction_radius)
	push.encode_float(24, repulsion_zone)
	push.encode_float(28, force_strength)
	push.encode_float(32, exp(-linear_drag * delta))
	push.encode_float(36, max_speed)
	push.encode_float(40, delta)
	push.encode_float(44, edge_margin)
	push.encode_float(48, plane_size.x)
	push.encode_float(52, plane_size.y)
	push.encode_float(56, contact_distance)
	push.encode_float(60, quadratic_drag)
	push.encode_float(64, surface_friction)
	push.encode_float(68, contact_friction)
	push.encode_float(76, _cutoff())
	push.encode_u32(80, cell_reach)
	push.encode_float(84, relative_damping)
	return push


func _cutoff() -> float:
	return clampf(force_cutoff, repulsion_zone, 1.0)


## Grows the per-cell buffers when a grid needs more cells, then rebuilds uniform sets.
func _ensure_cell_capacity(cells: int) -> void:
	if cells <= _cell_capacity and not _uniform_sets.is_empty():
		return
	_free_grid_buffers()
	_cell_capacity = cells
	_cell_count = _rd.storage_buffer_create(_cell_capacity * 4)
	# The scan re-zeroes every count it reads, so this is the only clear needed.
	_rd.buffer_clear(_cell_count, 0, _cell_capacity * 4)
	_cell_start = _rd.storage_buffer_create((_cell_capacity + 1) * 4)
	_block_offset = _rd.storage_buffer_create((_cell_capacity / SCAN_BLOCK + 1) * 4)
	_build_uniform_sets()


func _build_uniform_sets() -> void:
	_free_uniform_sets()
	var home: RID = _state_buffers[0]
	var sorted: RID = _state_buffers[1]
	_uniform_sets["count"] = _make_set(&"count", [home, _cell_count, _particle_cell, _particle_slot])
	_uniform_sets["scan_blocks"] = _make_set(&"scan_blocks", [_cell_count, _cell_start, _block_offset])
	_uniform_sets["scan_totals"] = _make_set(&"scan_totals", [_block_offset])
	for src in range(2):
		var species_in: RID = _species_buffers[src]
		var species_out: RID = _species_buffers[1 - src]
		_uniform_sets["reorder%d" % src] = _make_set(&"reorder", [
			home, species_in, _particle_cell, _particle_slot, _cell_start, _block_offset,
			sorted, species_out,
		])
		# Keyed by the species buffer that lines up with the sorted state.
		var collide_buffers: Array[RID] = [
			sorted, home, species_in, _cell_start, _block_offset, _palette_buffer, _instance_buffer,
		]
		var force_buffers: Array[RID] = collide_buffers.duplicate()
		force_buffers.append(_rules_buffer)
		var collide_back_buffers: Array[RID] = collide_buffers.duplicate()
		collide_back_buffers[0] = home
		collide_back_buffers[1] = sorted
		_uniform_sets["collide%d" % src] = _make_set(&"collide", collide_buffers)
		_uniform_sets["collide_back%d" % src] = _make_set(&"collide", collide_back_buffers)
		_uniform_sets["forces%d" % src] = _make_set(&"forces", force_buffers)


## Binds buffers to consecutive bindings (0, 1, 2, ...) of set 0 for the given pass.
func _make_set(pass_name: StringName, buffers: Array[RID]) -> RID:
	var uniforms: Array[RDUniform] = []
	for binding in range(buffers.size()):
		var uniform := RDUniform.new()
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		uniform.binding = binding
		uniform.add_id(buffers[binding])
		uniforms.append(uniform)
	return _rd.uniform_set_create(uniforms, _shaders[pass_name], 0)


## Uniform sets are also freed automatically when a buffer they use is freed
## (including the MultiMesh buffer when the MultiMesh is resized), so check validity first.
func _free_uniform_sets() -> void:
	for key in _uniform_sets:
		var rid: RID = _uniform_sets[key]
		if _rd.uniform_set_is_valid(rid):
			_rd.free_rid(rid)
	_uniform_sets.clear()


func _free_grid_buffers() -> void:
	_free_uniform_sets()
	for rid in [_cell_count, _cell_start, _block_offset]:
		if rid.is_valid():
			_rd.free_rid(rid)
	_cell_count = RID()
	_cell_start = RID()
	_block_offset = RID()
	_cell_capacity = 0


func _free_particle_buffers() -> void:
	_free_grid_buffers()
	for rid in _state_buffers + _species_buffers + [_particle_cell, _particle_slot]:
		if rid.is_valid():
			_rd.free_rid(rid)
	_state_buffers.clear()
	_species_buffers.clear()
	_particle_cell = RID()
	_particle_slot = RID()
	_instance_buffer = RID()
	particle_count = 0
