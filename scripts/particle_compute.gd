class_name ParticleCompute
extends RefCounted

## Runs the particle force simulation on the GPU.
##
## Particle state lives in GPU buffers, and every step writes straight into the
## MultiMesh used for drawing, so nothing is copied back to the CPU per frame.
## Neighbor lookups use a uniform grid built by three passes:
##   count   - assign each particle to a cell and count particles per cell
##   scan    - prefix-sum the counts into each cell's start offset
##   scatter - write particle indices sorted by cell
## Each step builds a grid with cell size = interaction radius and runs:
##   forces  - sum color forces from the 3x3 neighboring cells and integrate
## then, collision_iterations times, rebuilds a grid with cell size = contact distance and runs:
##   collide - push overlapping particles apart and write the MultiMesh

const SHADER_FILE: RDShaderFile = preload("res://shaders/particle_sim.glsl")
const PASSES: Array[StringName] = [&"count", &"scan", &"scatter", &"forces", &"collide"]
const MAX_SPECIES := 16
const WORKGROUP_SIZE := 256
## Byte size of the Params push constant block in particle_sim.glsl.
const PUSH_CONSTANT_SIZE := 64

var plane_size: Vector2 = Vector2.ONE
var interaction_radius: float = 80.0
var repulsion_zone: float = 0.3
var force_strength: float = 400.0
var friction: float = 3.0
var max_speed: float = 300.0
var edge_margin: float = 0.0
var species_count: int = 1
## Closest two particle centers may get (one particle diameter). 0 disables collisions.
var contact_distance: float = 0.0
var collision_iterations: int = 4

var particle_count: int = 0

var _rd: RenderingDevice
var _shaders: Dictionary[StringName, RID] = {}
var _pipelines: Dictionary[StringName, RID] = {}
var _uniform_sets: Dictionary[String, RID] = {}

var _rules_buffer: RID
var _palette_buffer: RID
var _state_buffers: Array[RID] = []
var _species_buffer: RID
var _particle_cell: RID
var _particle_slot: RID
var _sorted_index: RID
var _cell_count: RID
var _cell_start: RID
var _instance_buffer: RID
## Number of cells the cell buffers can hold; shared by the force and collision grids.
var _cell_capacity: int = 0
## Which of the two ping-pong state buffers holds the latest positions.
var _read_index: int = 0


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
## instance_buffer is the MultiMesh storage buffer the forces pass writes into.
func upload(states: PackedFloat32Array, species: PackedInt32Array, instance_buffer: RID) -> void:
	_free_particle_buffers()
	particle_count = species.size()
	_read_index = 0
	if particle_count == 0:
		return

	var state_bytes := states.to_byte_array()
	_state_buffers = [
		_rd.storage_buffer_create(state_bytes.size(), state_bytes),
		_rd.storage_buffer_create(state_bytes.size()),
	]
	_species_buffer = _rd.storage_buffer_create(particle_count * 4, species.to_byte_array())
	_particle_cell = _rd.storage_buffer_create(particle_count * 4)
	_particle_slot = _rd.storage_buffer_create(particle_count * 4)
	_sorted_index = _rd.storage_buffer_create(particle_count * 4)
	_instance_buffer = instance_buffer


## Reads the current particle states back from the GPU (blocks until the GPU catches up).
func read_states() -> PackedFloat32Array:
	if particle_count == 0:
		return PackedFloat32Array()
	return _rd.buffer_get_data(_state_buffers[_read_index]).to_float32_array()


## Advances the simulation by delta seconds. A delta of 0 only refreshes the MultiMesh.
func step(delta: float) -> void:
	if particle_count == 0:
		return

	var collisions: bool = contact_distance > 0.0
	var force_cell: float = maxf(maxf(interaction_radius, contact_distance), 1.0)
	var force_grid: Vector2i = _grid_for(force_cell)
	var collide_grid: Vector2i = _grid_for(contact_distance) if collisions else Vector2i.ONE
	_ensure_cell_capacity(maxi(force_grid.x * force_grid.y, collide_grid.x * collide_grid.y))

	var force_push: PackedByteArray = _push_constant(delta, force_grid, force_cell)
	var collide_push: PackedByteArray = _push_constant(delta, collide_grid, contact_distance)
	var groups: int = ceili(particle_count / float(WORKGROUP_SIZE))

	var src: int = _read_index
	_build_grid(src, force_push, force_grid, groups)
	_dispatch(&"forces", _uniform_sets["forces%d" % src], force_push, groups)
	src = 1 - src

	# Without collisions a single collide pass still runs to write the MultiMesh.
	var iterations: int = maxi(collision_iterations, 1) if collisions else 1
	for iteration in range(iterations):
		if collisions:
			_build_grid(src, collide_push, collide_grid, groups)
		_dispatch(&"collide", _uniform_sets["collide%d" % src], collide_push, groups)
		src = 1 - src
	_read_index = src


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


func _dispatch(pass_name: StringName, uniform_set: RID, push: PackedByteArray, groups: int) -> void:
	var list: int = _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(list, _pipelines[pass_name])
	_rd.compute_list_bind_uniform_set(list, uniform_set, 0)
	_rd.compute_list_set_push_constant(list, push, push.size())
	_rd.compute_list_dispatch(list, groups, 1, 1)
	_rd.compute_list_end()


## Buckets the particles in state buffer src into the given grid.
func _build_grid(src: int, push: PackedByteArray, grid: Vector2i, groups: int) -> void:
	_rd.buffer_clear(_cell_count, 0, grid.x * grid.y * 4)
	_dispatch(&"count", _uniform_sets["count%d" % src], push, groups)
	_dispatch(&"scan", _uniform_sets["scan"], push, 1)
	_dispatch(&"scatter", _uniform_sets["scatter"], push, groups)


func _grid_for(cell_size: float) -> Vector2i:
	return Vector2i(maxi(1, ceili(plane_size.x / cell_size)), maxi(1, ceili(plane_size.y / cell_size)))


func _push_constant(delta: float, grid: Vector2i, cell_size: float) -> PackedByteArray:
	var push := PackedByteArray()
	push.resize(PUSH_CONSTANT_SIZE)
	push.fill(0)
	push.encode_u32(0, particle_count)
	push.encode_u32(4, grid.x)
	push.encode_u32(8, grid.y)
	push.encode_u32(12, species_count)
	push.encode_float(16, cell_size)
	push.encode_float(20, interaction_radius)
	push.encode_float(24, repulsion_zone)
	push.encode_float(28, force_strength)
	push.encode_float(32, exp(-friction * delta))
	push.encode_float(36, max_speed)
	push.encode_float(40, delta)
	push.encode_float(44, edge_margin)
	push.encode_float(48, plane_size.x)
	push.encode_float(52, plane_size.y)
	push.encode_float(56, contact_distance)
	return push


## Grows the per-cell buffers when a grid needs more cells, then rebuilds uniform sets.
func _ensure_cell_capacity(cells: int) -> void:
	if cells <= _cell_capacity and not _uniform_sets.is_empty():
		return
	_free_grid_buffers()
	_cell_capacity = cells
	_cell_count = _rd.storage_buffer_create(cells * 4)
	_cell_start = _rd.storage_buffer_create((cells + 1) * 4)
	_build_uniform_sets()


func _build_uniform_sets() -> void:
	_free_uniform_sets()
	for src in range(2):
		var state_in: RID = _state_buffers[src]
		var state_out: RID = _state_buffers[1 - src]
		_uniform_sets["count%d" % src] = _make_set(&"count", [state_in, _cell_count, _particle_cell, _particle_slot])
		_uniform_sets["forces%d" % src] = _make_set(&"forces", [
			state_in, state_out, _species_buffer, _rules_buffer, _cell_start, _sorted_index,
		])
		_uniform_sets["collide%d" % src] = _make_set(&"collide", [
			state_in, state_out, _species_buffer, _cell_start, _sorted_index,
			_palette_buffer, _instance_buffer,
		])
	_uniform_sets["scan"] = _make_set(&"scan", [_cell_count, _cell_start])
	_uniform_sets["scatter"] = _make_set(&"scatter", [_particle_cell, _particle_slot, _cell_start, _sorted_index])


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
	for rid in [_cell_count, _cell_start]:
		if rid.is_valid():
			_rd.free_rid(rid)
	_cell_count = RID()
	_cell_start = RID()
	_cell_capacity = 0


func _free_particle_buffers() -> void:
	_free_grid_buffers()
	for rid in _state_buffers + [_species_buffer, _particle_cell, _particle_slot, _sorted_index]:
		if rid.is_valid():
			_rd.free_rid(rid)
	_state_buffers.clear()
	_species_buffer = RID()
	_particle_cell = RID()
	_particle_slot = RID()
	_sorted_index = RID()
	_instance_buffer = RID()
	particle_count = 0
