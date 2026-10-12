class_name SimulationPlane
extends Node2D

## Represents the 2D simulation plane and manages the particles residing on it.
## Particle physics runs on the GPU (see ParticleCompute); this node owns the settings,
## the color rules, and the MultiMesh the particles are drawn with.

signal particles_updated(count: int)

const PARTICLE_SHADER: Shader = preload("res://shaders/particle_draw.gdshader")
## Width of the soft glow drawn around each particle's solid core.
const GLOW_WIDTH := 1.5
## Longest time step simulated in one frame, so a frame hitch can't launch particles.
const MAX_STEP := 1.0 / 30.0

@export_group("Plane Geometry")
## The width and height of the 2D simulation plane.
@export var plane_size: Vector2 = Vector2(960.0, 540.0):
	set(value):
		plane_size = value
		queue_redraw()

## Fill color for the 2D plane surface.
@export var plane_color: Color = Color(0.08, 0.10, 0.15, 1.0)

## Outline border color around the perimeter of the plane.
@export var border_color: Color = Color(0.3, 0.55, 0.9, 1.0)

## Thickness of the border line.
@export var border_width: float = 2.5

@export_group("Grid")
## Whether to render coordinate grid lines on the plane surface.
@export var show_grid: bool = true:
	set(value):
		show_grid = value
		queue_redraw()

## Distance in pixels between consecutive grid lines.
@export var grid_step: float = 60.0

## Color for the background grid lines.
@export var grid_color: Color = Color(0.15, 0.19, 0.27, 0.75)

@export_group("Particles")
## Initial number of particles to scatter across the plane.
@export var initial_particle_count: int = 300

## Radius of every particle.
@export var particle_radius: float = 4.5

## When true, each particle is assigned a random species (color) from species_colors.
@export var random_colors: bool = true

## Base particle color when random_colors is false (all particles use species 0's rules).
@export var default_particle_color: Color = Color(0.35, 0.82, 1.0, 0.95)

@export_group("Species")
## Display color for each species (up to 16). A particle's species decides how it reacts to others.
@export var species_colors: Array[Color] = [
	Color(1.0, 0.32, 0.32, 0.95),
	Color(0.35, 0.95, 0.45, 0.95),
	Color(0.35, 0.6, 1.0, 0.95),
	Color(1.0, 0.85, 0.3, 0.95),
]

## Human-readable names matching species_colors, used when printing the rule table.
@export var species_names: PackedStringArray = ["Red", "Green", "Blue", "Yellow"]

@export_group("Forces")
## Whether the force simulation is advancing.
@export var simulation_running: bool = true

## Maximum distance at which particles influence each other. Smaller values are much
## faster at high particle counts, since each particle checks fewer neighbors.
@export var interaction_radius: float = 80.0

## Fraction of interaction_radius inside which particles always push apart, regardless of color.
@export_range(0.05, 0.9) var repulsion_zone: float = 0.3

## Scales all color-based forces (pixels / second²).
@export var force_strength: float = 400.0

## Velocity damping per second. Higher values make motion more sluggish.
@export var friction: float = 3.0

## Upper bound on particle speed (pixels / second).
@export var max_speed: float = 300.0

@export_group("Collisions")
## Treat particles as solid discs that push each other apart instead of overlapping.
@export var particle_collisions: bool = true

## Collision passes per frame. More passes keep dense clumps from overlapping, at a GPU cost.
@export_range(1, 16) var collision_iterations: int = 4

## Number of particles currently on the plane.
var particle_count: int:
	get:
		return _species.size()

## attraction_rules[a][b] is how species a reacts to species b:
## positive values attract, negative values repel, in the range [-1, 1].
var attraction_rules: Array[PackedFloat32Array] = []

var _gpu := ParticleCompute.new()
var _gpu_ready: bool = false
## Species index of each particle, kept on the CPU so particles can be re-uploaded.
var _species := PackedInt32Array()
var _multimesh := MultiMesh.new()


func _ready() -> void:
	center_in_viewport()
	get_viewport().size_changed.connect(_on_viewport_size_changed)
	_setup_particle_renderer()
	_gpu_ready = _gpu.setup()
	set_default_rules()
	populate_particles(initial_particle_count)


func _process(delta: float) -> void:
	if simulation_running and _gpu_ready and not _species.is_empty():
		_step(minf(delta, MAX_STEP))


func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE:
		_gpu.free_resources()


func _on_viewport_size_changed() -> void:
	center_in_viewport()


## Centers the 2D plane within the window/viewport, leaving room for top UI controls.
func center_in_viewport(top_offset: float = 60.0) -> void:
	var vp_size: Vector2 = get_viewport_rect().size
	var usable_h: float = maxf(plane_size.y, vp_size.y - top_offset)
	var pos_x: float = (vp_size.x - plane_size.x) * 0.5
	var pos_y: float = top_offset + (usable_h - plane_size.y) * 0.5
	position = Vector2(pos_x, pos_y).round()


## Populates the plane with particles at random positions across the entire area.
func populate_particles(count: int = -1) -> void:
	if count >= 0:
		initial_particle_count = count
	else:
		count = initial_particle_count

	var margin: float = particle_radius + border_width + 1.0
	var max_x: float = maxf(margin, plane_size.x - margin)
	var max_y: float = maxf(margin, plane_size.y - margin)

	var states := PackedFloat32Array()
	states.resize(count * 4)
	states.fill(0.0)
	var species := PackedInt32Array()
	species.resize(count)
	for i in range(count):
		states[i * 4] = randf_range(margin, max_x)
		states[i * 4 + 1] = randf_range(margin, max_y)
		species[i] = _pick_species()

	_upload(states, species)


## Adds an individual particle at the given plane-local coordinate.
## A negative species_override picks a random species.
func add_particle(local_pos: Vector2, species_override: int = -1) -> void:
	if not _gpu_ready:
		return
	var margin: float = particle_radius + border_width
	var clamped_pos := Vector2(
		clampf(local_pos.x, margin, maxf(margin, plane_size.x - margin)),
		clampf(local_pos.y, margin, maxf(margin, plane_size.y - margin))
	)

	var states: PackedFloat32Array = _gpu.read_states()
	states.append_array([clamped_pos.x, clamped_pos.y, 0.0, 0.0])
	var species: PackedInt32Array = _species.duplicate()
	species.append(_pick_species() if species_override < 0 else species_override)
	_upload(states, species)


## Clears all particles currently on the plane.
func clear_particles() -> void:
	_upload(PackedFloat32Array(), PackedInt32Array())


## Returns true if the given plane-local position lies inside the plane boundaries.
func is_point_inside(local_pos: Vector2) -> bool:
	var plane_rect := Rect2(Vector2.ZERO, plane_size)
	return plane_rect.has_point(local_pos)


## Default rules: each color is drawn to one partner color, pushed away by the others,
## and mildly attracted to its own kind. The chase cycle is Red -> Blue -> Green -> Yellow -> Red.
func set_default_rules() -> void:
	if _species_total() != 4:
		randomize_rules()
		return
	attraction_rules = [
		#                     Red   Green  Blue  Yellow
		PackedFloat32Array([ 0.3, -0.5,   0.8, -0.4]), # Red
		PackedFloat32Array([-0.4,  0.3,  -0.5,  0.8]), # Green
		PackedFloat32Array([-0.5,  0.8,   0.3, -0.4]), # Blue
		PackedFloat32Array([ 0.8, -0.4,  -0.5,  0.3]), # Yellow
	]
	_sync_rules()


## Replaces the rule table with random attraction/repulsion values in [-1, 1].
func randomize_rules() -> void:
	var n: int = _species_total()
	attraction_rules.clear()
	for a in range(n):
		var row := PackedFloat32Array()
		row.resize(n)
		for b in range(n):
			row[b] = snappedf(randf_range(-1.0, 1.0), 0.1)
		attraction_rules.append(row)
	_sync_rules()


## Prints the current rule table to the output console.
func print_rules() -> void:
	print("Particle rules (row reacts to column, + attracts, - repels):")
	for a in range(attraction_rules.size()):
		var parts := PackedStringArray()
		for b in range(attraction_rules[a].size()):
			parts.append("%s %+.1f" % [_species_name(b), attraction_rules[a][b]])
		print("  %s -> %s" % [_species_name(a), ", ".join(parts)])


func _setup_particle_renderer() -> void:
	var outer_radius: float = particle_radius + GLOW_WIDTH
	var mesh := QuadMesh.new()
	mesh.size = Vector2.ONE * outer_radius * 2.0

	var material := ShaderMaterial.new()
	material.shader = PARTICLE_SHADER
	material.set_shader_parameter("core_ratio", particle_radius / outer_radius)

	_multimesh.transform_format = MultiMesh.TRANSFORM_2D
	_multimesh.use_colors = true
	_multimesh.mesh = mesh

	var instance := MultiMeshInstance2D.new()
	instance.multimesh = _multimesh
	instance.material = material
	add_child(instance)


## Replaces every particle on the GPU and resizes the MultiMesh to match.
func _upload(states: PackedFloat32Array, species: PackedInt32Array) -> void:
	_species = species
	# Instance data is written on the GPU, so the bounds can't be computed from it.
	_multimesh.custom_aabb = AABB(Vector3(-particle_radius, -particle_radius, -1.0),
		Vector3(plane_size.x + particle_radius * 2.0, plane_size.y + particle_radius * 2.0, 2.0))
	_multimesh.instance_count = species.size()

	if _gpu_ready:
		var instance_buffer: RID = RenderingServer.multimesh_get_buffer_rd_rid(_multimesh.get_rid())
		_gpu.upload(states, species, instance_buffer)
		_sync_palette()
		_step(0.0)

	particles_updated.emit(species.size())


func _step(delta: float) -> void:
	_gpu.plane_size = plane_size
	_gpu.interaction_radius = interaction_radius
	_gpu.repulsion_zone = repulsion_zone
	_gpu.force_strength = force_strength
	_gpu.friction = friction
	_gpu.max_speed = max_speed
	_gpu.edge_margin = particle_radius + border_width
	_gpu.species_count = _species_total()
	_gpu.contact_distance = particle_radius * 2.0 if particle_collisions else 0.0
	_gpu.collision_iterations = collision_iterations
	_gpu.step(delta)


func _sync_rules() -> void:
	print_rules()
	if not _gpu_ready:
		return
	var flat := PackedFloat32Array()
	for row in attraction_rules:
		flat.append_array(row)
	_gpu.set_rules(flat)


func _sync_palette() -> void:
	var colors := PackedColorArray()
	for species in range(_species_total()):
		colors.append(_species_color(species))
	_gpu.set_palette(colors)


func _species_total() -> int:
	return clampi(species_colors.size(), 1, ParticleCompute.MAX_SPECIES)


func _pick_species() -> int:
	if not random_colors:
		return 0
	return randi() % _species_total()


func _species_color(species: int) -> Color:
	if not random_colors or species >= species_colors.size():
		return default_particle_color
	return species_colors[species]


func _species_name(species: int) -> String:
	if species < species_names.size():
		return species_names[species]
	return "Species %d" % species


func _draw() -> void:
	var plane_rect := Rect2(Vector2.ZERO, plane_size)

	# 1. 2D Plane Surface
	draw_rect(plane_rect, plane_color, true)

	# 2. Grid lines
	if show_grid and grid_step > 0.0:
		var x: float = grid_step
		while x < plane_size.x:
			draw_line(Vector2(x, 0), Vector2(x, plane_size.y), grid_color, 1.0)
			x += grid_step

		var y: float = grid_step
		while y < plane_size.y:
			draw_line(Vector2(0, y), Vector2(plane_size.x, y), grid_color, 1.0)
			y += grid_step

	# 3. Plane Border Outline
	# Particles are drawn on top by the MultiMeshInstance2D child.
	draw_rect(plane_rect, border_color, false, border_width)
