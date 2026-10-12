class_name SimulationPlane
extends Node2D

## Represents the 2D simulation plane and manages the particles residing on it.

signal particles_updated(count: int)

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

## Default radius for spawned particles.
@export var particle_radius: float = 4.5

## When true, each particle receives a vibrant random hue.
@export var random_colors: bool = true

## Base particle color when random_colors is false.
@export var default_particle_color: Color = Color(0.35, 0.82, 1.0, 0.95)

## Array storing all currently active particles on the plane.
var particles: Array[Particle] = []


func _ready() -> void:
	# Center the plane in the viewport and populate particles
	center_in_viewport()
	get_viewport().size_changed.connect(_on_viewport_size_changed)
	populate_particles(initial_particle_count)


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

	particles.clear()

	var margin: float = particle_radius + border_width + 1.0
	var min_x: float = margin
	var max_x: float = maxf(margin, plane_size.x - margin)
	var min_y: float = margin
	var max_y: float = maxf(margin, plane_size.y - margin)

	for i in range(count):
		var rand_x: float = randf_range(min_x, max_x)
		var rand_y: float = randf_range(min_y, max_y)
		var pos := Vector2(rand_x, rand_y)

		var col: Color = _generate_particle_color()
		var p := Particle.new(pos, col, particle_radius)
		particles.append(p)

	particles_updated.emit(particles.size())
	queue_redraw()


## Adds an individual particle at the given plane-local coordinate.
func add_particle(local_pos: Vector2, color_override: Color = Color(0, 0, 0, 0), radius_override: float = -1.0) -> void:
	var r: float = particle_radius if radius_override <= 0.0 else radius_override
	var col: Color = _generate_particle_color() if color_override.a <= 0.0 else color_override

	# Clamp to plane bounds
	var margin: float = r + border_width
	var clamped_pos := Vector2(
		clampf(local_pos.x, margin, maxf(margin, plane_size.x - margin)),
		clampf(local_pos.y, margin, maxf(margin, plane_size.y - margin))
	)

	var p := Particle.new(clamped_pos, col, r)
	particles.append(p)
	particles_updated.emit(particles.size())
	queue_redraw()


## Clears all particles currently on the plane.
func clear_particles() -> void:
	particles.clear()
	particles_updated.emit(0)
	queue_redraw()


## Returns true if the given plane-local position lies inside the plane boundaries.
func is_point_inside(local_pos: Vector2) -> bool:
	var plane_rect := Rect2(Vector2.ZERO, plane_size)
	return plane_rect.has_point(local_pos)


## Helper to generate an aesthetically vibrant particle color.
func _generate_particle_color() -> Color:
	if not random_colors:
		return default_particle_color
	var hue: float = randf()
	var saturation: float = randf_range(0.65, 0.95)
	var value: float = randf_range(0.85, 1.0)
	return Color.from_hsv(hue, saturation, value, 0.95)


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
	draw_rect(plane_rect, border_color, false, border_width)

	# 4. Particles populated on the plane
	for p in particles:
		# Soft outer glow
		var glow_col := Color(p.color.r, p.color.g, p.color.b, 0.25)
		draw_circle(p.position, p.radius + 1.5, glow_col)
		# Solid particle core
		draw_circle(p.position, p.radius, p.color)
