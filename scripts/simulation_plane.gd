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

## When true, each particle is assigned a random species (color) from species_colors.
@export var random_colors: bool = true

## Base particle color when random_colors is false (all particles use species 0's rules).
@export var default_particle_color: Color = Color(0.35, 0.82, 1.0, 0.95)

@export_group("Species")
## Display color for each species. A particle's species decides how it reacts to others.
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

## Maximum distance at which particles influence each other.
@export var interaction_radius: float = 80.0

## Fraction of interaction_radius inside which particles always push apart, regardless of color.
@export_range(0.05, 0.9) var repulsion_zone: float = 0.3

## Scales all color-based forces (pixels / second²).
@export var force_strength: float = 400.0

## Velocity damping per second. Higher values make motion more sluggish.
@export var friction: float = 3.0

## Upper bound on particle speed (pixels / second).
@export var max_speed: float = 300.0

## Array storing all currently active particles on the plane.
var particles: Array[Particle] = []

## attraction_rules[a][b] is how species a reacts to species b:
## positive values attract, negative values repel, in the range [-1, 1].
var attraction_rules: Array[PackedFloat32Array] = []


func _ready() -> void:
	# Center the plane in the viewport and populate particles
	center_in_viewport()
	get_viewport().size_changed.connect(_on_viewport_size_changed)
	set_default_rules()
	populate_particles(initial_particle_count)


func _physics_process(delta: float) -> void:
	if not simulation_running or particles.is_empty():
		return
	_apply_forces(delta)
	_integrate(delta)
	queue_redraw()


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

		var species: int = _pick_species()
		var p := Particle.new(pos, _species_color(species), particle_radius, species)
		particles.append(p)

	particles_updated.emit(particles.size())
	queue_redraw()


## Adds an individual particle at the given plane-local coordinate.
## A negative species_override picks a random species.
func add_particle(local_pos: Vector2, species_override: int = -1, radius_override: float = -1.0) -> void:
	var r: float = particle_radius if radius_override <= 0.0 else radius_override
	var species: int = _pick_species() if species_override < 0 else species_override

	# Clamp to plane bounds
	var margin: float = r + border_width
	var clamped_pos := Vector2(
		clampf(local_pos.x, margin, maxf(margin, plane_size.x - margin)),
		clampf(local_pos.y, margin, maxf(margin, plane_size.y - margin))
	)

	var p := Particle.new(clamped_pos, _species_color(species), r, species)
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


## Default rules: each color is drawn to one partner color, pushed away by the others,
## and mildly attracted to its own kind. The chase cycle is Red -> Blue -> Green -> Yellow -> Red.
func set_default_rules() -> void:
	if species_colors.size() != 4:
		randomize_rules()
		return
	attraction_rules = [
		#                     Red   Green  Blue  Yellow
		PackedFloat32Array([ 0.3, -0.5,   0.8, -0.4]), # Red
		PackedFloat32Array([-0.4,  0.3,  -0.5,  0.8]), # Green
		PackedFloat32Array([-0.5,  0.8,   0.3, -0.4]), # Blue
		PackedFloat32Array([ 0.8, -0.4,  -0.5,  0.3]), # Yellow
	]
	print_rules()


## Replaces the rule table with random attraction/repulsion values in [-1, 1].
func randomize_rules() -> void:
	var n: int = species_colors.size()
	attraction_rules.clear()
	for a in range(n):
		var row := PackedFloat32Array()
		row.resize(n)
		for b in range(n):
			row[b] = snappedf(randf_range(-1.0, 1.0), 0.1)
		attraction_rules.append(row)
	print_rules()


## Prints the current rule table to the output console.
func print_rules() -> void:
	print("Particle rules (row reacts to column, + attracts, - repels):")
	for a in range(attraction_rules.size()):
		var parts := PackedStringArray()
		for b in range(attraction_rules[a].size()):
			parts.append("%s %+.1f" % [_species_name(b), attraction_rules[a][b]])
		print("  %s -> %s" % [_species_name(a), ", ".join(parts)])


## Signed force for a pair at normalized distance r (0..1 of interaction_radius).
## Close range always repels so particles don't collapse; beyond that, the
## attraction value peaks halfway through the remaining range and fades to zero.
func _force_profile(r: float, attraction: float) -> float:
	if r < repulsion_zone:
		return r / repulsion_zone - 1.0
	if r < 1.0:
		return attraction * (1.0 - absf(2.0 * r - 1.0 - repulsion_zone) / (1.0 - repulsion_zone))
	return 0.0


## Accumulates color-based forces into each particle's velocity using a uniform grid
## so each particle only checks neighbors in the surrounding cells.
func _apply_forces(delta: float) -> void:
	var count: int = particles.size()
	var cell_size: float = maxf(interaction_radius, 1.0)
	var cols: int = maxi(1, ceili(plane_size.x / cell_size))
	var rows: int = maxi(1, ceili(plane_size.y / cell_size))
	var cell_total: int = cols * rows

	# Bucket particle indices by cell (counting sort into a flat array).
	var cell_of := PackedInt32Array()
	cell_of.resize(count)
	var cell_start := PackedInt32Array()
	cell_start.resize(cell_total + 1)
	cell_start.fill(0)
	for i in range(count):
		var pos: Vector2 = particles[i].position
		var cx: int = clampi(int(pos.x / cell_size), 0, cols - 1)
		var cy: int = clampi(int(pos.y / cell_size), 0, rows - 1)
		var c: int = cx + cy * cols
		cell_of[i] = c
		cell_start[c + 1] += 1
	for c in range(cell_total):
		cell_start[c + 1] += cell_start[c]
	var fill_pos: PackedInt32Array = cell_start.duplicate()
	var sorted := PackedInt32Array()
	sorted.resize(count)
	for i in range(count):
		var c: int = cell_of[i]
		sorted[fill_pos[c]] = i
		fill_pos[c] += 1

	var r_max: float = interaction_radius
	var r_max_sq: float = r_max * r_max
	var species_count: int = attraction_rules.size()

	for i in range(count):
		var p: Particle = particles[i]
		if p.species >= species_count:
			continue
		var rules: PackedFloat32Array = attraction_rules[p.species]
		var cx: int = cell_of[i] % cols
		@warning_ignore("integer_division")
		var cy: int = cell_of[i] / cols
		var acc := Vector2.ZERO

		for ny in range(maxi(cy - 1, 0), mini(cy + 2, rows)):
			for nx in range(maxi(cx - 1, 0), mini(cx + 2, cols)):
				var c: int = nx + ny * cols
				for k in range(cell_start[c], cell_start[c + 1]):
					var j: int = sorted[k]
					if j == i:
						continue
					var q: Particle = particles[j]
					var offset: Vector2 = q.position - p.position
					var dist_sq: float = offset.length_squared()
					if dist_sq >= r_max_sq or dist_sq < 0.0001:
						continue
					var attraction: float = rules[q.species] if q.species < rules.size() else 0.0
					var dist: float = sqrt(dist_sq)
					acc += offset / dist * _force_profile(dist / r_max, attraction)

		p.velocity += acc * force_strength * delta


## Applies friction, moves particles, and bounces them off the plane's edges.
func _integrate(delta: float) -> void:
	var damping: float = exp(-friction * delta)
	for p in particles:
		p.velocity = (p.velocity * damping).limit_length(max_speed)
		p.position += p.velocity * delta

		var margin: float = p.radius + border_width
		var max_x: float = maxf(margin, plane_size.x - margin)
		var max_y: float = maxf(margin, plane_size.y - margin)
		if p.position.x < margin:
			p.position.x = margin
			p.velocity.x = absf(p.velocity.x)
		elif p.position.x > max_x:
			p.position.x = max_x
			p.velocity.x = -absf(p.velocity.x)
		if p.position.y < margin:
			p.position.y = margin
			p.velocity.y = absf(p.velocity.y)
		elif p.position.y > max_y:
			p.position.y = max_y
			p.velocity.y = -absf(p.velocity.y)


func _pick_species() -> int:
	if not random_colors or species_colors.is_empty():
		return 0
	return randi() % species_colors.size()


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
	draw_rect(plane_rect, border_color, false, border_width)

	# 4. Particles populated on the plane
	for p in particles:
		# Soft outer glow
		var glow_col := Color(p.color.r, p.color.g, p.color.b, 0.25)
		draw_circle(p.position, p.radius + 1.5, glow_col)
		# Solid particle core
		draw_circle(p.position, p.radius, p.color)
