class_name Main
extends Node2D

@onready var simulation_plane: SimulationPlane = $SimulationPlane

# UI Node References
@onready var info_label: Label = %InfoLabel
@onready var count_slider: HSlider = %CountSlider
@onready var count_display: Label = %CountDisplay
@onready var repopulate_btn: Button = %RepopulateBtn
@onready var toggle_grid_btn: Button = %ToggleGridBtn
@onready var color_mode_btn: Button = %ColorModeBtn
@onready var clear_btn: Button = %ClearBtn
@onready var pause_btn: Button = %PauseBtn
@onready var rules_btn: Button = %RulesBtn
@onready var default_rules_btn: Button = %DefaultRulesBtn


func _ready() -> void:
	# Connect plane signals
	simulation_plane.particles_updated.connect(_on_particles_updated)

	# Setup UI initial values
	count_slider.value = simulation_plane.initial_particle_count
	count_display.text = str(int(count_slider.value))

	# Connect UI signals
	repopulate_btn.pressed.connect(_on_repopulate_pressed)
	toggle_grid_btn.pressed.connect(_on_toggle_grid_pressed)
	color_mode_btn.pressed.connect(_on_color_mode_pressed)
	clear_btn.pressed.connect(_on_clear_pressed)
	pause_btn.pressed.connect(_on_pause_pressed)
	rules_btn.pressed.connect(simulation_plane.randomize_rules)
	default_rules_btn.pressed.connect(simulation_plane.set_default_rules)
	count_slider.value_changed.connect(_on_count_slider_changed)

	_update_info_label(simulation_plane.particle_count)


func _process(_delta: float) -> void:
	_update_info_label(simulation_plane.particle_count)


func _on_particles_updated(count: int) -> void:
	_update_info_label(count)


func _update_info_label(count: int) -> void:
	var pw: int = int(simulation_plane.plane_size.x)
	var ph: int = int(simulation_plane.plane_size.y)
	var fps: int = int(Engine.get_frames_per_second())
	info_label.text = "Particles: %d  |  Plane: %d × %d  |  FPS: %d" % [count, pw, ph, fps]


func _on_repopulate_pressed() -> void:
	var count: int = int(count_slider.value)
	simulation_plane.populate_particles(count)


func _on_toggle_grid_pressed() -> void:
	simulation_plane.show_grid = not simulation_plane.show_grid


func _on_color_mode_pressed() -> void:
	simulation_plane.random_colors = not simulation_plane.random_colors
	color_mode_btn.text = "Colors: Multi" if simulation_plane.random_colors else "Colors: Single"
	simulation_plane.populate_particles(int(count_slider.value))


func _on_clear_pressed() -> void:
	simulation_plane.clear_particles()


func _on_pause_pressed() -> void:
	simulation_plane.simulation_running = not simulation_plane.simulation_running
	pause_btn.text = "Pause" if simulation_plane.simulation_running else "Play"


func _on_count_slider_changed(value: float) -> void:
	count_display.text = str(int(value))


## Allow users to click on the plane to spawn additional particles interactively.
func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_LEFT:
			var local_pos: Vector2 = simulation_plane.to_local(event.position)
			if simulation_plane.is_point_inside(local_pos):
				simulation_plane.add_particle(local_pos)
