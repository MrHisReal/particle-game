class_name Main
extends Node2D

## Plane sizes offered in the Resolution menu. Larger planes fit more particles.
const RESOLUTIONS: Array[Vector2i] = [
	Vector2i(960, 540),
	Vector2i(1920, 1080),
	Vector2i(2560, 1440),
	Vector2i(3840, 2160),
	Vector2i(7680, 4320),
]
## Gap kept between the plane and the surrounding UI.
const PLANE_PADDING := 12.0
## Seconds between refreshes of the stats in the top bar.
const STATS_INTERVAL := 0.25
const SLIDER_LABEL_WIDTH := 112.0
const SLIDER_VALUE_WIDTH := 64.0
const SHORTCUTS := [
	["Space", "Pause / play"],
	[".", "Step one frame"],
	["R", "Respawn particles"],
	["N", "New random rules"],
	["C", "Clear particles"],
	["G", "Toggle grid"],
	["Click", "Add a particle"],
]

@onready var simulation_plane: SimulationPlane = $SimulationPlane
@onready var ui: Control = $HUD/UI
@onready var top_bar: Control = %TopBar
@onready var side_panel: Control = %SidePanel
@onready var stats_label: Label = %StatsLabel
@onready var pause_btn: Button = %PauseBtn
@onready var step_btn: Button = %StepBtn
@onready var controls: VBoxContainer = %Controls

var _count_slider: HSlider
var _grid_toggle: CheckButton
## Sliders bound to SimulationPlane properties, with the values they started at.
var _property_sliders: Dictionary[StringName, HSlider] = {}
var _property_defaults: Dictionary[StringName, float] = {}
var _stats_timer: float = 0.0


func _ready() -> void:
	ui.theme = _build_theme()
	_build_controls()

	pause_btn.pressed.connect(_toggle_pause)
	step_btn.pressed.connect(simulation_plane.step_once)
	simulation_plane.particles_updated.connect(_update_stats.unbind(1))
	get_viewport().size_changed.connect(_fit_plane)
	top_bar.resized.connect(_fit_plane)
	side_panel.resized.connect(_fit_plane)

	_fit_plane()
	_update_pause_ui()
	_update_stats()


func _process(delta: float) -> void:
	_stats_timer += delta
	if _stats_timer >= STATS_INTERVAL:
		_stats_timer = 0.0
		_update_stats()


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		var local_pos: Vector2 = simulation_plane.to_local(event.position)
		if simulation_plane.is_point_inside(local_pos):
			simulation_plane.add_particle(local_pos)
			get_viewport().set_input_as_handled()
	elif event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_SPACE:
				_toggle_pause()
			KEY_PERIOD:
				simulation_plane.step_once()
			KEY_R:
				_respawn()
			KEY_N:
				simulation_plane.randomize_rules()
			KEY_C:
				simulation_plane.clear_particles()
			KEY_G:
				_grid_toggle.button_pressed = not _grid_toggle.button_pressed
			_:
				return
		get_viewport().set_input_as_handled()


func _build_controls() -> void:
	var particles: VBoxContainer = _add_section("Particles")
	var resolution_option := OptionButton.new()
	resolution_option.focus_mode = Control.FOCUS_NONE
	resolution_option.tooltip_text = "Size of the simulation plane. Larger planes hold more particles."
	for resolution in RESOLUTIONS:
		resolution_option.add_item("%d × %d" % [resolution.x, resolution.y])
	resolution_option.select(RESOLUTIONS.find(Vector2i(simulation_plane.plane_size)))
	resolution_option.item_selected.connect(_on_resolution_selected)
	_add_row(particles, "Resolution", resolution_option)

	_count_slider = _add_slider(particles, "Count", 10.0, 1_000_000.0, 10.0,
		simulation_plane.initial_particle_count,
		"Number of particles. Applied when you release the slider.", _format_count_value)
	_count_slider.exp_edit = true
	_count_slider.drag_ended.connect(_on_count_drag_ended)

	_add_buttons(particles, [
		_make_button("Respawn", _respawn, "Scatter a fresh set of particles (R)"),
		_make_button("Clear", simulation_plane.clear_particles, "Remove every particle (C)"),
	])
	_add_toggle(particles, "Multiple colors", simulation_plane.random_colors,
		"Give particles different species, each with its own rules", _on_colors_toggled)

	var rules: VBoxContainer = _add_section("Rules")
	var matrix := RuleMatrix.new()
	matrix.plane = simulation_plane
	rules.add_child(matrix)
	_add_hint(rules, "Row reacts to column. Green attracts, red repels. Click a cell to make it more attractive, right-click to make it more repulsive.")
	_add_buttons(rules, [
		_make_button("Randomize", simulation_plane.randomize_rules, "Pick new random rules (N)"),
		_make_button("Default", simulation_plane.set_default_rules, "Restore the built-in chase cycle"),
	])

	var forces: VBoxContainer = _add_section("Forces")
	_add_property_slider(forces, "Strength", &"force_strength", 0.0, 2000.0, 10.0, "%.0f",
		"Scales every color force (pixels / second²)")
	_add_property_slider(forces, "Radius", &"interaction_radius", 10.0, 250.0, 1.0, "%.0f",
		"How far particles sense each other. Smaller is much faster.")
	_add_property_slider(forces, "Repulsion zone", &"repulsion_zone", 0.05, 0.9, 0.01, "%.2f",
		"Fraction of the radius where particles always push apart")
	_add_property_slider(forces, "Max speed", &"max_speed", 10.0, 1500.0, 5.0, "%.0f",
		"Upper bound on particle speed (pixels / second)")

	var friction: VBoxContainer = _add_section("Drag & Friction")
	_add_property_slider(friction, "Linear drag", &"linear_drag", 0.0, 20.0, 0.1, "%.1f",
		"Drag proportional to speed, like moving through thick fluid")
	_add_property_slider(friction, "Air drag", &"quadratic_drag", 0.0, 0.02, 0.0001, "%.4f",
		"Drag proportional to speed squared. Mostly slows fast particles.")
	_add_property_slider(friction, "Surface friction", &"surface_friction", 0.0, 300.0, 1.0, "%.0f",
		"Constant slowdown that brings weakly pushed particles to a stop")
	_add_property_slider(friction, "Contact friction", &"contact_friction", 0.0, 2.0, 0.01, "%.2f",
		"How strongly touching particles resist sliding past each other")

	var collisions: VBoxContainer = _add_section("Collisions")
	_add_toggle(collisions, "Solid particles", simulation_plane.particle_collisions,
		"Particles push each other apart instead of overlapping",
		_set_plane_property.bind(&"particle_collisions"))
	_add_property_slider(collisions, "Iterations", &"collision_iterations", 1.0, 16.0, 1.0, "%d",
		"Collision passes per frame. More keeps dense clumps from overlapping.")
	_add_buttons(collisions, [
		_make_button("Reset physics", _reset_physics, "Restore every Forces, Drag & Friction and Collisions slider"),
	])

	var display: VBoxContainer = _add_section("Display")
	_grid_toggle = _add_toggle(display, "Show grid", simulation_plane.show_grid,
		"Draw grid lines on the plane (G)", _set_plane_property.bind(&"show_grid"))

	var help: VBoxContainer = _add_section("Shortcuts")
	var shortcuts := GridContainer.new()
	shortcuts.columns = 2
	shortcuts.add_theme_constant_override("h_separation", 16)
	for shortcut in SHORTCUTS:
		var key := Label.new()
		key.text = shortcut[0]
		key.theme_type_variation = &"KeyLabel"
		shortcuts.add_child(key)
		var action := Label.new()
		action.text = shortcut[1]
		action.theme_type_variation = &"HintLabel"
		shortcuts.add_child(action)
	help.add_child(shortcuts)


func _add_section(title: String) -> VBoxContainer:
	if controls.get_child_count() > 0:
		controls.add_child(HSeparator.new())
	var header := Label.new()
	header.text = title.to_upper()
	header.theme_type_variation = &"SectionHeader"
	controls.add_child(header)
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 6)
	controls.add_child(box)
	return box


func _add_row(parent: Container, label_text: String, control: Control) -> void:
	var row := HBoxContainer.new()
	var label := Label.new()
	label.text = label_text
	label.custom_minimum_size.x = SLIDER_LABEL_WIDTH
	label.tooltip_text = control.tooltip_text
	label.mouse_filter = Control.MOUSE_FILTER_PASS
	row.add_child(label)
	control.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(control)
	parent.add_child(row)


## Adds a labeled slider with a readout of its value, as formatted by format_value(value).
func _add_slider(parent: Container, label_text: String, min_value: float, max_value: float,
		step: float, value: float, tooltip: String, format_value: Callable) -> HSlider:
	var slider := HSlider.new()
	slider.min_value = min_value
	slider.max_value = max_value
	slider.step = step
	slider.value = value
	slider.focus_mode = Control.FOCUS_NONE
	slider.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	slider.tooltip_text = tooltip

	var value_label := Label.new()
	value_label.custom_minimum_size.x = SLIDER_VALUE_WIDTH
	value_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	value_label.theme_type_variation = &"ValueLabel"
	_show_slider_value(slider.value, value_label, format_value)
	slider.value_changed.connect(_show_slider_value.bind(value_label, format_value))

	_add_row(parent, label_text, slider)
	slider.get_parent().add_child(value_label)
	return slider


## Adds a slider that edits a SimulationPlane property live, showing its value with format.
func _add_property_slider(parent: Container, label_text: String, property: StringName,
		min_value: float, max_value: float, step: float, format: String, tooltip: String) -> void:
	var start: float = simulation_plane.get(property)
	var format_value := func(value: float) -> String: return format % value
	var slider: HSlider = _add_slider(parent, label_text, min_value, max_value, step, start, tooltip, format_value)
	slider.value_changed.connect(_set_plane_property.bind(property))
	_property_sliders[property] = slider
	_property_defaults[property] = start


func _show_slider_value(value: float, label: Label, format_value: Callable) -> void:
	label.text = format_value.call(value)


## Sets a SimulationPlane property, keeping integer properties integers.
func _set_plane_property(value: Variant, property: StringName) -> void:
	if typeof(simulation_plane.get(property)) == TYPE_INT:
		value = int(value)
	simulation_plane.set(property, value)


func _add_toggle(parent: Container, text: String, pressed: bool, tooltip: String, on_toggled: Callable) -> CheckButton:
	var toggle := CheckButton.new()
	toggle.text = text
	toggle.button_pressed = pressed
	toggle.focus_mode = Control.FOCUS_NONE
	toggle.tooltip_text = tooltip
	toggle.toggled.connect(on_toggled)
	parent.add_child(toggle)
	return toggle


func _add_hint(parent: Container, text: String) -> void:
	var hint := Label.new()
	hint.text = text
	hint.theme_type_variation = &"HintLabel"
	hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	parent.add_child(hint)


func _add_buttons(parent: Container, buttons: Array[Button]) -> void:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	for button in buttons:
		button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		row.add_child(button)
	parent.add_child(row)


func _make_button(text: String, on_pressed: Callable, tooltip: String) -> Button:
	var button := Button.new()
	button.text = text
	button.focus_mode = Control.FOCUS_NONE
	button.tooltip_text = tooltip
	button.pressed.connect(on_pressed)
	return button


## Fits the plane into the space left of the side panel and below the top bar.
func _fit_plane() -> void:
	side_panel.offset_top = top_bar.size.y
	var view: Vector2 = get_viewport_rect().size
	var area := Rect2(0.0, top_bar.size.y, view.x - side_panel.size.x, view.y - top_bar.size.y)
	simulation_plane.fit_to_rect(area.grow(-PLANE_PADDING))


func _respawn() -> void:
	simulation_plane.populate_particles(int(_count_slider.value))


func _on_count_drag_ended(value_changed: bool) -> void:
	if value_changed:
		_respawn()


func _on_colors_toggled(on: bool) -> void:
	simulation_plane.random_colors = on
	_respawn()


func _toggle_pause() -> void:
	simulation_plane.simulation_running = not simulation_plane.simulation_running
	_update_pause_ui()


func _update_pause_ui() -> void:
	var running: bool = simulation_plane.simulation_running
	pause_btn.text = "Pause" if running else "Play"
	step_btn.disabled = running


func _reset_physics() -> void:
	for property in _property_sliders:
		_property_sliders[property].value = _property_defaults[property]


## Switches the plane size and respawns it with the same particle density.
func _on_resolution_selected(index: int) -> void:
	var new_size := Vector2(RESOLUTIONS[index])
	var old_size: Vector2 = simulation_plane.plane_size
	var area_ratio: float = (new_size.x * new_size.y) / (old_size.x * old_size.y)
	simulation_plane.plane_size = new_size
	_count_slider.value = _count_slider.value * area_ratio
	_respawn()


func _update_stats() -> void:
	var plane: Vector2 = simulation_plane.plane_size
	var parts: PackedStringArray = [
		"%s particles" % _format_count(simulation_plane.particle_count),
		"%d FPS" % Engine.get_frames_per_second(),
		"Plane %d × %d" % [plane.x, plane.y],
		"Zoom %d%%" % roundi(simulation_plane.scale.x * 100.0),
	]
	if not simulation_plane.simulation_running:
		parts.append("Paused")
	stats_label.text = "    ·    ".join(parts)


func _format_count_value(value: float) -> String:
	return _format_count(int(value))


func _format_count(count: int) -> String:
	var digits: String = str(count)
	var grouped: String = ""
	while digits.length() > 3:
		grouped = "," + digits.right(3) + grouped
		digits = digits.left(-3)
	return digits + grouped


func _build_theme() -> Theme:
	var theme := Theme.new()
	theme.default_font_size = 14

	var panel := StyleBoxFlat.new()
	panel.bg_color = Color(0.06, 0.07, 0.1, 0.97)
	panel.border_color = Color(0.17, 0.21, 0.29)
	panel.border_width_bottom = 1
	panel.border_width_left = 1
	theme.set_stylebox("panel", "PanelContainer", panel)

	for type in ["Button", "OptionButton"]:
		theme.set_stylebox("normal", type, _button_box(Color(0.13, 0.16, 0.23)))
		theme.set_stylebox("hover", type, _button_box(Color(0.18, 0.22, 0.31)))
		theme.set_stylebox("pressed", type, _button_box(Color(0.22, 0.4, 0.68)))
		theme.set_stylebox("disabled", type, _button_box(Color(0.1, 0.11, 0.15)))
		theme.set_stylebox("focus", type, StyleBoxEmpty.new())
		theme.set_color("font_disabled_color", type, Color(0.4, 0.43, 0.5))

	var separator := StyleBoxLine.new()
	separator.color = Color(0.17, 0.21, 0.29)
	theme.set_stylebox("separator", "HSeparator", separator)
	theme.set_constant("separation", "HSeparator", 12)

	theme.set_type_variation(&"SectionHeader", &"Label")
	theme.set_color("font_color", &"SectionHeader", Color(0.45, 0.68, 1.0))
	theme.set_font_size("font_size", &"SectionHeader", 12)

	theme.set_type_variation(&"HintLabel", &"Label")
	theme.set_color("font_color", &"HintLabel", Color(0.55, 0.6, 0.7))
	theme.set_font_size("font_size", &"HintLabel", 12)

	theme.set_type_variation(&"KeyLabel", &"Label")
	theme.set_color("font_color", &"KeyLabel", Color(0.85, 0.88, 0.95))
	theme.set_font_size("font_size", &"KeyLabel", 12)

	theme.set_type_variation(&"ValueLabel", &"Label")
	theme.set_color("font_color", &"ValueLabel", Color(0.7, 0.85, 1.0))
	return theme


func _button_box(color: Color) -> StyleBoxFlat:
	var box := StyleBoxFlat.new()
	box.bg_color = color
	box.set_corner_radius_all(4)
	box.content_margin_left = 10.0
	box.content_margin_right = 10.0
	box.content_margin_top = 5.0
	box.content_margin_bottom = 5.0
	return box
