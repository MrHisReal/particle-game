class_name RuleMatrix
extends Control

## Shows the attraction rules as a grid of colored cells, where each row's species reacts to
## each column's species. Left-click or scroll up on a cell to make it more attractive,
## right-click or scroll down to make it more repulsive.

const STEP := 0.1
## Size of the row and column header strips holding the species swatches.
const HEADER := 22.0
const CELL_GAP := 2.0
## Smallest cell that still has room for its value printed inside.
const MIN_LABELED_CELL := 30.0
const NEUTRAL_COLOR := Color(0.15, 0.17, 0.22)
const ATTRACT_COLOR := Color(0.2, 0.75, 0.4)
const REPEL_COLOR := Color(0.9, 0.28, 0.3)

var plane: SimulationPlane

var _hover := Vector2i(-1, -1)


func _ready() -> void:
	custom_minimum_size = Vector2(220.0, 220.0)
	size_flags_horizontal = Control.SIZE_SHRINK_CENTER
	mouse_filter = Control.MOUSE_FILTER_STOP
	tooltip_text = "Left-click: attract more. Right-click: repel more."
	plane.rules_changed.connect(queue_redraw)
	mouse_exited.connect(_set_hover.bind(Vector2i(-1, -1)))


func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		_set_hover(_cell_at(event.position))
	elif event is InputEventMouseButton and event.pressed:
		var cell: Vector2i = _cell_at(event.position)
		if cell.x < 0:
			return
		var change: float
		match event.button_index:
			MOUSE_BUTTON_LEFT, MOUSE_BUTTON_WHEEL_UP:
				change = STEP
			MOUSE_BUTTON_RIGHT, MOUSE_BUTTON_WHEEL_DOWN:
				change = -STEP
			_:
				return
		plane.set_rule(cell.y, cell.x, plane.attraction_rules[cell.y][cell.x] + change)
		accept_event()


func _get_tooltip(at_position: Vector2) -> String:
	var cell: Vector2i = _cell_at(at_position)
	if cell.x < 0:
		return tooltip_text
	var value: float = plane.attraction_rules[cell.y][cell.x]
	var effect: String = "attracted to" if value > 0.0 else ("repelled by" if value < 0.0 else "ignores")
	return "%s is %s %s (%+.1f)\nLeft-click: attract more. Right-click: repel more." % [
		plane.species_name(cell.y), effect, plane.species_name(cell.x), value,
	]


func _draw() -> void:
	var n: int = plane.attraction_rules.size()
	if n == 0:
		return
	var cell: float = _cell_size(n)
	var origin: Vector2 = _origin(n)
	var font: Font = get_theme_default_font()
	var font_size: int = get_theme_default_font_size() - 2
	var swatch_radius: float = minf(HEADER, cell) * 0.3

	for i in range(n):
		var color: Color = plane.species_display_color(i)
		var along: float = HEADER + cell * (i + 0.5)
		draw_circle(origin + Vector2(along, HEADER * 0.5), swatch_radius, color)
		draw_circle(origin + Vector2(HEADER * 0.5, along), swatch_radius, color)

	for a in range(n):
		for b in range(n):
			var value: float = plane.attraction_rules[a][b]
			var rect: Rect2 = _cell_rect(origin, cell, a, b)
			var tint: Color = ATTRACT_COLOR if value > 0.0 else REPEL_COLOR
			draw_rect(rect, NEUTRAL_COLOR.lerp(tint, absf(value)))
			if cell >= MIN_LABELED_CELL:
				var baseline: float = rect.position.y + (rect.size.y + font.get_ascent(font_size) - font.get_descent(font_size)) * 0.5
				draw_string(font, Vector2(rect.position.x, baseline), "%+.1f" % value,
					HORIZONTAL_ALIGNMENT_CENTER, rect.size.x, font_size, Color(1.0, 1.0, 1.0, 0.9))

	if _hover.x >= 0 and _hover.y < n and _hover.x < n:
		draw_rect(_cell_rect(origin, cell, _hover.y, _hover.x), Color(1.0, 1.0, 1.0, 0.8), false, 1.5)


## Returns (column, row) of the cell under pos, or (-1, -1) if there is none.
func _cell_at(pos: Vector2) -> Vector2i:
	var n: int = plane.attraction_rules.size()
	if n == 0:
		return Vector2i(-1, -1)
	var local: Vector2 = (pos - _origin(n) - Vector2.ONE * HEADER) / _cell_size(n)
	var cell := Vector2i(floori(local.x), floori(local.y))
	if cell.x < 0 or cell.y < 0 or cell.x >= n or cell.y >= n:
		return Vector2i(-1, -1)
	return cell


func _cell_size(n: int) -> float:
	return (minf(size.x, size.y) - HEADER) / n


## Top-left corner of the header strips, centering the grid horizontally.
func _origin(n: int) -> Vector2:
	return Vector2((size.x - HEADER - _cell_size(n) * n) * 0.5, 0.0)


func _cell_rect(origin: Vector2, cell: float, row: int, column: int) -> Rect2:
	var corner: Vector2 = origin + Vector2(HEADER + cell * column, HEADER + cell * row)
	return Rect2(corner, Vector2(cell, cell)).grow(-CELL_GAP * 0.5)


func _set_hover(cell: Vector2i) -> void:
	if cell != _hover:
		_hover = cell
		queue_redraw()
