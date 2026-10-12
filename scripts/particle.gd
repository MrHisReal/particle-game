class_name Particle
extends RefCounted

## Represents a single particle in the 2D simulation plane.

var position: Vector2 = Vector2.ZERO
var velocity: Vector2 = Vector2.ZERO
var radius: float = 4.5
var color: Color = Color.WHITE
var mass: float = 1.0
## Index into the simulation's species table; determines which interaction rules apply.
var species: int = 0

func _init(p_position: Vector2 = Vector2.ZERO, p_color: Color = Color.WHITE, p_radius: float = 4.5, p_species: int = 0) -> void:
	position = p_position
	color = p_color
	radius = p_radius
	species = p_species
