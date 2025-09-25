extends Node2D

@export var interaction_text: String = ""
@export var inter_id: String = "ladder"

@onready var anim = $AnimatedSprite2D
@onready var area = $Area2D
@onready var interaction_box = $interaction_box
@onready var collisionShape = $Area2D/CollisionShape2D

var player_near = false
func _ready():
	interaction_box.visible = false
	interaction_box.text = interaction_text
	area.body_entered.connect(_on_body_entered)
	area.body_exited.connect(_on_body_exited)
	
	match inter_id:
		"ladder":
			anim.play("ladder")
			collisionShape.shape =  RectangleShape2D.new()
			collisionShape.shape.extents = Vector2(24,32)
			collisionShape.position = Vector2(-4,0)
	
func _on_body_entered(body):
	if body.is_in_group("player"):
		interaction_box.visible = true
		player_near = true

func _on_body_exited(body):
	if body.is_in_group("player"):
		interaction_box.visible = false
		player_near = false
		match inter_id:
			"ladder":
					body.contact_ladder = false

func _process(_delta):
	if player_near:
		_trigger_event()
		
func _trigger_event():
	match inter_id:
		"ladder":
			if get_tree().current_scene.has_node("Player"):
				get_tree().current_scene.get_node("Player").contact_ladder = true
