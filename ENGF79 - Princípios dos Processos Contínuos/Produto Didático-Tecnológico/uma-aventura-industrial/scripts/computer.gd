extends StaticBody2D

@export var interaction_text: String = "[E]"
@export var event_id: String = "computador1" # identificador do evento

@onready var interaction_box = $InteractionBox
@onready var area = $Area2D

var player_near = false
var panel_open: bool = false

func _ready():
	interaction_box.visible = false
	interaction_box.text = interaction_text
	area.body_entered.connect(_on_body_entered)
	area.body_exited.connect(_on_body_exited)

func _on_body_entered(body):
	if body.is_in_group("player"):
		interaction_box.visible = true
		player_near = true

func _on_body_exited(body):
	if body.is_in_group("player"):
		interaction_box.visible = false
		player_near = false

func _process(_delta):
	if panel_open:
		return
	if player_near and Input.is_action_just_pressed("interact"):
		_trigger_event()

func _trigger_event():
	if panel_open:
		return
	print("Evento disparado!")
	match event_id:
		"computador1":
			var miniGameRLC = load("res://scenes/mini_game_rlc.tscn")
			var panel_instance =  miniGameRLC.instantiate()
			get_tree().current_scene.add_child(panel_instance)
			var global_pos = global_position
			panel_instance.position = global_pos + Vector2(-200, -150)  # 100px acima
		"computador2":
			print("Mostrando mensagem secreta")
		"computador3":
			print("Abrindo minigame de hacking")
		_:
			print("Evento padrão")
