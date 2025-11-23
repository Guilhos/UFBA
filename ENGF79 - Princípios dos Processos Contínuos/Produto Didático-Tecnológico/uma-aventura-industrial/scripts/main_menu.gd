extends Control

func _ready() -> void:
	for button in get_tree().get_nodes_in_group("buttonMenu"):
		button.pressed.connect(Callable(self,"on_button_pressed").bind(button))
		button.mouse_exited.connect(Callable(self, "mouse_interaction").bind(button, "exited"))
		button.mouse_entered.connect(Callable(self, "mouse_interaction").bind(button, "entered"))
		
func on_button_pressed(button: Button) -> void:
	match button.name:
		"novo_jogo":
			var _game: bool = get_tree().change_scene_to_file("res://scenes/level.tscn")
		"continuar":
			var _game: bool = get_tree().change_scene_to_file("res://scenes/level.tscn")
		"sair":
			get_tree().quit()
		"DevGuilhos":
			var _open_channel: bool = OS.shell_open("https://github.com/Guilhos")
			
func mouse_interaction(button: Button, state: String) -> void:
	match state:
		"exited":
			button.modulate.a = 1
		"entered":
			button.modulate.a = 0.5
