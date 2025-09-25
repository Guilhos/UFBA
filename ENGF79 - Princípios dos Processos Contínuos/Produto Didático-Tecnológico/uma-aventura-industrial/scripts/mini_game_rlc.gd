extends Control

@onready var graph = $Panel/Graph
@onready var slider_r = $Panel/slider_R
@onready var slider_l = $Panel/slider_L
@onready var slider_c = $Panel/slider_C
@onready var button_close = $Panel/Button
@onready var label_r = $Panel/LabelR
@onready var label_l = $Panel/LabelL
@onready var label_c = $Panel/LabelC
@onready var label_sis = $Panel/Sistema

var R: float = 10.0
var L: float = 1.0
var C: float = 0.01

func _ready():
	if get_tree().current_scene.has_node("Player"):
		get_tree().current_scene.get_node("Player").panel_open = true
	if get_tree().current_scene.has_node("computerBase"):
		get_tree().current_scene.get_node("computerBase").panel_open = true

	# Conectar sliders
	slider_r.value_changed.connect(_on_r_changed)
	slider_l.value_changed.connect(_on_l_changed)
	slider_c.value_changed.connect(_on_c_changed)

	# Conectar botão fechar
	button_close.pressed.connect(_on_close_pressed)

	_update_labels()

func _on_r_changed(value):
	R = value
	graph.R = R
	_update_labels()

func _on_l_changed(value):
	L = value
	graph.L = L
	_update_labels()
	
func _on_c_changed(value):
	C = value
	graph.C = C
	_update_labels()

func _update_labels():
	if R**2 > 4*L/C:
		label_sis.text = "Sobreamortecido"
	elif R**2 == 4*L/C:
		label_sis.text = "Criticamente Amortecido"
	else:
		label_sis.text = "Subamortecido"
	label_r.text = "R = %.2f Ω" % R
	label_l.text = "L = %.2f H" % L
	label_c.text = "C = %.2f F" % C

func _on_close_pressed():
	if get_tree().current_scene.has_node("Player"):
		get_tree().current_scene.get_node("Player").panel_open = false
	if get_tree().current_scene.has_node("computerBase"):
		get_tree().current_scene.get_node("computerBase").panel_open = false
	queue_free()  # fecha o painel
