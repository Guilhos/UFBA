extends Node2D

@export var font: Font

var points: Array = []
var max_points: int = 400

var q: float = 0.0   # carga
var i: float = 0.0   # corrente
var R: float = 10.0
var L: float = 1.0
var C: float = 0.1
var v_in: float = 5.0
var dt: float = 0.01

func _process(delta):
	# Atualiza o sistema numérico
	var dqdt = i
	var didt = (v_in - R*i - q/C) / L
	q += dqdt * dt
	i += didt * dt
	
	# Adiciona o ponto ao gráfico
	var x = 0
	if points.size() > 0:
		x = points[points.size()-1].x + 2
	var y = -i*50 + 100
	points.append(Vector2(x, y))
	
	# Remove pontos antigos
	if points.size() > max_points:
		points.pop_front()
	# Redesenha
	queue_redraw()

func _draw():
	if points.size() < 2:
		return

	var panel_width = 250  # largura em pixels do painel ou do Graph
	var panel_height = 150  # altura

	# desenhar eixo horizontal e vertical
	draw_line(Vector2(0, panel_height/2), Vector2(panel_width, panel_height/2), Color(1,1,1), 1)
	draw_line(Vector2(0, 0), Vector2(0, panel_height), Color(1,1,1), 1)

	# desenhar gráfico mapeando X
	for j in range(points.size() - 1):
		var x1 = j * panel_width / max_points
		var x2 = (j+1) * panel_width / max_points
		var y1 = points[j].y - panel_height/6 
		var y2 = points[j+1].y - panel_height/6 
		draw_line(Vector2(x1, y1), Vector2(x2, y2), Color(0,1,0), 2)

	# legenda
	draw_string(font, Vector2(10, 20), "i(t) [A]")
	draw_string(font, Vector2(panel_width-50, panel_height-10), "t [s]")
