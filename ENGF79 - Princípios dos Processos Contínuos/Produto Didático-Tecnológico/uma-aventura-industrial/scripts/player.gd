extends CharacterBody2D

# Velocidades
var speed = 100
var jump_force = -300
var gravity = 900

var panel_open: bool = false
var contact_ladder: bool = false

@onready var anim = $AnimatedSprite2D
@onready var collisionShape = $CollisionShape2D

func _physics_process(delta):
	if panel_open:
		return
	# Gravidade
	if not is_on_floor() and not contact_ladder:
		velocity.y += gravity * delta

	# Movimento esquerda/direita
	var input_dir = Input.get_axis("ui_left", "ui_right")
	velocity.x = input_dir * speed

	# Pulo
	if Input.is_action_just_pressed("ui_accept") and is_on_floor():
		velocity.y = jump_force
	
	if Input.is_action_pressed("ui_accept") and contact_ladder:
		velocity.y = jump_force/2
		
	if input_dir != 0:
		anim.play("walk")
		anim.flip_h = input_dir < 0
	else:
		anim.play("idle")

	# Move o personagem
	move_and_slide()
