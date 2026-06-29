import taichi as ti
from .config import BOUNCE_COEF, DRAG_COEF, GRAVITY_STRENGTH, MOUSE_DEAD_ZONE, NUM_PARTICLES

pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)
vel = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)


@ti.kernel
def init_particles():
    for i in range(NUM_PARTICLES):
        pos[i] = [ti.random(), ti.random()]
        vel[i] = [0.0, 0.0]


@ti.kernel
def update_particles(mouse_x: ti.f32, mouse_y: ti.f32):
    mouse_pos = ti.Vector([mouse_x, mouse_y])
    for i in range(NUM_PARTICLES):
        offset = mouse_pos - pos[i]
        distance = offset.norm() + 1e-4
        if distance > MOUSE_DEAD_ZONE:
            strength = GRAVITY_STRENGTH / ti.sqrt(distance)
            vel[i] += offset.normalized() * strength
        vel[i] *= DRAG_COEF
        pos[i] += vel[i]
        for axis in ti.static(range(2)):
            if pos[i][axis] < 0.0:
                pos[i][axis] = 0.0
                vel[i][axis] *= BOUNCE_COEF
            elif pos[i][axis] > 1.0:
                pos[i][axis] = 1.0
                vel[i][axis] *= BOUNCE_COEF
