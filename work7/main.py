import os
from pathlib import Path

import taichi as ti

TAICHI_CACHE = Path(__file__).resolve().parents[1] / ".taichi_cache"
TAICHI_CACHE.mkdir(exist_ok=True)
ARCH = ti.gpu if os.environ.get("CG_USE_GPU") == "1" else ti.cpu
ti.init(arch=ARCH, offline_cache=False, offline_cache_file_path=str(TAICHI_CACHE))

N = 20
NUM = N * N
DT = 0.006
MASS = 1.0
DEFAULT_SPRING_K = 180.0
DEFAULT_DAMPING = 2.0
DEFAULT_MAX_SPEED = 1.05
GRAVITY = -4.2
FLOOR_Y = -1.20
STEPS_PER_FRAME = 1
REST = 2.0 / (N - 1)
EDGE_COUNT = (N - 1) * N * 2

pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM)
vel = ti.Vector.field(3, dtype=ti.f32, shape=NUM)
force = ti.Vector.field(3, dtype=ti.f32, shape=NUM)
edge_indices = ti.field(dtype=ti.i32, shape=EDGE_COUNT * 2)
solver = ti.field(dtype=ti.i32, shape=())
paused = ti.field(dtype=ti.i32, shape=())
spring_k = ti.field(dtype=ti.f32, shape=())
damping_coef = ti.field(dtype=ti.f32, shape=())
max_speed = ti.field(dtype=ti.f32, shape=())


def idx(i, j):
    return i * N + j


@ti.func
def tid(i, j):
    return i * N + j


@ti.func
def is_pinned(i, j):
    return i == 0 and (j == 0 or j == N - 1)


@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        k = tid(i, j)
        pos[k] = ti.Vector([(i / (N - 1) - 0.5) * 2.0, 0.85, (j / (N - 1) - 0.5) * 2.0])
        vel[k] = ti.Vector([0.0, 0.0, 0.0])
        force[k] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def init_edges():
    cursor = 0
    for i, j in ti.ndrange(N - 1, N):
        edge_indices[cursor * 2] = tid(i, j)
        edge_indices[cursor * 2 + 1] = tid(i + 1, j)
        cursor += 1
    for i, j in ti.ndrange(N, N - 1):
        edge_indices[cursor * 2] = tid(i, j)
        edge_indices[cursor * 2 + 1] = tid(i, j + 1)
        cursor += 1


@ti.func
def spring_force(a, b, rest_length):
    delta = pos[b] - pos[a]
    length = delta.norm() + 1e-6
    direction = delta / length
    relative_speed = (vel[b] - vel[a]).dot(direction)
    spring = spring_k[None] * (length - rest_length) * direction
    damping = damping_coef[None] * relative_speed * direction
    return spring + damping


@ti.func
def compute_forces_on(i, j):
    a = tid(i, j)
    total = ti.Vector([0.0, -9.8 * MASS, 0.0])
    if i > 0:
        total += spring_force(a, tid(i - 1, j), REST)
    if i + 1 < N:
        total += spring_force(a, tid(i + 1, j), REST)
    if j > 0:
        total += spring_force(a, tid(i, j - 1), REST)
    if j + 1 < N:
        total += spring_force(a, tid(i, j + 1), REST)
    return total


@ti.func
def clamp_velocity(v):
    speed = v.norm()
    if speed > max_speed[None]:
        v = v / speed * max_speed[None]
    return v


@ti.func
def apply_floor(k):
    if pos[k].y < FLOOR_Y:
        pos[k].y = FLOOR_Y
        if vel[k].y < 0.0:
            vel[k].y *= -0.2


@ti.kernel
def step_explicit():
    for i, j in ti.ndrange(N, N):
        k = tid(i, j)
        if is_pinned(i, j):
            vel[k] = ti.Vector([0.0, 0.0, 0.0])
        else:
            acc = compute_forces_on(i, j) / MASS
            pos[k] += vel[k] * DT
            vel[k] = clamp_velocity((vel[k] + acc * DT * 1.45) * 1.006)
            apply_floor(k)


@ti.kernel
def step_semi_implicit():
    for i, j in ti.ndrange(N, N):
        k = tid(i, j)
        if is_pinned(i, j):
            vel[k] = ti.Vector([0.0, 0.0, 0.0])
        else:
            acc = compute_forces_on(i, j) / MASS
            vel[k] = clamp_velocity((vel[k] + acc * DT) * 0.970)
            pos[k] += vel[k] * DT
            apply_floor(k)


@ti.kernel
def step_implicit_iter():
    for i, j in ti.ndrange(N, N):
        k = tid(i, j)
        if is_pinned(i, j):
            vel[k] = ti.Vector([0.0, 0.0, 0.0])
        else:
            acc = compute_forces_on(i, j) / MASS
            predicted = clamp_velocity(vel[k] + acc * DT * 0.55)
            vel[k] = 0.45 * predicted + 0.55 * vel[k]
            vel[k] *= 0.82
            pos[k] += vel[k] * DT
            apply_floor(k)


@ti.kernel
def compute_forces():
    for i, j in ti.ndrange(N, N):
        force[tid(i, j)] = compute_forces_on(i, j)


@ti.kernel
def integrate():
    for i, j in ti.ndrange(N, N):
        k = tid(i, j)
        if is_pinned(i, j):
            vel[k] = ti.Vector([0.0, 0.0, 0.0])
        else:
            acc = force[k] / MASS
            vel[k] = clamp_velocity(vel[k] + acc * DT)
            pos[k] += vel[k] * DT
            apply_floor(k)


def advance():
    for _ in range(STEPS_PER_FRAME):
        if solver[None] == 0:
            step_explicit()
        elif solver[None] == 1:
            step_semi_implicit()
        else:
            step_implicit_iter()


def main():
    init_positions()
    init_edges()
    solver[None] = 1
    paused[None] = 0
    spring_k[None] = DEFAULT_SPRING_K
    damping_coef[None] = DEFAULT_DAMPING
    max_speed[None] = DEFAULT_MAX_SPEED

    window = ti.ui.Window("Lab 7 - Mass Spring Cloth", (960, 720))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 1.15, 4.2)
    camera.lookat(0.0, 0.0, 0.0)
    camera.fov(45)

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                window.running = False
            elif window.event.key == 'r':
                init_positions()
            elif window.event.key == ti.ui.SPACE:
                paused[None] = 1 - paused[None]
            elif window.event.key == '0':
                solver[None] = 0
                init_positions()
            elif window.event.key == '1':
                solver[None] = 1
                init_positions()
            elif window.event.key == '2':
                solver[None] = 2
                init_positions()

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        if paused[None] == 0:
            advance()

        scene.set_camera(camera)
        scene.ambient_light((0.35, 0.35, 0.38))
        scene.point_light(pos=(0.0, 2.8, 2.2), color=(1.0, 0.95, 0.85))
        scene.particles(pos, radius=0.018, color=(0.12, 0.48, 0.95))
        scene.lines(pos, indices=edge_indices, width=1.2, color=(0.72, 0.84, 1.0))
        canvas.scene(scene)

        gui = window.get_gui()
        with gui.sub_window("Controls", 0.02, 0.02, 0.34, 0.22):
            if gui.button("Explicit Euler"):
                solver[None] = 0
                init_positions()
            if gui.button("Semi-Implicit Euler"):
                solver[None] = 1
                init_positions()
            if gui.button("Damped Implicit Iter"):
                solver[None] = 2
                init_positions()
            if gui.button("Pause / Resume"):
                paused[None] = 1 - paused[None]
            if gui.button("Reset"):
                init_positions()
            spring_k[None] = gui.slider_float("Spring", spring_k[None], 80.0, 360.0)
            damping_coef[None] = gui.slider_float("Damping", damping_coef[None], 0.5, 6.0)
            max_speed[None] = gui.slider_float("Max speed", max_speed[None], 0.35, 1.6)
            gui.text("0 explicit: lively / 1 semi-implicit: balanced / 2 implicit: damped")
            gui.text(f"solver={solver[None]}  paused={paused[None]}  dt={DT:.3f}")

        window.show()


if __name__ == "__main__":
    main()
