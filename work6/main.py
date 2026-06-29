from pathlib import Path
import os

import taichi as ti

WIDTH, HEIGHT = 256, 256
STEPS = 220
LR = 0.055
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-6
INITIAL_LIGHT = [-0.75, -0.55, 1.25]

TAICHI_CACHE = Path(__file__).resolve().parents[1] / ".taichi_cache"
TAICHI_CACHE.mkdir(exist_ok=True)
ARCH = ti.gpu if os.environ.get("CG_USE_GPU") == "1" else ti.cpu
ti.init(arch=ARCH, offline_cache=False, offline_cache_file_path=str(TAICHI_CACHE))

target = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
current = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT), needs_grad=True)
display = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH * 2, HEIGHT))
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
light = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)
adam_m = ti.Vector.field(3, dtype=ti.f32, shape=())
adam_v = ti.Vector.field(3, dtype=ti.f32, shape=())
reference_light = ti.Vector([0.85, 0.85, 0.18])


@ti.func
def norm(v):
    return v / (v.norm() + 1e-6)


@ti.func
def leaky_lambert(ndotl):
    return ti.select(ndotl >= 0.0, ndotl, 0.12 * ndotl)


@ti.func
def shade(i, j, light_pos):
    uv = ti.Vector([(i + 0.5) / WIDTH * 2.0 - 1.0, (j + 0.5) / HEIGHT * 2.0 - 1.0])
    r2 = uv.dot(uv)
    color = ti.Vector([0.015, 0.02, 0.03])
    if r2 < 0.62:
        z = ti.sqrt(0.62 - r2)
        p = ti.Vector([uv.x * 0.32 + 0.5, uv.y * 0.32 + 0.5, z * 0.32 + 0.5])
        n = norm(ti.Vector([uv.x, uv.y, z]))
        l = norm(light_pos - p)
        ndotl = n.dot(l)
        intensity = 0.10 + 0.90 * leaky_lambert(ndotl)
        view_dir = ti.Vector([0.0, 0.0, 1.0])
        half_vec = norm(l + view_dir)
        spec = ti.pow(ti.max(half_vec.dot(n), 0.0), 48.0)
        color = ti.Vector([0.85, 0.36, 0.18]) * intensity + ti.Vector([1.0, 0.92, 0.70]) * spec * 0.24
    return ti.max(0.0, ti.min(color, 1.0))


@ti.kernel
def make_target():
    for i, j in target:
        target[i, j] = shade(i, j, reference_light)


@ti.kernel
def render_current():
    for i, j in current:
        current[i, j] = shade(i, j, light[None])


@ti.kernel
def clear_loss():
    loss[None] = 0.0


@ti.kernel
def compute_loss():
    for i, j in current:
        d = current[i, j] - target[i, j]
        loss[None] += d.dot(d) / (WIDTH * HEIGHT)


@ti.kernel
def adam_update(step: ti.i32):
    grad = light.grad[None]
    adam_m[None] = ADAM_BETA1 * adam_m[None] + (1.0 - ADAM_BETA1) * grad
    adam_v[None] = ADAM_BETA2 * adam_v[None] + (1.0 - ADAM_BETA2) * grad * grad
    beta1_pow = ti.pow(ADAM_BETA1, ti.cast(step, ti.f32))
    beta2_pow = ti.pow(ADAM_BETA2, ti.cast(step, ti.f32))
    m_hat = adam_m[None] / (1.0 - beta1_pow)
    v_hat = adam_v[None] / (1.0 - beta2_pow)
    light[None] -= LR * m_hat / (ti.sqrt(v_hat) + ADAM_EPS)
    light.grad[None] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def compose_display():
    for i, j in display:
        if i < WIDTH:
            display[i, j] = target[i, j]
        else:
            display[i, j] = current[i - WIDTH, j]


def reset_state() -> tuple[int, float]:
    light[None] = INITIAL_LIGHT
    adam_m[None] = [0.0, 0.0, 0.0]
    adam_v[None] = [0.0, 0.0, 0.0]
    render_current()
    clear_loss()
    compute_loss()
    return 0, float(loss[None])


def optimize_one_step(step: int) -> float:
    clear_loss()
    with ti.ad.Tape(loss):
        render_current()
        compute_loss()
    value = float(loss[None])
    adam_update(step)
    return value


def main():
    make_target()
    frame, last_loss = reset_state()
    playing = False
    steps_per_frame = 1
    gui = ti.GUI("Lab 6 Easy - Differentiable Rendering", res=(WIDTH * 2, HEIGHT))
    while gui.running:
        for event in gui.get_events(ti.GUI.PRESS):
            if event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif event.key == ' ':
                playing = not playing
            elif event.key == 'r':
                frame, last_loss = reset_state()
                playing = False
            elif event.key == 'n' and frame < STEPS:
                frame += 1
                last_loss = optimize_one_step(frame)
            elif event.key == '1':
                steps_per_frame = 1
            elif event.key == '2':
                steps_per_frame = 3
            elif event.key == '3':
                steps_per_frame = 8

        if playing and frame < STEPS:
            for _ in range(steps_per_frame):
                if frame < STEPS:
                    frame += 1
                    last_loss = optimize_one_step(frame)

        render_current()
        compose_display()
        gui.set_image(display)
        gui.text("left: target", pos=(0.03, 0.93), color=0xFFFFFF)
        gui.text("right: optimized", pos=(0.54, 0.93), color=0xFFFFFF)
        gui.text(
            f"Space play/pause | N step | R reset | 1/2/3 speed={steps_per_frame} | step {frame}/{STEPS} | loss {last_loss:.6f}",
            pos=(0.02, 0.07),
            color=0xFFFFFF,
        )
        gui.text(f"light={light[None]}", pos=(0.02, 0.02), color=0xFFFFFF)
        gui.show()


if __name__ == "__main__":
    main()
