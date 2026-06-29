import taichi as ti
from pathlib import Path
import os

WIDTH, HEIGHT = 400, 300
MAX_DIST = 1e6
EPS = 1e-3
BOUNCES = 2

TAICHI_CACHE = Path(__file__).resolve().parents[1] / ".taichi_cache"
TAICHI_CACHE.mkdir(exist_ok=True)
ARCH = ti.gpu if os.environ.get("CG_USE_GPU") == "1" else ti.cpu
ti.init(arch=ARCH, offline_cache=False, offline_cache_file_path=str(TAICHI_CACHE))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
reflection_strength = ti.field(dtype=ti.f32, shape=())
shadow_strength = ti.field(dtype=ti.f32, shape=())
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())


@ti.func
def norm(v):
    return v / (v.norm() + 1e-6)


@ti.func
def hit_sphere(ro, rd, c, r):
    oc = ro - c
    b = oc.dot(rd)
    c2 = oc.dot(oc) - r * r
    h = b * b - c2
    t = MAX_DIST
    if h > 0.0:
        h = ti.sqrt(h)
        t0 = -b - h
        t1 = -b + h
        if t0 > EPS:
            t = t0
        elif t1 > EPS:
            t = t1
    return t


@ti.func
def hit_plane(ro, rd):
    t = MAX_DIST
    if ti.abs(rd.y) > 1e-5:
        candidate = (-0.85 - ro.y) / rd.y
        if candidate > EPS:
            t = candidate
    return t


@ti.func
def scene_hit(ro, rd):
    t = MAX_DIST
    normal = ti.Vector([0.0, 1.0, 0.0])
    color = ti.Vector([0.0, 0.0, 0.0])
    mirror = 0.0
    ts = hit_sphere(ro, rd, ti.Vector([-0.7, -0.05, -3.1]), 0.72)
    if ts < t:
        t = ts
        p = ro + rd * t
        normal = norm(p - ti.Vector([-0.7, -0.05, -3.1]))
        color = ti.Vector([0.95, 0.28, 0.18])
        mirror = reflection_strength[None]
    ts2 = hit_sphere(ro, rd, ti.Vector([0.75, -0.25, -2.55]), 0.48)
    if ts2 < t:
        t = ts2
        p = ro + rd * t
        normal = norm(p - ti.Vector([0.75, -0.25, -2.55]))
        color = ti.Vector([0.15, 0.55, 0.95])
        mirror = 0.2
    tp = hit_plane(ro, rd)
    if tp < t:
        t = tp
        p = ro + rd * t
        normal = ti.Vector([0.0, 1.0, 0.0])
        checker = (ti.floor(p.x * 2.0) + ti.floor(p.z * 2.0)) % 2.0
        color = ti.Vector([0.75, 0.75, 0.72]) * (0.55 + 0.35 * checker)
        mirror = 0.08
    return t, normal, color, mirror


@ti.func
def visible_to_light(p, n, light):
    to_light = light - p
    dist = to_light.norm()
    rd = to_light / (dist + 1e-6)
    t, _, _, _ = scene_hit(p + n * EPS * 4.0, rd)
    return ti.select(t < dist, 1.0 - shadow_strength[None], 1.0)


@ti.kernel
def render():
    light = light_pos[None]
    for i, j in pixels:
        uv = ti.Vector([(i + 0.5) / WIDTH * 2.0 - 1.0, (j + 0.5) / HEIGHT * 2.0 - 1.0])
        uv.x *= WIDTH / HEIGHT
        ro = ti.Vector([0.0, 0.0, 0.0])
        rd = norm(ti.Vector([uv.x, uv.y, -1.45]))
        throughput = ti.Vector([1.0, 1.0, 1.0])
        final = ti.Vector([0.0, 0.0, 0.0])
        active = 1.0
        for _ in ti.static(range(BOUNCES)):
            if active > 0.5:
                t, n, base, mirror = scene_hit(ro, rd)
                if t >= MAX_DIST * 0.5:
                    sky = ti.Vector([0.08, 0.11, 0.18]) + 0.25 * ti.max(rd.y, 0.0)
                    final += throughput * sky
                    active = 0.0
                else:
                    p = ro + rd * t
                    l = norm(light - p)
                    shade = visible_to_light(p, n, light)
                    diffuse = ti.max(0.0, n.dot(l)) * shade
                    local = base * (0.12 + 0.88 * diffuse)
                    final += throughput * local * (1.0 - mirror)
                    throughput *= mirror
                    rd = norm(rd - 2.0 * rd.dot(n) * n)
                    ro = p + n * EPS * 4.0
        pixels[i, j] = ti.min(final, 1.0)


def main():
    reflection_strength[None] = 0.45
    shadow_strength[None] = 0.75
    light_pos[None] = [-2.0, 4.0, 0.5]
    window = ti.ui.Window("Lab 5 - Whitted Ray Tracing", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    while window.running:
        if window.get_event(ti.ui.PRESS) and window.event.key == ti.ui.ESCAPE:
            window.running = False

        gui = window.get_gui()
        light = light_pos[None]
        with gui.sub_window("Ray Tracing Controls", 0.02, 0.02, 0.34, 0.28):
            light.x = gui.slider_float("Light X", light.x, -4.0, 4.0)
            light.y = gui.slider_float("Light Y", light.y, 0.5, 6.0)
            light.z = gui.slider_float("Light Z", light.z, -3.0, 3.0)
            reflection_strength[None] = gui.slider_float("Reflection", reflection_strength[None], 0.0, 0.9)
            shadow_strength[None] = gui.slider_float("Shadow", shadow_strength[None], 0.0, 1.0)
        light_pos[None] = light

        render()
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    main()
