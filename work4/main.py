import taichi as ti
from pathlib import Path
import os

WIDTH, HEIGHT = 480, 360
MAX_DIST = 1e6
EPS = 1e-3

TAICHI_CACHE = Path(__file__).resolve().parents[1] / ".taichi_cache"
TAICHI_CACHE.mkdir(exist_ok=True)
ARCH = ti.gpu if os.environ.get("CG_USE_GPU") == "1" else ti.cpu
ti.init(arch=ARCH, offline_cache=False, offline_cache_file_path=str(TAICHI_CACHE))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
ka = ti.field(dtype=ti.f32, shape=())
kd = ti.field(dtype=ti.f32, shape=())
ks = ti.field(dtype=ti.f32, shape=())
shininess = ti.field(dtype=ti.f32, shape=())


@ti.func
def normalize(v):
    return v / (v.norm() + 1e-6)


@ti.func
def intersect_sphere(origin, direction, center, radius):
    oc = origin - center
    a = direction.dot(direction)
    b = 2.0 * oc.dot(direction)
    c = oc.dot(oc) - radius * radius
    disc = b * b - 4.0 * a * c
    t = MAX_DIST
    if disc >= 0.0:
        root = ti.sqrt(disc)
        t0 = (-b - root) / (2.0 * a)
        t1 = (-b + root) / (2.0 * a)
        if t0 > EPS:
            t = t0
        elif t1 > EPS:
            t = t1
    return t


@ti.func
def intersect_cone(origin, direction):
    apex = ti.Vector([0.7, 0.85, -3.2])
    height = 1.35
    radius = 0.55
    k = radius / height
    rel_o = origin - apex
    dxz = direction.x * direction.x + direction.z * direction.z
    oxz = rel_o.x * rel_o.x + rel_o.z * rel_o.z
    a = dxz - k * k * direction.y * direction.y
    b = 2.0 * (rel_o.x * direction.x + rel_o.z * direction.z - k * k * rel_o.y * direction.y)
    c = oxz - k * k * rel_o.y * rel_o.y
    t = MAX_DIST
    disc = b * b - 4.0 * a * c
    if disc >= 0.0 and ti.abs(a) > 1e-6:
        root = ti.sqrt(disc)
        for candidate in ti.static(range(2)):
            tt = (-b + (2.0 * candidate - 1.0) * root) / (2.0 * a)
            y = rel_o.y + tt * direction.y
            if tt > EPS and -height <= y <= 0.0 and tt < t:
                t = tt
    return t


@ti.func
def phong(point, normal, base_color):
    light_pos = ti.Vector([-2.5, 4.0, 1.5])
    light_color = ti.Vector([1.0, 0.95, 0.86])
    view_dir = normalize(ti.Vector([0.0, 0.0, 0.0]) - point)
    light_dir = normalize(light_pos - point)
    half_vec = normalize(light_dir + view_dir)
    ambient = ka[None] * base_color
    diffuse = kd[None] * ti.max(0.0, normal.dot(light_dir)) * base_color * light_color
    specular = ks[None] * ti.pow(ti.max(0.0, normal.dot(half_vec)), shininess[None]) * light_color
    return ti.min(ambient + diffuse + specular, 1.0)


@ti.kernel
def render():
    for i, j in pixels:
        uv = ti.Vector([(i + 0.5) / WIDTH * 2.0 - 1.0, (j + 0.5) / HEIGHT * 2.0 - 1.0])
        uv.x *= WIDTH / HEIGHT
        origin = ti.Vector([0.0, 0.0, 0.0])
        direction = normalize(ti.Vector([uv.x, uv.y, -1.5]))
        color = ti.Vector([0.03, 0.04, 0.07]) + 0.12 * ti.Vector([uv.y + 0.4, uv.y + 0.5, uv.y + 0.7])
        closest = MAX_DIST
        sphere_center = ti.Vector([-0.65, 0.05, -3.0])
        t_s = intersect_sphere(origin, direction, sphere_center, 0.75)
        if t_s < closest:
            closest = t_s
            point = origin + direction * t_s
            normal = normalize(point - sphere_center)
            color = phong(point, normal, ti.Vector([0.9, 0.25, 0.18]))
        t_c = intersect_cone(origin, direction)
        if t_c < closest:
            closest = t_c
            point = origin + direction * t_c
            apex = ti.Vector([0.7, 0.85, -3.2])
            rel = point - apex
            normal = normalize(ti.Vector([rel.x, -0.25 * rel.y, rel.z]))
            color = phong(point, normal, ti.Vector([0.2, 0.55, 1.0]))
        pixels[i, j] = color


def main():
    ka[None], kd[None], ks[None], shininess[None] = 0.2, 0.7, 0.5, 32.0
    window = ti.ui.Window("Lab 4 - Phong Lighting", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    while window.running:
        if window.get_event(ti.ui.PRESS) and window.event.key == ti.ui.ESCAPE:
            window.running = False

        gui = window.get_gui()
        with gui.sub_window("Phong Parameters", 0.02, 0.02, 0.32, 0.24):
            ka[None] = gui.slider_float("Ka", ka[None], 0.0, 1.0)
            kd[None] = gui.slider_float("Kd", kd[None], 0.0, 1.0)
            ks[None] = gui.slider_float("Ks", ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float("Shininess", shininess[None], 1.0, 128.0)

        render()
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    main()
