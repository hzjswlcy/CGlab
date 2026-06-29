from pathlib import Path
import os
import taichi as ti
from .config import PARTICLE_COLOR, PARTICLE_RADIUS, WINDOW_RES

TAICHI_CACHE = Path(__file__).resolve().parents[2] / ".taichi_cache"
TAICHI_CACHE.mkdir(exist_ok=True)
ARCH = ti.gpu if os.environ.get("CG_USE_GPU") == "1" else ti.cpu
ti.init(arch=ARCH, offline_cache=False, offline_cache_file_path=str(TAICHI_CACHE))

from .physics import init_particles, pos, update_particles


def run():
    init_particles()
    gui = ti.GUI("Lab 1 - Gravity Particle Swarm", res=WINDOW_RES)
    while gui.running:
        if gui.get_event(ti.GUI.PRESS) and gui.event.key == ti.GUI.ESCAPE:
            gui.running = False
        mouse_x, mouse_y = gui.get_cursor_pos()
        update_particles(mouse_x, mouse_y)
        gui.circles(pos.to_numpy(), color=PARTICLE_COLOR, radius=PARTICLE_RADIUS)
        gui.show()


if __name__ == "__main__":
    run()
