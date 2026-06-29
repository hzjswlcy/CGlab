import taichi as ti
import numpy as np
from pathlib import Path
import os

# 使用 gpu 后端。关闭离线缓存，避免缓存锁导致窗口启动失败。
TAICHI_CACHE = Path(__file__).resolve().parents[1] / ".taichi_cache"
TAICHI_CACHE.mkdir(exist_ok=True)
ARCH = ti.gpu if os.environ.get("CG_USE_GPU") == "1" else ti.cpu
ti.init(arch=ARCH, offline_cache=False, offline_cache_file_path=str(TAICHI_CACHE))

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000 # 曲线采样点数量

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# GUI 绘制数据缓冲池
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

# --- 【性能优化核心 1】：新增一个用于存放曲线坐标的 GPU 缓冲区 ---
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

def de_casteljau(points, t):
    """用迭代方式计算 De Casteljau，避免控制点较多时递归过深。"""
    work = np.array(points, dtype=np.float32)
    count = len(work)
    for level in range(1, count):
        work[:count - level] = (1.0 - t) * work[:count - level] + t * work[1:count - level + 1]
    return work[0]

@ti.kernel
def clear_pixels():
    """并行清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

# --- 【性能优化核心 2】：将“点亮像素”的工作交给 GPU 并行执行 ---
@ti.kernel
def draw_curve_kernel(n: ti.i32):
    # 这个 for 循环在 kernel 中，Taichi 会自动将其在 GPU 上极速执行
    for i in range(n):
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([0.0, 1.0, 0.0])

def main():
    window = ti.ui.Window("Bezier Curve (60 FPS Restored)", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    curve_dirty = True
    
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB: 
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
                    curve_dirty = True
                    print(f"Added control point: {pos}")
            elif e.key == 'c': 
                control_points = []
                curve_dirty = True
                print("Canvas cleared.")
        
        clear_pixels()
        
        current_count = len(control_points)
        if current_count >= 2:
            if curve_dirty:
                curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
                for t_int in range(NUM_SEGMENTS + 1):
                    t = t_int / NUM_SEGMENTS
                    curve_points_np[t_int] = de_casteljau(control_points, t)
                curve_points_field.from_numpy(curve_points_np)
                curve_dirty = False
            
            draw_curve_kernel(NUM_SEGMENTS + 1)
                    
        canvas.set_image(pixels)
        
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))
            
            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))
        
        window.show()

if __name__ == '__main__':
    main()
