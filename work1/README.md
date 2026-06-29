# 实验一：图形学开发工具

龙彦汐-202411081077-人工智能


本次实验主要完成图形学实验环境的搭建，并用 Taichi 实现一个可以实时交互的粒子群程序。工程采用 `src/Work0` 的组织方式，把参数配置、物理更新和窗口显示拆开管理：`config.py` 中集中保存粒子数量、窗口大小、阻尼和反弹系数，`physics.py` 负责 Taichi field 与并行更新，`main.py` 只处理窗口事件、鼠标位置和绘制。这样的结构比较适合后续实验复用，也方便单独调整物理参数。

程序中一共维护了 `pos` 和 `vel` 两个二维向量场，分别表示 6000 个粒子的位置和速度。初始化时每个粒子随机分布在归一化屏幕坐标中，速度清零：

```python
pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)
vel = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)

@ti.kernel
def init_particles():
    for i in range(NUM_PARTICLES):
        pos[i] = [ti.random(), ti.random()]
        vel[i] = [0.0, 0.0]
```

交互部分的核心是根据鼠标位置给每个粒子施加吸引力。每一帧先计算粒子指向鼠标的偏移向量，再按距离调整吸引强度，并叠加速度衰减和边界反弹：

```python
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
```

这里的 `MOUSE_DEAD_ZONE` 可以避免粒子非常接近鼠标时出现过强抖动，`DRAG_COEF` 用来模拟能量损失，使粒子不会无限加速。边界检测把越界坐标夹回 `[0, 1]`，同时让对应方向速度乘以负的反弹系数，因此粒子撞到窗口边缘后会有回弹效果。运行时默认使用 CPU 后端以保证兼容性，如果机器支持 GPU，也可以通过环境变量 `CG_USE_GPU=1` 切换后端。

## 运行方式

```bash
cd work1
uv run -m src.Work0.main
```

## 结果说明

窗口打开后可以看到大量浅蓝色粒子随机铺满画面。移动鼠标时，粒子会朝鼠标位置聚集；鼠标停留或快速移动时，粒子群会在吸引力、阻尼和边界反弹共同作用下形成连续的涌动效果。该程序验证了 Taichi kernel、GUI 窗口、鼠标输入和实时绘制流程可以正常工作。

![alt text](20260621100952_rec_.gif)
