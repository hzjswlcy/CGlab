# 实验七：质点弹簧模型

龙彦汐-202411081077-人工智能


本次实验实现了一个三维布料质点弹簧系统。布料被离散成 `20 × 20` 的质点网格，左侧两个角点固定，其余质点在重力、结构弹簧力和阻尼力作用下运动。初始化分成位置初始化和边索引初始化两个 kernel：前者设置质点初始位置、速度和受力，后者生成横向和纵向结构弹簧边，用于后续线框绘制。

质点索引通过二维网格坐标映射到一维数组，固定点判断也基于网格坐标完成：

```python
def idx(i, j):
    return i * N + j

@ti.func
def is_pinned(i, j):
    return i == 0 and (j == 0 or j == N - 1)
```

受力计算主要围绕胡克定律展开。相邻质点之间的弹簧力由当前长度与静止长度的差决定，阻尼项取相对速度在弹簧方向上的投影，用来削弱沿弹簧方向的振荡：

```python
@ti.func
def spring_force(a, b, rest_length):
    delta = pos[b] - pos[a]
    length = delta.norm() + 1e-6
    direction = delta / length
    relative_speed = (vel[b] - vel[a]).dot(direction)
    spring = spring_k[None] * (length - rest_length) * direction
    damping = damping_coef[None] * relative_speed * direction
    return spring + damping
```

积分部分实现了三种模式：显式欧拉、半隐式欧拉和带阻尼的迭代近似。显式欧拉先更新位置再更新速度，运动更活跃，也更容易体现不稳定趋势；半隐式欧拉先更新速度再更新位置，整体表现更稳定；阻尼迭代模式会混合预测速度和旧速度，并额外施加衰减，因此布料下落更慢、抖动更少。为了避免参数调大后速度过快，程序在每一步都通过 `clamp_velocity()` 限制最大速度，并在触碰地板时进行简单反弹处理。

窗口中使用 `scene.particles()` 绘制质点，用 `scene.lines()` 绘制弹簧边。右下角控制面板可以切换三种求解器，也可以调整弹簧强度、阻尼和最大速度。按 `0/1/2` 切换求解器时会自动重置布料，便于从同一初始状态比较不同积分方式的效果。

## 运行方式

```bash
cd work7
uv run python main.py
```

## 结果说明

窗口打开后，布料会从水平状态开始下垂，两个固定角保持在原位。按 `0` 使用显式欧拉时，下落和摆动更明显；按 `1` 使用半隐式欧拉时，运动较均衡；按 `2` 使用阻尼迭代时，布料明显变慢，抖动也更少。录屏展示了布料在不同求解器和参数下的动态变化。

![质点弹簧录屏](20260621103453_rec_.gif)
