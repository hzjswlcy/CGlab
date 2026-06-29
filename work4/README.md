# 实验四：Phong 光照模型

龙彦汐-202411081077-人工智能


本次实验用光线投射的方式渲染一个由隐式几何体组成的三维场景，并在命中点上计算 Phong 光照。场景中包含一个红色球体和一个蓝色圆锥体，程序没有读取外部模型，而是在 Taichi kernel 中直接计算射线与球体、圆锥的交点。每个像素只保留最近的有效交点，相当于在光线方向上完成了深度测试，因此两个物体的遮挡关系可以正确显示。

射线从相机原点出发，方向由像素坐标换算得到。球体交点通过二次方程求解，圆锥交点则根据圆锥隐式方程求解，并额外限制命中点的高度范围：

```python
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
```

Phong 着色由环境光、漫反射和镜面高光三部分组成。实验中使用半程向量计算高光，这样公式简洁，也能得到稳定的高光效果：

```python
@ti.func
def phong(point, normal, base_color):
    light_dir = normalize(light_pos - point)
    view_dir = normalize(ti.Vector([0.0, 0.0, 0.0]) - point)
    half_vec = normalize(light_dir + view_dir)
    ambient = ka[None] * base_color
    diffuse = kd[None] * ti.max(0.0, normal.dot(light_dir)) * base_color * light_color
    specular = ks[None] * ti.pow(ti.max(0.0, normal.dot(half_vec)), shininess[None]) * light_color
    return ti.min(ambient + diffuse + specular, 1.0)
```

窗口左下角提供 `Ka`、`Kd`、`Ks` 和 `Shininess` 四个滑动条。`Ka` 控制暗部基础亮度，`Kd` 控制物体朝向光源时的漫反射强度，`Ks` 和 `Shininess` 分别影响高光亮度和集中程度。参数调整后每帧都会重新渲染，因此可以直接观察光照模型各项参数对画面的影响。

## 运行方式

```bash
cd work4
uv run python main.py
```

## 结果说明

窗口中可以看到红色球体和蓝色圆锥体，物体表面具有明显的明暗过渡和镜面高光。增大 `Ka` 时暗部整体变亮，增大 `Kd` 时受光面更亮，增大 `Ks` 或 `Shininess` 时高光会更明显或更集中。截图和录屏展示了默认参数下的渲染结果以及滑动条交互效果。

![alt text](image.png)

![alt text](20260621100649_rec_.gif)
