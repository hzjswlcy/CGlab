from __future__ import annotations

import inspect
import os
import pickle
import shutil
import warnings
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, message="In the future `np\.object`.*")
warnings.filterwarnings("ignore", category=FutureWarning, message="In the future `np\.str`.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="Please import `csc_matrix`.*")

WORK_DIR = Path(__file__).resolve().parent
DATA_DIR = WORK_DIR / "data"
DATA_PATH = DATA_DIR / "SMPL_NEUTRAL.pkl"
OUTPUT_DIR = WORK_DIR / "outputs"
OUTPUT_PATH = WORK_DIR / "lbs_summary.png"
MATPLOTLIB_CACHE = WORK_DIR.parent / ".matplotlib_cache"
MATPLOTLIB_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))


def patch_legacy_pickle_deps() -> None:
    if not hasattr(inspect, "getargspec"):
        ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

        def getargspec(func):
            spec = inspect.getfullargspec(func)
            return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

        inspect.getargspec = getargspec
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }
    for name, value in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def as_array(value) -> np.ndarray:
    if hasattr(value, "r"):
        value = value.r
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float64)


def load_smpl_data(path: Path = DATA_PATH) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}，请先下载 SMPL_NEUTRAL.pkl 到 work8/data/。")
    patch_legacy_pickle_deps()
    with path.open("rb") as file:
        raw = pickle.load(file, encoding="latin1")
    kintree = np.asarray(raw["kintree_table"], dtype=np.int64)
    parents = np.zeros(24, dtype=np.int64)
    id_to_col = {int(kintree[1, i]): i for i in range(kintree.shape[1])}
    parents[0] = -1
    for i in range(1, 24):
        parents[i] = id_to_col[int(kintree[0, i])]
    return {
        "v_template": as_array(raw["v_template"]),
        "shapedirs": as_array(raw["shapedirs"])[:, :, :10],
        "posedirs": as_array(raw["posedirs"]),
        "J_regressor": as_array(raw["J_regressor"]),
        "weights": as_array(raw["weights"]),
        "faces": np.asarray(raw["f"], dtype=np.int64),
        "parents": parents,
    }


def load_official_model():
    import smplx

    smpl_dir = DATA_DIR / "smpl"
    smpl_dir.mkdir(exist_ok=True)
    expected = smpl_dir / "SMPL_NEUTRAL.pkl"
    if not expected.exists():
        try:
            expected.symlink_to(DATA_PATH)
        except OSError:
            shutil.copy2(DATA_PATH, expected)
    return smplx.create(str(DATA_DIR), model_type="smpl", gender="neutral", ext="pkl", batch_size=1)


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-12:
        return np.eye(3)
    axis = axis_angle / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def batch_rodrigues(axis_angles: np.ndarray) -> np.ndarray:
    return np.stack([axis_angle_to_matrix(vec) for vec in axis_angles], axis=0)


def with_zeros(transform: np.ndarray) -> np.ndarray:
    return np.vstack([transform, np.array([[0.0, 0.0, 0.0, 1.0]])])


def pack_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return with_zeros(np.hstack([rotation, translation.reshape(3, 1)]))


def make_demo_parameters() -> tuple[np.ndarray, np.ndarray]:
    betas = np.zeros(10, dtype=np.float64)
    betas[:5] = [1.2, -0.8, 0.6, 0.35, -0.25]
    pose = np.zeros((24, 3), dtype=np.float64)
    pose[0] = [0.0, 0.0, 0.18]
    pose[1] = [0.0, 0.0, 0.35]
    pose[2] = [0.0, 0.0, -0.35]
    pose[12] = [0.18, 0.0, 0.0]
    pose[16] = [0.0, 0.0, -0.95]
    pose[17] = [0.0, 0.0, 0.95]
    pose[18] = [0.0, -0.65, 0.0]
    pose[19] = [0.0, 0.65, 0.0]
    return betas, pose


def shaped_vertices(v_template: np.ndarray, shapedirs: np.ndarray, betas: np.ndarray) -> np.ndarray:
    return v_template + np.tensordot(shapedirs, betas, axes=([2], [0]))


def pose_offsets(posedirs: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    pose_feature = (rotations[1:] - np.eye(3)).reshape(-1)
    flat = posedirs.reshape(-1, posedirs.shape[-1]) @ pose_feature
    return flat.reshape(-1, 3)


def global_rigid_transforms(rotations: np.ndarray, joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    transforms = [pack_transform(rotations[0], joints[0])]
    for i in range(1, len(parents)):
        relative_joint = joints[i] - joints[parents[i]]
        transforms.append(transforms[parents[i]] @ pack_transform(rotations[i], relative_joint))
    transforms = np.stack(transforms, axis=0)
    joint_homo = np.concatenate([joints, np.zeros((joints.shape[0], 1))], axis=1)
    init_bone = np.matmul(transforms, joint_homo[:, :, None])
    transforms[:, :, 3:4] -= init_bone
    return transforms


def manual_lbs(data: dict[str, np.ndarray], betas: np.ndarray, pose: np.ndarray) -> dict[str, np.ndarray]:
    v_template = data["v_template"]
    shapedirs = data["shapedirs"]
    posedirs = data["posedirs"]
    regressor = data["J_regressor"]
    weights = data["weights"]
    parents = data["parents"]

    v_shaped = shaped_vertices(v_template, shapedirs, betas)
    joints = regressor @ v_shaped
    rotations = batch_rodrigues(pose)
    offsets = pose_offsets(posedirs, rotations)
    v_posed = v_shaped + offsets
    transforms = global_rigid_transforms(rotations, joints, parents)
    blended = np.tensordot(weights, transforms, axes=([1], [0]))
    v_homo = np.concatenate([v_posed, np.ones((v_posed.shape[0], 1))], axis=1)
    v_lbs = np.einsum("vij,vj->vi", blended, v_homo)[:, :3]
    posed_joints = transforms[:, :3, 3]
    return {
        "v_shaped": v_shaped,
        "joints": joints,
        "rotations": rotations,
        "pose_offsets": offsets,
        "v_posed": v_posed,
        "transforms": transforms,
        "v_lbs": v_lbs,
        "posed_joints": posed_joints,
    }


def official_forward_vertices(betas: np.ndarray, pose: np.ndarray) -> np.ndarray:
    import torch

    model = load_official_model()
    model = model.to(dtype=torch.float64)
    with torch.no_grad():
        output = model(
            betas=torch.tensor(betas[None], dtype=torch.float64),
            global_orient=torch.tensor(pose[0:1], dtype=torch.float64),
            body_pose=torch.tensor(pose[1:].reshape(1, -1), dtype=torch.float64),
            return_verts=True,
        )
    return output.vertices[0].detach().cpu().numpy()


def normalize_for_plot(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(centered, axis=1))
    return centered / (scale + 1e-9)


def sample_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count)
    return np.linspace(0, count - 1, limit, dtype=np.int64)


def scatter_vertices(ax, vertices: np.ndarray, values: np.ndarray, title: str, limit: int, joints: np.ndarray | None = None) -> None:
    indices = sample_indices(len(vertices), limit)
    points = normalize_for_plot(vertices)
    shown = points[indices]
    colors = values[indices]
    sc = ax.scatter(shown[:, 0], shown[:, 1], c=colors, s=2.5, cmap="viridis", linewidths=0)
    if joints is not None:
        jp = normalize_for_plot(np.vstack([vertices, joints]))[-len(joints):]
        ax.scatter(jp[:, 0], jp[:, 1], c="red", s=14, marker="x")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)


def save_single_panel(path: Path, vertices: np.ndarray, values: np.ndarray, title: str, limit: int, joints: np.ndarray | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter_vertices(ax, vertices, values, title, limit, joints=joints)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_summary_figure(data: dict[str, np.ndarray], result: dict[str, np.ndarray], max_vertices_for_plot: int) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    weights = data["weights"]
    v_template = data["v_template"]
    v_shaped = result["v_shaped"]
    joints = result["joints"]
    offsets = result["pose_offsets"]
    v_posed = result["v_posed"]
    v_lbs = result["v_lbs"]
    posed_joints = result["posed_joints"]
    dominant = np.argmax(weights, axis=1)
    dominant_weight = np.max(weights, axis=1)
    shape_delta = np.linalg.norm(v_shaped - v_template, axis=1)
    pose_delta = np.linalg.norm(offsets, axis=1)
    final_height = v_lbs[:, 1]

    save_single_panel(OUTPUT_DIR / "stage_a_template_weights.png", v_template, weights[:, 16], "stage a: template + left shoulder weights", max_vertices_for_plot)
    save_single_panel(OUTPUT_DIR / "all_joint_weights.png", v_template, dominant + dominant_weight, "stage a: dominant joint distribution", max_vertices_for_plot)
    save_single_panel(OUTPUT_DIR / "stage_b_shaped_joints.png", v_shaped, shape_delta, "stage b: shaped mesh + regressed joints", max_vertices_for_plot, joints=joints)
    save_single_panel(OUTPUT_DIR / "stage_c_pose_offsets.png", v_posed, pose_delta, "stage c: pose corrective offsets", max_vertices_for_plot)
    save_single_panel(OUTPUT_DIR / "stage_d_lbs_result.png", v_lbs, final_height, "stage d: final skinned mesh + joints", max_vertices_for_plot, joints=posed_joints)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    scatter_vertices(axes[0, 0], v_template, weights[:, 16], "(a) template + weights", max_vertices_for_plot)
    scatter_vertices(axes[0, 1], v_shaped, shape_delta, "(b) shape + joints", max_vertices_for_plot, joints=joints)
    scatter_vertices(axes[1, 0], v_posed, pose_delta, "(c) pose offsets", max_vertices_for_plot)
    scatter_vertices(axes[1, 1], v_lbs, final_height, "(d) final skinned mesh", max_vertices_for_plot, joints=posed_joints)
    fig.suptitle("Lab 8 - SMPL Linear Blend Skinning")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_grid.png", dpi=180)
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)


def write_summary(metrics: dict[str, float | int]) -> None:
    lines = [
        "SMPL LBS summary",
        f"vertex_count: {metrics['vertex_count']}",
        f"face_count: {metrics['face_count']}",
        f"joint_count: {metrics['joint_count']}",
        f"betas_dim: {metrics['betas_dim']}",
        f"mean_abs_error: {metrics['mean_abs_error']:.10e}",
        f"max_abs_error: {metrics['max_abs_error']:.10e}",
        f"shape_delta_max: {metrics['shape_delta_max']:.10e}",
        f"pose_offset_max: {metrics['pose_offset_max']:.10e}",
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smpl_pipeline(max_vertices_for_plot: int = 2500) -> dict[str, float | int]:
    data = load_smpl_data()
    betas, pose = make_demo_parameters()
    result = manual_lbs(data, betas, pose)
    official = official_forward_vertices(betas, pose)
    abs_error = np.abs(result["v_lbs"] - official)
    metrics = {
        "vertex_count": int(data["v_template"].shape[0]),
        "joint_count": int(data["weights"].shape[1]),
        "face_count": int(data["faces"].shape[0]),
        "betas_dim": int(betas.shape[0]),
        "mean_abs_error": float(abs_error.mean()),
        "max_abs_error": float(abs_error.max()),
        "shape_delta_max": float(np.linalg.norm(result["v_shaped"] - data["v_template"], axis=1).max()),
        "pose_offset_max": float(np.linalg.norm(result["pose_offsets"], axis=1).max()),
    }
    make_summary_figure(data, result, max_vertices_for_plot)
    write_summary(metrics)
    return metrics


def main() -> None:
    metrics = run_smpl_pipeline()
    print("SMPL vertices:", metrics["vertex_count"])
    print("SMPL faces:", metrics["face_count"])
    print("SMPL joints:", metrics["joint_count"])
    print("betas dim:", metrics["betas_dim"])
    print("mean absolute error:", metrics["mean_abs_error"])
    print("max absolute error:", metrics["max_abs_error"])
    print("saved outputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
