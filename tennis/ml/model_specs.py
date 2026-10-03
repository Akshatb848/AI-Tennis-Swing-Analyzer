"""
ML Model Specifications — Architecture definitions for all CoreML models.

All models are optimized for Apple Neural Engine (ANE) deployment:
- BallNet: YOLOv8-nano ball detector + Kalman tracker
- PlayerNet: MobileNetV3 person detector + MoveNet pose
- CourtNet: Lightweight line segmentation + RANSAC homography
- ShotNet: 1D-CNN temporal shot classifier
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ModelTarget(str, Enum):
    COREML = "coreml"
    ONNX = "onnx"
    TFLITE = "tflite"
    PYTORCH = "pytorch"


@dataclass
class ModelSpec:
    """Specification for a single ML model."""
    name: str
    version: str
    task: str
    input_shape: list[int]           # [batch, channels, height, width]
    output_shape: list[int]
    backbone: str
    num_parameters: int
    flops_millions: float
    target_latency_ms: float
    target_device: str
    training_dataset: str
    training_strategy: str
    export_format: ModelTarget
    input_preprocessing: dict = field(default_factory=dict)
    output_postprocessing: str = ""
    notes: str = ""


# ── BallNet ──────────────────────────────────────────────────────────────────

BALLNET_SPEC = ModelSpec(
    name="BallNet",
    version="1.0",
    task="Ball Detection & Tracking",
    input_shape=[1, 3, 640, 640],
    output_shape=[1, 5, 8400],       # [x, y, w, h, conf] × 8400 anchors
    backbone="YOLOv8-nano",
    num_parameters=3_200_000,
    flops_millions=8.7,
    target_latency_ms=8.0,
    target_device="Apple Neural Engine (A12+)",
    training_dataset="Custom tennis ball dataset (50K frames, 12 courts, varied lighting)",
    training_strategy=(
        "1. Pre-train YOLOv8n on COCO\n"
        "2. Fine-tune on tennis ball dataset with augmentation:\n"
        "   - Motion blur (sigma 1-5px)\n"
        "   - Brightness/contrast variation\n"
        "   - Court surface color jitter\n"
        "3. Quantize to INT8 for ANE\n"
        "4. Post-training: Kalman filter + optical flow fusion for tracking"
    ),
    export_format=ModelTarget.COREML,
    input_preprocessing={
        "resize": [640, 640],
        "normalize": "0-1 range",
        "color_space": "RGB",
    },
    output_postprocessing="NMS (IoU=0.45, conf=0.25) → Kalman update → trajectory",
    notes="Handles occlusion via Kalman prediction. Optical flow provides velocity.",
)

# ── PlayerNet ────────────────────────────────────────────────────────────────

PLAYERNET_SPEC = ModelSpec(
    name="PlayerNet",
    version="1.0",
    task="Player Detection + Pose Estimation",
    input_shape=[1, 3, 256, 256],
    output_shape=[1, 17, 3],         # 17 COCO keypoints × (x, y, conf)
    backbone="MobileNetV3-Small + MoveNet Thunder",
    num_parameters=5_400_000,
    flops_millions=12.3,
    target_latency_ms=12.0,
    target_device="Apple Neural Engine (A12+)",
    training_dataset="COCO Keypoints + Custom tennis pose (20K annotated frames)",
    training_strategy=(
        "1. Use pre-trained MoveNet Thunder as backbone\n"
        "2. Fine-tune on tennis-specific poses:\n"
        "   - Serve motion, forehand/backhand follow-through\n"
        "   - Ready position, split step\n"
        "3. Add person detector head (MobileNetV3-Small)\n"
        "4. Multi-task training: detect + pose simultaneously\n"
        "5. Player ID tracking via DeepSORT embeddings"
    ),
    export_format=ModelTarget.COREML,
    input_preprocessing={
        "resize": [256, 256],
        "normalize": "ImageNet mean/std",
        "color_space": "RGB",
    },
    output_postprocessing="Keypoint confidence filtering → skeleton construction → pose sequence buffer",
    notes="17 COCO keypoints. Player ID maintained via appearance embedding + Kalman.",
)

# ── CourtNet ─────────────────────────────────────────────────────────────────

COURTNET_SPEC = ModelSpec(
    name="CourtNet",
    version="1.0",
    task="Court Line Detection + Homography",
    input_shape=[1, 3, 320, 320],
    output_shape=[1, 14, 2],         # 14 court keypoints × (x, y)
    backbone="ResNet18-Lite",
    num_parameters=2_100_000,
    flops_millions=5.2,
    target_latency_ms=6.0,
    target_device="Apple Neural Engine (A12+)",
    training_dataset="Synthetic + real court images (30K, all surfaces, day/night)",
    training_strategy=(
        "1. Generate synthetic courts (all surfaces, lighting, camera angles)\n"
        "2. Augment with real court images from YouTube matches\n"
        "3. Train keypoint regression for 14 court landmarks\n"
        "4. RANSAC homography estimation from keypoints → court plane\n"
        "5. Continuous recalibration filter (EMA) for camera shake"
    ),
    export_format=ModelTarget.COREML,
    input_preprocessing={
        "resize": [320, 320],
        "normalize": "0-1 range",
        "color_space": "RGB",
    },
    output_postprocessing="RANSAC homography → EMA smoothing → coordinate transform matrix",
    notes="14 keypoints: 4 corners, 4 service box corners, 4 baseline/sideline mids, center mark (×2).",
)

# ── ShotNet ──────────────────────────────────────────────────────────────────

SHOTNET_SPEC = ModelSpec(
    name="ShotNet",
    version="1.0",
    task="Shot Type Classification",
    input_shape=[1, 30, 51],         # 30 frames × 17 keypoints × 3 (x,y,conf)
    output_shape=[1, 12],            # 12 shot types
    backbone="1D-CNN + LSTM",
    num_parameters=890_000,
    flops_millions=1.8,
    target_latency_ms=3.0,
    target_device="Apple Neural Engine (A12+)",
    training_dataset="Tennis shot sequences (15K labeled swings, 12 shot types)",
    training_strategy=(
        "1. Extract pose sequences from training videos via PlayerNet\n"
        "2. Label shot types: serve, FH, BH, volley FH/BH, slice, lob, drop, overhead\n"
        "3. Train 1D temporal CNN on 30-frame windows\n"
        "4. Add LSTM head for temporal dependencies\n"
        "5. Data aug: speed variation, mirror for handedness"
    ),
    export_format=ModelTarget.COREML,
    input_preprocessing={
        "window_size": 30,
        "normalize": "skeleton-relative coordinates",
        "center": "hip midpoint",
    },
    output_postprocessing="Softmax → top-1 class with confidence threshold (0.6)",
    notes="Classifies from pose sequence. Works on 30-frame sliding window (~1 second at 30fps).",
)

# ── Model Registry ──────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "ballnet": BALLNET_SPEC,
    "playernet": PLAYERNET_SPEC,
    "courtnet": COURTNET_SPEC,
    "shotnet": SHOTNET_SPEC,
}


def get_model_spec(name: str) -> ModelSpec:
    """Get model specification by name."""
    spec = MODEL_REGISTRY.get(name.lower())
    if not spec:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return spec


def get_total_parameters() -> int:
    """Get total parameters across all models."""
    return sum(s.num_parameters for s in MODEL_REGISTRY.values())


def get_total_latency_budget() -> float:
    """Get total latency budget per frame (ms)."""
    return sum(s.target_latency_ms for s in MODEL_REGISTRY.values())


def print_model_summary():
    """Print a summary of all model specifications."""
    total_params = get_total_parameters()
    total_latency = get_total_latency_budget()
    print(f"\n{'='*70}")
    print(f"TennisIQ ML Model Stack — Total: {total_params/1e6:.1f}M params, {total_latency:.0f}ms/frame")
    print(f"{'='*70}")
    for name, spec in MODEL_REGISTRY.items():
        print(f"\n  {spec.name} v{spec.version}")
        print(f"    Task: {spec.task}")
        print(f"    Backbone: {spec.backbone}")
        print(f"    Parameters: {spec.num_parameters/1e6:.1f}M")
        print(f"    Target latency: {spec.target_latency_ms}ms")
        print(f"    Input: {spec.input_shape}")
        print(f"    Output: {spec.output_shape}")
    print(f"\n  Combined latency: {total_latency:.0f}ms/frame (target <50ms)")
    print(f"{'='*70}\n")
