"""Focused tests for the optional AlphaFace integration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torchvision.transforms import v2

from app.processors.alphaface.model import IdentityFeedingBlock, OperationUnit
from app.processors.face_swappers import FaceSwappers
from app.processors.models_data import (
    arcface_mapping_model_dict,
    fp16_safe_models_list,
    models_list,
)
from app.processors.alphaface.profiles import (
    ALPHAFACE_DEFAULT_PROFILE,
    ALPHAFACE_FP16_MODEL_NAME,
    ALPHAFACE_PROFILE_OPTIONS,
    get_alphaface_profile,
)
from app.processors.utils import faceutil, platform_support
from app.processors.workers.frame_worker import FrameWorker
from app.processors.workers.frame_worker_pipeline import PipelineProcessor


def test_alphaface_is_optional_and_uses_shared_arcface() -> None:
    expected_files = {
        "AlphaFace": "alphaface_swapper.onnx",
        "AlphaFace Exact": "alphaface_swapper_optimized.onnx",
        "AlphaFace Fused": "alphaface_swapper_fused_norm.onnx",
        ALPHAFACE_FP16_MODEL_NAME: "alphaface_swapper_fused_norm.onnx",
    }
    alphaface_models = {
        item["model_name"]: item
        for item in models_list
        if item["model_name"] in expected_files
    }

    assert alphaface_models.keys() == expected_files.keys()
    for model_name, filename in expected_files.items():
        model = alphaface_models[model_name]
        assert model["optional"] is True
        assert model["url"].endswith(f"/alphaface-model-v1/{filename}")
    assert arcface_mapping_model_dict["AlphaFace"] == "Inswapper128ArcFace"
    assert ALPHAFACE_FP16_MODEL_NAME in fp16_safe_models_list
    assert "AlphaFace Fused" not in fp16_safe_models_list


def test_alphaface_profile_contract() -> None:
    assert ALPHAFACE_PROFILE_OPTIONS[0] == ALPHAFACE_DEFAULT_PROFILE
    assert get_alphaface_profile("Exact").fast_runtime is True
    assert get_alphaface_profile("Exact").lean_crops is True
    assert get_alphaface_profile("not-a-profile").model_name == "AlphaFace"


def test_alphaface_projection_is_matrix_multiply_then_l2_normalize() -> None:
    swapper = FaceSwappers.__new__(FaceSwappers)
    swapper._alphaface_emap = np.eye(512, dtype=np.float32) * 2.0
    embedding = np.arange(1, 513, dtype=np.float32)

    latent = swapper.calc_swapper_latent_alphaface(embedding)

    assert latent is not None
    expected = embedding.reshape(1, -1) / np.linalg.norm(embedding)
    np.testing.assert_allclose(latent, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(latent), 1.0, atol=1e-6)


def test_alphaface_identity_block_matches_official_singleton_adain() -> None:
    torch.manual_seed(7)
    block = IdentityFeedingBlock(output_dim=8, identity_dim=4)
    identity = torch.randn((1, 4))
    target = torch.randn((1, 4, 8, 8))

    projected = block.fc(identity).unsqueeze(2).unsqueeze(3)
    first, second = projected.chunk(2, dim=1)

    def official_adain(x: torch.Tensor) -> torch.Tensor:
        x_mean = torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])
        centered = (x.permute(2, 3, 0, 1) - x_mean).permute(2, 3, 0, 1)
        x_std = torch.sqrt(
            (torch.sum(centered**2, (2, 3)) + 2.3e-8)
            / (x.shape[2] * x.shape[3])
        )
        target_mean = torch.sum(target, (2, 3)) / (
            target.shape[2] * target.shape[3]
        )
        target_centered = (
            target.permute(2, 3, 0, 1) - target_mean
        ).permute(2, 3, 0, 1)
        target_std = torch.sqrt(
            (torch.sum(target_centered**2, (2, 3)) + 2.3e-8)
            / (target.shape[2] * target.shape[3])
        )
        normalized = (x.permute(2, 3, 0, 1) - x_mean) / x_std
        return (target_std * normalized + target_mean).permute(2, 3, 0, 1)

    expected = (
        (first + official_adain(first)) / 2.0,
        (second + official_adain(second)) / 2.0,
    )
    actual = block(identity, target)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_alphaface_fused_instance_norm_stays_close_to_manual_path() -> None:
    torch.manual_seed(11)
    manual = OperationUnit(4, 8, activate=False)
    fused = OperationUnit(4, 8, activate=False, fused_instance_norm=True)
    fused.load_state_dict(manual.state_dict(), strict=True)
    features = torch.randn((1, 4, 16, 16))
    identity = torch.randn((1, 512))

    manual_output = manual(features, identity)
    fused_output = fused(features, identity)

    torch.testing.assert_close(fused_output, manual_output, rtol=2e-5, atol=2e-5)


def test_alphaface_selects_256px_face_and_projected_latent() -> None:
    source = np.ones(512, dtype=np.float32)
    target = np.full(512, 2.0, dtype=np.float32)

    class Functions:
        @staticmethod
        def calc_swapper_latent_alphaface(embedding: np.ndarray) -> np.ndarray:
            value = 3.0 if embedding is source else 4.0
            return np.full((1, 512), value, dtype=np.float32)

    worker = SimpleNamespace(
        function_worker=Functions(),
        models_processor=SimpleNamespace(device=torch.device("cpu")),
    )
    pipeline = PipelineProcessor(worker)
    faces = tuple(torch.zeros((3, size, size)) for size in (512, 384, 256, 128))

    selected, dfm, dim, latent = pipeline.get_affined_face_dim_and_swapping_latents(
        faces, "AlphaFace", None, source, target, {}, False, None
    )

    assert selected is faces[2]
    assert dfm is None
    assert dim == 2
    assert torch.is_tensor(latent)
    assert torch.all(latent == 3.0)


def test_alphaface_lean_crop_path_skips_unused_resizes() -> None:
    worker = FrameWorker.__new__(FrameWorker)
    worker.t256 = v2.Resize((256, 256), antialias=True)
    worker.t384 = None
    worker.t128 = None
    image = torch.zeros((3, 512, 512), dtype=torch.uint8)
    transform = SimpleNamespace(params=np.eye(3, dtype=np.float32))

    face_512, face_384, face_256, face_128 = (
        worker.get_transformed_and_scaled_faces(
            transform, image, interp_mode="bilinear", only_256=True
        )
    )

    assert face_384.data_ptr() == face_512.data_ptr()
    assert face_128.data_ptr() == face_256.data_ptr()
    assert face_256.shape == (3, 256, 256)


def test_alphaface_uses_pose_aware_target_alignment() -> None:
    worker = FrameWorker.__new__(FrameWorker)
    profile_landmarks = faceutil.get_arcface_template(
        image_size=512, mode="arcfacemap"
    )[0]

    alphaface = worker.get_face_similarity_tform("AlphaFace", profile_landmarks)
    inswapper = worker.get_face_similarity_tform("Inswapper128", profile_landmarks)

    np.testing.assert_allclose(alphaface.params, np.eye(3), atol=3e-5)
    assert not np.allclose(inswapper.params, alphaface.params, atol=1e-3)


def test_alphaface_excludes_scale_popping_pitch_templates() -> None:
    worker = FrameWorker.__new__(FrameWorker)
    pitch_up_landmarks = faceutil.get_arcface_template(
        image_size=512, mode="arcfacemap"
    )[5]

    alphaface = worker.get_face_similarity_tform("AlphaFace", pitch_up_landmarks)

    assert not np.allclose(alphaface.params, np.eye(3), atol=1e-3)


def test_alphaface_inference_path_preserves_unit_range_contract() -> None:
    class Functions:
        @staticmethod
        def run_swapper_alphaface(image, embedding, output, model_name) -> None:
            assert image.shape == (1, 3, 256, 256)
            assert embedding.shape == (1, 512)
            assert model_name == "AlphaFace"
            output.fill_(0.25)

    worker = SimpleNamespace(
        function_worker=Functions(),
        models_processor=SimpleNamespace(device=torch.device("cpu"), device_type="cpu"),
        t512=lambda tensor: tensor,
        GHOSTFACE_MODELS=frozenset(),
    )
    pipeline = PipelineProcessor(worker)
    face = torch.full((256, 256, 3), 0.5)

    swap, previous = pipeline.get_swapped_and_prev_face(
        output=torch.empty_like(face),
        input_face_affined=face,
        original_face_512=torch.zeros((3, 512, 512)),
        latent=torch.ones((1, 512)),
        itex=1,
        dim=2,
        swapper_model="AlphaFace",
        dfm_model=None,
        parameters={"PreSwapSharpnessDecimalSlider": 1.0},
    )

    assert swap.shape == (3, 256, 256)
    assert torch.allclose(swap, torch.full_like(swap, 63.75))
    assert previous is face


def test_alphaface_nonfinite_output_falls_back_to_aligned_crop() -> None:
    class Functions:
        @staticmethod
        def run_swapper_alphaface(image, embedding, output, model_name) -> None:
            output.fill_(float("nan"))

    worker = SimpleNamespace(
        function_worker=Functions(),
        models_processor=SimpleNamespace(device=torch.device("cpu"), device_type="cpu"),
        t512=lambda tensor: tensor,
        GHOSTFACE_MODELS=frozenset(),
    )
    pipeline = PipelineProcessor(worker)
    face = torch.full((256, 256, 3), 0.5)

    swap, _ = pipeline.get_swapped_and_prev_face(
        output=torch.empty_like(face),
        input_face_affined=face,
        original_face_512=torch.zeros((3, 512, 512)),
        latent=torch.ones((1, 512)),
        itex=1,
        dim=2,
        swapper_model="AlphaFace",
        dfm_model=None,
        parameters={"PreSwapSharpnessDecimalSlider": 1.0},
    )

    assert torch.isfinite(swap).all()
    assert torch.allclose(swap, torch.full_like(swap, 127.5))


def test_alphaface_exact_profile_preserves_runtime_output(monkeypatch) -> None:
    model_names = []

    class Functions:
        @staticmethod
        def run_swapper_alphaface(image, embedding, output, model_name) -> None:
            model_names.append(model_name)
            output.fill_(0.25)

    worker = SimpleNamespace(
        function_worker=Functions(),
        models_processor=SimpleNamespace(device=torch.device("cpu"), device_type="cuda"),
        t512=lambda tensor: tensor,
        GHOSTFACE_MODELS=frozenset(),
    )
    pipeline = PipelineProcessor(worker)
    face = torch.full((256, 256, 3), 0.5)
    sync_calls = 0

    def count_sync() -> None:
        nonlocal sync_calls
        sync_calls += 1

    monkeypatch.setattr(platform_support, "blocking_stream_sync", count_sync)

    results = []
    for profile_name in ("Baseline", "Exact"):
        swap, _ = pipeline.get_swapped_and_prev_face(
            output=torch.empty_like(face),
            input_face_affined=face,
            original_face_512=torch.zeros((3, 512, 512)),
            latent=torch.ones((1, 512)),
            itex=1,
            dim=2,
            swapper_model="AlphaFace",
            dfm_model=None,
            parameters={
                "PreSwapSharpnessDecimalSlider": 1.0,
                "AlphaFacePerformanceProfileSelection": profile_name,
            },
        )
        results.append(swap)

    assert sync_calls == 1
    assert model_names == ["AlphaFace", "AlphaFace Exact"]
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


def test_alphaface_exact_profile_skips_unused_target_latent() -> None:
    source = np.ones(512, dtype=np.float32)
    target = np.full(512, 2.0, dtype=np.float32)

    class Functions:
        @staticmethod
        def calc_swapper_latent_alphaface(embedding: np.ndarray) -> np.ndarray:
            if embedding is target:
                raise AssertionError("target latent should not be projected")
            return np.ones((1, 512), dtype=np.float32)

    worker = SimpleNamespace(
        function_worker=Functions(),
        models_processor=SimpleNamespace(device=torch.device("cpu")),
    )
    pipeline = PipelineProcessor(worker)
    faces = tuple(torch.zeros((3, size, size)) for size in (512, 384, 256, 128))
    selected, _dfm, _dim, latent = (
        pipeline.get_affined_face_dim_and_swapping_latents(
            faces,
            "AlphaFace",
            None,
            source,
            target,
            {
                "FaceLikenessEnableToggle": False,
                "AlphaFacePerformanceProfileSelection": "Exact",
            },
            False,
            None,
        )
    )

    assert selected is faces[2]
    assert torch.is_tensor(latent)
