"""Focused tests for the optional AlphaFace integration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from app.processors.face_swappers import FaceSwappers
from app.processors.models_data import (
    arcface_mapping_model_dict,
    fp16_safe_models_list,
    models_list,
)
from app.processors.utils import faceutil
from app.processors.workers.frame_worker import FrameWorker
from app.processors.workers.frame_worker_pipeline import PipelineProcessor


def test_alphaface_is_optional_and_uses_shared_arcface() -> None:
    model = next(item for item in models_list if item["model_name"] == "AlphaFace")

    assert model["optional"] is True
    assert model["url"].endswith("/alphaface-model-v1/alphaface_swapper.onnx")
    assert arcface_mapping_model_dict["AlphaFace"] == "Inswapper128ArcFace"
    assert "AlphaFace" not in fp16_safe_models_list


def test_alphaface_projection_is_matrix_multiply_then_l2_normalize() -> None:
    swapper = FaceSwappers.__new__(FaceSwappers)
    swapper._alphaface_emap = np.eye(512, dtype=np.float32) * 2.0
    embedding = np.arange(1, 513, dtype=np.float32)

    latent = swapper.calc_swapper_latent_alphaface(embedding)

    assert latent is not None
    expected = embedding.reshape(1, -1) / np.linalg.norm(embedding)
    np.testing.assert_allclose(latent, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(latent), 1.0, atol=1e-6)


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
        def run_swapper_alphaface(image, embedding, output) -> None:
            assert image.shape == (1, 3, 256, 256)
            assert embedding.shape == (1, 512)
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
        def run_swapper_alphaface(image, embedding, output) -> None:
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
