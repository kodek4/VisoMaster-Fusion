# AlphaFace performance tests

Choose a mode from **AlphaFace Optimization** in the Face Swap panel.

| Mode | Change |
| --- | --- |
| Baseline | Original model and runtime path. |
| Exact | Equivalent graph with the safe runtime and crop changes. Output matches Baseline. |
| Fused | Uses standard ONNX instance normalization. This can change pixels slightly. |
| Fused FP16 | Runs the fused model with TensorRT FP16. Select the TensorRT provider for this mode. |

Changing modes reloads AlphaFace. The first TensorRT run may pause while its
engine is built.

## Input alignment

**AlphaFace Input Alignment** controls only the crop sent to the model.

| Mode | Change |
| --- | --- |
| Baseline | Current VisoMaster crop. |
| Training Matched | Matches the vertical placement in the official AlphaFace samples. |
| Training Matched Direct | Uses the matched placement and warps the 256px input directly from the frame. |

**AlphaFace Identity Strength** adjusts the identity-conditioned residual in
the final three CAII blocks. `1.00` matches the released model. Values around
`1.15` to `1.25` can strengthen identity, while larger values may soften detail.
