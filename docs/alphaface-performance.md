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
