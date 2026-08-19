# AlphaFace performance tests

The experiment branch defaults to the original AlphaFace path. Set any of these
environment variables to `1` before starting VisoMaster:

| Variable | Change |
| --- | --- |
| `VISOMASTER_ALPHAFACE_FAST_RUNTIME` | Removes redundant buffer clearing, latent work, and CUDA synchronization. |
| `VISOMASTER_ALPHAFACE_SIMPLIFIED_GRAPH` | Loads the equivalent 793-node ONNX graph. |
| `VISOMASTER_ALPHAFACE_LEAN_CROPS` | Skips the unused 384px and 128px crops. |
| `VISOMASTER_ALPHAFACE_FUSED_NORM` | Loads the TensorRT-friendly normalization graph. |
| `VISOMASTER_ALPHAFACE_TRT_FP16` | Enables TensorRT FP16 for the fused-normalization graph. |

Example enabling the exact-output changes:

```powershell
$env:VISOMASTER_ALPHAFACE_FAST_RUNTIME = "1"
$env:VISOMASTER_ALPHAFACE_SIMPLIFIED_GRAPH = "1"
$env:VISOMASTER_ALPHAFACE_LEAN_CROPS = "1"
.venv\Scripts\python.exe main.py
```

Restart VisoMaster after changing a flag. Leave all flags unset for the
baseline.

The FP16 switch is ignored unless fused normalization is also enabled. Select
the TensorRT provider to test that pair.
