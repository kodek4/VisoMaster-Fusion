# AlphaFace

AlphaFace is available as an optional 256 x 256 face swapper. It uses
VisoMaster's existing W600K ArcFace encoder with the projection supplied by the
AlphaFace authors.

Download `alphaface_demo.pt` from the
[official repository](https://github.com/andrewyu90/Alphaface_Official), then
export it from the VisoMaster repository root:

```powershell
.venv\Scripts\python.exe app\tools\export_alphaface_onnx.py C:\path\to\alphaface_demo.pt
```

The exporter writes
`model_assets/alphaface/alphaface_swapper.onnx`. Restart VisoMaster and select
**AlphaFace** from the Swapper Model list.

The ONNX model is not downloaded automatically. AlphaFace runs in FP32 and is
not included in the TensorRT FP16 allowlist.

The inference code is based on the authors' MIT-licensed
[implementation](https://github.com/andrewyu90/Alphaface_Official) at commit
`d41fbd4974ed3a68d9a48b79019f9be297726c30`. See the
[AlphaFace paper](https://arxiv.org/abs/2601.16429) for model details.
