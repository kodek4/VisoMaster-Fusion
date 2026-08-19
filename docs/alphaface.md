# AlphaFace

AlphaFace is available as an optional 256 x 256 face swapper. It uses
VisoMaster's existing W600K ArcFace encoder with the projection supplied by the
AlphaFace authors.

The model downloader installs the ONNX model in
`model_assets/alphaface/alphaface_swapper.onnx`.

To export the model yourself, download `alphaface_demo.pt` from the
[official repository](https://github.com/andrewyu90/Alphaface_Official) and run:

```powershell
.venv\Scripts\python.exe app\tools\export_alphaface_onnx.py C:\path\to\alphaface_demo.pt
```

Restart VisoMaster after installing the model, then select **AlphaFace** from
the Swapper Model list. AlphaFace runs in FP32 and is not included in the
TensorRT FP16 allowlist.

The inference code is based on the authors' MIT-licensed
[implementation](https://github.com/andrewyu90/Alphaface_Official) at commit
`d41fbd4974ed3a68d9a48b79019f9be297726c30`. See the
[AlphaFace paper](https://arxiv.org/abs/2601.16429) for model details.
