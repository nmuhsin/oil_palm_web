---
title: FFB Ripeness Detector
emoji: 🌴
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Oil Palm FFB Ripeness Detector

A Flask web app that classifies fresh fruit bunch (FFB) images as **Overripe**,
**Ripe**, or **Unripe** using a fine-tuned MobileNetV2 model (93.24% test accuracy).

Upload a photo on the Detect page to get a prediction with confidence scores.
Images below an 85% confidence threshold are flagged as likely not oil palm FFB.

## Local development

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:7860
