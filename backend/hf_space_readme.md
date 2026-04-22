---
title: AMSI-Net LULC Recognition
emoji: 🛰️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# AMSI-Net Multi-label LULC Recognition

This is a professional backend for Land Use & Land Cover (LULC) recognition using the AMSI-Net architecture.

## Features
- **Dataset**: MLRSNet (60 classes)
- **Architecture**: AMSI-Net (ResNet50 + Dynamic FPN + GAT)
- **Explainability**: GradCAM, GradCAM++, Saliency, LIME
- **API**: FastAPI with automatic Swagger docs

## API Documentation
The API documentation is available at `/docs` (Swagger UI).

## Deployment
This Space is built using Docker. The backend runs on port 8000 (mapped to HF default).
