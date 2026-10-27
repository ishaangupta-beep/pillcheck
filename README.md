---
title: PillCheck
emoji: 💊
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Photo-verified medication adherence with OpenCV 5 + agent
---

# PillCheck

**Be there for every dose, even from miles away.**

An elderly patient photographs their blister strip after each dose. OpenCV 5 (ALIKED + LightGlue, MAGSAC homography, per-pocket comparison, DNN text models for the expiry date) works out which tablets were taken, and an agent records the dose, reminds, asks for a retake, alerts the caregiver or asks a human to decide.

Built for the OpenCV AI Competition 2026 (powered by AWS).

- **Technical report, architecture, agent workflow, evaluation:** [REPORT.md](REPORT.md)
- **Run locally:** see REPORT.md, section 7
- **AWS deployment:** `backend/infra/template.yaml` + `backend/infra/deploy.sh` (Lambda, DynamoDB, S3, SNS, EventBridge, Bedrock, CloudWatch)

Demo photos are real strip photos from the public MedCheck dataset (github.com/MahiiBamba/mini_project_medcheck).
