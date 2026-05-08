# Humor Detection with XAI

A full-stack NLP humor classification application using RoBERTa, Integrated Gradients explainable AI, FastAPI, React, and Docker.
---

## Overview

This project fine-tunes a RoBERTa transformer model for binary humor classification. In addition to standard text classification, the system incorporates explainable AI (XAI) using Integrated Gradients from the Captum library to visualize token-level attribution scores and better understand model behavior.

The application was deployed using a modern full-stack architecture consisting of:
- FastAPI backend
- React frontend
- Docker containerization
- Recharts visualization library
- HuggingFace Transformers
- PyTorch

---

## Features

- Humor vs Non-Humor classification
- Confidence score prediction
- Integrated Gradients token attribution visualization
- Interactive React frontend
- Dockerized backend deployment
- REST API endpoints using FastAPI

---

## Model Architecture

- Base Model: `roberta-base`
- Framework: PyTorch + HuggingFace Transformers
- Explainability: Captum Integrated Gradients

---


# Backend Setup

## Build Docker Image

```bash
cd backend
docker build -t humor-api .
```

## Run Backend Container

```bash
docker run -p 8000:8000 humor-api
```

Backend API:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Frontend Setup

## Install Frontend Dependencies

```bash
cd frontend
npm install
```

## Run React Frontend

```bash
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

---

## Example Prediction

Input:

```text
I used to be a banker but I lost interest.
```

Output:

```json
{
  "prediction": "Humor",
  "confidence": 0.999
}
```

---

## Explainable AI

Integrated Gradients is used to generate token-level attribution scores showing how individual tokens contribute to the model prediction.

Positive attribution:
- pushes prediction toward humor

Negative attribution:
- pushes prediction away from humor

---

## Technologies Used

- Python
- FastAPI
- React
- Docker
- PyTorch
- HuggingFace Transformers
- Captum
- Recharts

---

## Author

Brendan Lauterborn