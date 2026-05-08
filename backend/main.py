### Brendan Lauterborn - Humor Detection with XAI ###

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware
from captum.attr import LayerIntegratedGradients


app = FastAPI(title="Humor Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "humor_model.pt"
MODEL_NAME = "roberta-base"
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Humor Detection API is running...",
        "endpoint": "/predict"
    }


@app.post("/predict")
def predict_humor(input_data: TextInput):
    text = input_data.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    prediction = "Humor" if pred_class == 1 else "Not Humor"

    return {
        "text": text,
        "prediction": prediction,
        "confidence": round(probs[0][pred_class].item(), 4),
        "probabilities": {
            "Not Humor": round(probs[0][0].item(), 4),
            "Humor": round(probs[0][1].item(), 4)
        }
    }

@app.post("/explain")
def explain_humor(input_data: TextInput):
    text = input_data.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    def forward_pass(input_ids, attention_mask):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

    lig = LayerIntegratedGradients(
        forward_pass,
        model.roberta.embeddings
    )

    baseline_ids = torch.full_like(
        input_ids,
        tokenizer.pad_token_id
    )

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=pred_class,
        return_convergence_delta=True
    )

    scores = attributions.sum(dim=-1).squeeze(0)

    norm = torch.norm(scores)
    if norm != 0:
        scores = scores / norm

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    valid_len = attention_mask.squeeze(0).sum().item()
    tokens = tokens[:valid_len]
    scores = scores[:valid_len].detach().cpu().tolist()

    filtered_tokens = []
    filtered_scores = []

    for token, score in zip(tokens, scores):
        if token not in ["<s>", "</s>", "<pad>"]:
            filtered_tokens.append(token)
            filtered_scores.append(round(score, 6))

    return {
        "text": text,
        "prediction": "Humor" if pred_class == 1 else "Not Humor",
        "confidence": round(probs[0][pred_class].item(), 4),
        "convergence_delta": round(delta.item(), 6),
        "tokens": filtered_tokens,
        "scores": filtered_scores
    }