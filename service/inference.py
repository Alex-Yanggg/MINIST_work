from __future__ import annotations

from typing import Dict

import torch


def predict(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device) -> Dict:
    """执行一次前向推理并返回预测细节。"""
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0)

    all_probs = {str(i): float(probs[i]) for i in range(10)}
    pred_idx = int(probs.argmax().item())
    confidence = float(probs[pred_idx].item() * 100)

    top3_probs, top3_indices = torch.topk(probs, k=min(3, len(probs)))
    top_predictions = [
        {
            "digit": int(idx.item()),
            "probability": float(prob.item()),
            "confidence": f"{prob.item() * 100:.2f}%",
        }
        for prob, idx in zip(top3_probs, top3_indices)
    ]

    return {
        "prediction": pred_idx,
        "confidence": round(confidence, 2),
        "confidence_percent": f"{confidence:.2f}%",
        "top_predictions": top_predictions,
        "all_probabilities": all_probs,
    }

