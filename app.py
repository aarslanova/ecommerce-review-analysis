import streamlit as st
import torch
from torch import nn
from transformers import DistilBertTokenizer


def preprocess_input(
    input_text: str, tokenizer: DistilBertTokenizer, max_len: int = 512
):
    encoding = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    return (encoding["input_ids"].flatten(), encoding["attention_mask"].flatten())


def model_output(
    input_text: str,
    tokenizer: DistilBertTokenizer,
    model: nn.Module,
    device: torch.device,
    max_len: int = 512,
):
    """
    Generates an output.

    params:
    input_text (str): Text to process.
    """
    with torch.inference_mode():
        inputs = preprocess_input(input_text, tokenizer, max_len)
        inputs = [i.to(device) for i in inputs]
        prediction = model(inputs)

    return f"This text has a sentiment of {round(prediction.item(), 3)}"


def main():
    BERT_MODEL_NAME = "distilbert-base-uncased"
    DEVICE = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    distillbert_tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = torch.load("model.pth", DEVICE)

    st.title("Predict Book Score Based On Customer's Review")

    user_input = st.text_area("Input a review", "Paste a review here...")

    if st.button("Predict a book score based on customer's review"):
        output = model_output(user_input, distillbert_tokenizer, model, DEVICE)
        st.text_area("Output", output, height=300)


if __name__ == "__main__":
    main()
