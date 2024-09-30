import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")


@app.post("/predict/")
async def predict(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    outputs = model.generate(inputs.input_ids, max_length=100)
    print("outputs:", outputs)
    return outputs

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
