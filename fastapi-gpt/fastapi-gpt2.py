import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")


class TextGenerationRequest(BaseModel):
    text: str
    max_length: int = 50
    num_return_sequences: int = 1


@app.post("/generate/")
async def generate_text(request: TextGenerationRequest):
    try:
        # Tokenize the input text
        inputs = tokenizer(request.text, return_tensors="pt")

        # Generate output text
        outputs = model.generate(
            inputs.input_ids,
            max_length=request.max_length,
            num_return_sequences=request.num_return_sequences,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )

        # Decode the generated text
        generated_texts = [
            tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]

        return {"generated_texts": generated_texts}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# For testing locally, run this code directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
