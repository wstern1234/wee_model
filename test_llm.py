from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer from base model (not checkpoint folder)
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

# Load your fine-tuned model weights from checkpoint folder
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
model.eval()

print("Interactive LLM Chat (type 'exit' to quit)\n")

while True:
    prompt = input("You: ").strip()
    if prompt.lower() in ["exit", "quit"]:
        break

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    print(f"Bot: {response}\n")
