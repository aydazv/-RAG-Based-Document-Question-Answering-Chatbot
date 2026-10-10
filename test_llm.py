from transformers import pipeline

print("Loading language model...")

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device=-1
)

messages = [
    {
        "role": "system",
        "content": "به زبان فارسی و ساده پاسخ بده."
    },
    {
        "role": "user",
        "content": "یادگیری ماشین چیست؟"
    }
]

result = generator(
    messages,
    max_new_tokens=100,
    do_sample=False
)

answer = result[0]["generated_text"][-1]["content"]

print("\nپاسخ مدل:")
print(answer)