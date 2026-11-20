import ollama

response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": "Introduce yourself as a supportive student assistant."
        }
    ]
)

print(response["message"]["content"])