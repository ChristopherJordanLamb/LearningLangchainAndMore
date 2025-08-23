from gpt4all import GPT4All

# Force Vulkan backend explicitly
model = GPT4All(
    "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    device="gpu"
)

with model.chat_session():
    output = model.generate("what are the top 3 things to do in paris?", max_tokens=128)
    print(output)
    output = model.generate(f"explain why these three are the top things to do in paris? {output}")
    print(output)