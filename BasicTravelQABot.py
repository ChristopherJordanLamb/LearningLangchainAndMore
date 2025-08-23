from gpt4all import GPT4All

model = GPT4All(
    "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    device="gpu"
)
uinp = input("where would you like to travel? ")
with model.chat_session():
    hotels = model.generate(f"What are the three best budget hotels at {uinp}?", max_tokens=128)
    print(hotels)
    print("*******************")
    result = model.generate(f"give a short description of each of these hotels:{hotels}")
    print(result)