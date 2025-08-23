from langchain.agents import Tool
from gpt4all import GPT4All
from langchain.agents import initialize_agent
from langchain.llms.base import LLM
from pydantic import PrivateAttr
from langchain.schema import Generation, LLMResult
import re

class GPT4AllLangChain(LLM):
    _model: GPT4All = PrivateAttr()

    def __init__(self, model_path, device="gpu", **kwargs):
        super().__init__(**kwargs)
        self._model = GPT4All(model_path, device=device)

    @property
    def _llm_type(self):
        return "gpt4all"

    def _call(self, prompt: str, stop=None):
        with self._model.chat_session():
            response = self._model.generate(prompt, max_tokens=200)  # Shorter responses
            
            if stop:
                for stop_seq in stop:
                    if stop_seq in response:
                        response = response.split(stop_seq)[0]
                        break
            
            return response
    
    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)
hotels = {
    "Paris": ["Hotel A", "Hotel B", "Hotel C"],
    "New York": ["Hotel D", "Hotel E", "Hotel F"],
    "Tokyo": ["Hotel G", "Hotel H", "Hotel I"]
}

def getHotels(inp: str) -> str:
    print(f"[DEBUG] Tool received input: '{inp}'")
    city = inp.split(" in ")[-1] if " in " in inp else inp
    print(f"[DEBUG] Extracted city: {city}")

    if city in hotels:
        hotel_list = hotels[city]
        print(f"[DEBUG] Found hotels: {hotel_list}")
        return ", ".join(hotel_list)
    else:
        return "No hotels found for this city."
findHotels = Tool( 
    name="findHotels",
    description="Returns a list of hotels in the specified city. Call with 'findHotels <city>'",
    func=getHotels)

wrapped_model = GPT4AllLangChain("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device="gpu")

# Simpler agent setup
agent = initialize_agent(
    [findHotels], 
    wrapped_model, 
    agent_type="conversational-react-description",
    verbose=True,
    max_iterations=2,
    agent_kwargs={
        "prefix": "You are a calculator assistant. If you have a tool fits a problem, use it. otherwise, Make your own choice. Tools are called with 'Toolname' not with '[Toolname]'. once you decide on a result, return it immediately. "
    }
)

print("=== TESTING ===")
result = agent.run("please get me some hotels in Paris")
print(f"\n=== FINAL RESULT: {result} ===")

# Also test the tool directly
direct_result = findHotels("findHotels in Paris")
print(f"Direct tool result: {direct_result}")