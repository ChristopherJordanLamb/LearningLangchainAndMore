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
hotel_descriptions = {
    "Hotel A": "A budget-friendly hotel with basic amenities.",
    "Hotel B": "A mid-range hotel with comfortable rooms and free breakfast.",
    "Hotel C": "A luxury hotel with a spa and fine dining.",
    "Hotel D": "A budget hotel located in the heart of the city.",
    "Hotel E": "A boutique hotel with unique decor and personalized service.",
    "Hotel F": "A family-friendly hotel with a pool and play area.",
    "Hotel G": "A modern hotel with stunning views of the city skyline.",
    "Hotel H": "An eco-friendly hotel with sustainable practices.",
    "Hotel I": "A traditional hotel with a rich history."
}
def getHotels(inp: str) -> str:
    print(f"[DEBUG] Tool received input: '{inp}'")
    city = inp.split(" in ")[-1] if " in " in inp else inp
    city = city.strip("'\" ")
    if city in hotels:
        hotel_list = {"hotels": hotels[city]} 
        print(f"[DEBUG] Found hotels: {hotel_list}")
        return hotel_list
    else:
        return "No hotels found for this city."
findHotels = Tool( 
    name="findHotels",
    description="Returns a list of hotels in the given city. Call with 'findHotels <city>'",
    func=getHotels)

def GetHotelDescription(hotelName: str) -> str:
    hotelName = hotelName.strip("'\" ")
    print(f"[DEBUG] Tool received input: '{hotelName}'")
    description = hotel_descriptions.get(hotelName, "No description available for this hotel.")
    print(f"[DEBUG] Description: {description}")
    return description
describeHotels = Tool(
    name="describeHotels",
    description="Returns a description of a given hotel name, you don't need the city, just the name. It can only handle one hotel at once. Call with 'describeHotels <hotel_name>'.",
    func=GetHotelDescription
)
wrapped_model = GPT4AllLangChain("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device="gpu")

# Simpler agent setup
agent = initialize_agent(
    [findHotels, describeHotels],
    wrapped_model,
    agent_type="zero-shot-react-description",
    verbose=True,
    max_iterations=8,
    agent_kwargs={
        "prefix": "You are a travel assistant. Use the tools when needed, otherwise answer directly."
    }
)

print("=== TESTING ===")
hotel_options = agent.run("please get me the name of some hotels in Paris")
print(hotel_options)
result = agent.run(f"give a short description of each of these hotel names: {str(hotel_options)}")
print(f"\n=== FINAL RESULT: {result} ===")

# Also test the tool directly
direct_result = describeHotels("Hotel A")
print(f"Direct tool result: {direct_result}")