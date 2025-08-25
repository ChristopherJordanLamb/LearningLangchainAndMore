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

def GetSumPlus1(inp: str) -> str:
    print(f"[DEBUG] Tool received input: '{inp}'")
    numbers = re.findall(r'-?\d+\.?\d*', inp)
    print(f"[DEBUG] Extracted numbers: {numbers}")
    
    if len(numbers) >= 2:
        float_numbers = [float(n) for n in numbers]
        result = 1 + sum(float_numbers)
        print(f"[DEBUG] Calculation: 1 + {' + '.join(map(str, float_numbers))} = {result}")
        return str(result)
    else:
        return "Need at least 2 numbers"

sumPlus1 = Tool(
    name="sumPlus1",
    description="Returns the sum of the numbers plus one. call with 'SumPlus1'",
    func=GetSumPlus1
)
def Multiply(inp: str) -> str:
    print(f"[DEBUG] Tool received input: '{inp}'")
    numbers = re.findall(r'-?\d+\.?\d*', inp)
    print(f"[DEBUG] Extracted numbers: {numbers}")
    
    if len(numbers) >= 2:
        float_numbers = [float(n) for n in numbers]
        res = 1
        for i in float_numbers:
            res*=i
        return str(res)
    else:
        return "Need at least 2 numbers"

multiply = Tool(
    name="multiply",
    description="Returns the product of the numbers. call with 'multiply'",
    func=Multiply
)

wrapped_model = GPT4AllLangChain("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device="gpu")
agent = initialize_agent(
    [sumPlus1, multiply], 
    wrapped_model, 
    agent_type="conversational-react-description",
    verbose=True,
    max_iterations=2,
    agent_kwargs={
        "prefix": "You are a calculator assistant. If you have a tool fits a problem, use it. otherwise, Make your own choice. Tools are called with 'Toolname' not with '[Toolname]'. once you decide on a result, return it immediately. "
    }
)

print("=== TESTING ===")
result = agent.run("what is 5615 multiplied by 576")
print(f"\n=== FINAL RESULT: {result} ===")

print("\n=== DIRECT TOOL TEST ===")
direct_result = GetSumPlus1("11.521, 2")
print(f"Direct tool result: {direct_result}")