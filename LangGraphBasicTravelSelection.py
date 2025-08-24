from typing import Annotated
from typing_extensions import TypedDict
from gpt4all import GPT4All
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import AIMessage, HumanMessage
from langchain_core.tools import tool
# --- Wrapper to handle memory + current input ---

class ChatLLMWrapper:
    def __init__(self, model, context_size=3):
        self.model = model
        self.context_size = context_size  # last N exchanges

    def invoke(self, messages: list) -> AIMessage:
        """
        messages: list of HumanMessage/AIMessage objects
        Returns AIMessage
        """
        # Keep last N message pairs (human + assistant)
        relevant_messages = messages[-(self.context_size * 2):] if len(messages) > (self.context_size * 2) else messages

        # Build a clean, structured prompt
        system_prompt = "You are a helpful assistant. Answer questions directly and accurately. Do not make up or hallucinate previous conversation history."
        
        context_parts = [system_prompt, ""]
        
        for msg in relevant_messages:
            if isinstance(msg, HumanMessage):
                context_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content}")

        # Add clear instruction for the current response
        context_parts.append("Assistant:")

        prompt = "\n".join(context_parts)
        
        # Generate response with stricter parameters to reduce hallucination
        response = self.model.generate(
            prompt, 
            max_tokens=150,
            temp=0.3,     
            top_p=0.8,       
            repeat_penalty=1.1, 
        )
        
        response = response.strip()
        
        if response.startswith("Assistant:"):
            response = response[10:].strip()
        if response.startswith("Human:"):
            response = response[6:].strip()
        
        if "Human:" in response:
            response = response.split("Human:")[0].strip()
        
        lines = response.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith("Human:") or line.startswith("You:"):
                break
            if line:
                clean_lines.append(line)
        
        response = '\n'.join(clean_lines).strip()
            
        return AIMessage(content=response)


# --- LangGraph setup ---
class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    """Process the conversation state and generate a response"""
    messages = state["messages"]
    
    # Generate response using our wrapper
    response = chat_llm.invoke(messages)
    
    return {"messages": [response]}
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
@tool
def findHotels(city: str) -> str:
    """Returns a list of hotels
    Args:
        city (str): The name of the city to find hotels in.
    """
    city = city.strip("'\" ")
    if city in hotels:
        return ", ".join(hotels[city])
    return "No hotels found."

# Initialize components
llm = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device="gpu")
chat_llm = ChatLLMWrapper(llm, context_size=3)

tools = [findHotels]
tool_map = {tool.name: tool for tool in tools}
# Build graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

print("Chatbot initialized successfully!")
print("ASCII Graph:")
print(graph.get_graph().draw_ascii())

# --- Main conversation loop ---
def main():
    conversation_state = {"messages": []}
    
    print("\n=== GPT4All Chatbot ===")
    print("Type 'quit', 'exit', or 'q' to end the conversation")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
                
            if not user_input:
                print("Please enter a message.")
                continue
            
            # Add user message to conversation
            user_message = HumanMessage(content=user_input)
            conversation_state["messages"].append(user_message)
            
            # Get AI response using the graph
            result = graph.invoke(conversation_state)
            
            # Update conversation state with the response
            conversation_state["messages"].extend(result["messages"])
            
            # Print the AI response
            ai_response = result["messages"][-1].content
            print(f"Assistant: {ai_response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()