from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain.tools import tool

from AmadeusCall import searchHotels
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


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
# @tool
# def findHotelsByCity(city: str) -> str:
#     """Returns a list of hotels
#     Args:
#         city (str): The name of the city to find hotels in.
#     """
#     city = city.strip("'\" ")
#     if city in hotels:
#         return ", ".join(hotels[city])
#     return "No hotels found."
@tool
def FindHotelsByCoords(coords: list, radius: int) -> str:
    """Returns a list of hotels - ID is used for future functions using the hotel, not to be returned to user. 
    Args:
        coords (list): The Coordinates of the location, in the format [latitude, longitude].
        radius (int): The radius (in km) to search for hotels.
    """
    latitude, longitude = coords
    return searchHotels(latitude, longitude, radius)

@tool
def DescribeHotel(hotel_ID: str) -> str:
    """Returns a description of a hotel
    Args:
        hotel_ID (str): The ID of the hotel to describe.
    """
    #  had this working on local data - but trying to figure out how to transfer hotelID to amadeus hotelID for the API. 
    return "Hotel description not available."
# tools = [findHotelsByCity, FindHotelsByCoords, DescribeHotel]
tools = [FindHotelsByCoords, DescribeHotel]

class State(TypedDict):
    messages: Annotated[list, add_messages]

load_dotenv()
graph_builder = StateGraph(State)

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


tool_node = BasicToolNode(tools)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

from IPython.display import Image, display




try:
    display(Image(graph.get_graph().draw_ascii()))
except Exception:
    pass

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    stream_graph_updates(user_input)