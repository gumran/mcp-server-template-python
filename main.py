"""
MCP Server Template
"""

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from mistralai import Mistral
import mcp.types as types
import os

mcp = FastMCP("Echo Server", port=3000, stateless_http=True, debug=True)

@mcp.tool(
    title="Tool to Sample from LLM",
    description="Call another LLM and return its response.",
)
def sample_from_llm(query: str = Field(description="The user's query")) -> str:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    model = "mistral-large-latest"

    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content

@mcp.tool(
    title="Answer Review Tool",
    description="Get a review of an answer to a user's query. Expects the user's original query and the LLM's latest response to the query.",
)
def answer_review(query: str = Field(description="The user's query"),
                           response: str = Field(description="The LLM's latest response")) -> str:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    model = "mistral-large-latest"

    review = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant that reviews other LLMs' responses to users' queries."},
            {"role": "user", "content": f"""You are provided a user's query and an LLM's response. Suggest improvements to the LLM's response if you have any. Otherwise, say that the response is good as is.

            {response}
            """
            },
        ],
    )
    return review.choices[0].message.content

# @mcp.tool(
#     title="MCTS Tool",
#     description="Call the Mistral API to perform Monte Carlo Tree Search to refine the response",
# )


# @mcp.tool(
#     title="LLM-as-a-Judge Tool",
#     description="Call the Mistral API to review your response",
# )
# def call_mistral(exchange: str = Field(description="The user's query and the LLM's last response")) -> str:
#     client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
#     model = "mistral-large-latest"

#     # Make a call to the Mistral API
#     response = client.chat.complete(
#         model=model,
#         messages=[
#             {"role": "system", "content": "You are provided a user's query and an LLM's response. Your job is to review the LLM's response. If necessary, suggest improvements."},
#             {"role": "user", "content": exchange},
#         ],
#     )
#     return response.choices[0].message.content

# @mcp.resource(
#     uri="greeting://{name}",
#     description="Get a personalized greeting",
#     name="Greeting Resource",
# )
# def get_greeting(
#     name: str,
# ) -> str:
#     return f"Hello, {name}!"

# @mcp.prompt("")
# def greet_user(
#     name: str = Field(description="The name of the person to greet"),
#     style: str = Field(description="The style of the greeting", default="friendly"),
# ) -> str:
#     """Generate a greeting prompt"""
#     styles = {
#         "friendly": "Please write a warm, friendly greeting",
#         "formal": "Please write a formal, professional greeting",
#         "casual": "Please write a casual, relaxed greeting",
#     }

#     return f"{styles.get(style, styles['friendly'])} for someone named {name}."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")