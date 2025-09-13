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
    title="Repeated Sampling Tool",
    description="Call 5 separate instances of the Mistral API and retrieve the best response as judged by the Mistral API itself",
)
def repeated_sampling(query: str = Field(description="The user's query")) -> str:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    model = "mistral-large-latest"

    responses = []
    for _ in range(5):
        response = client.chat.complete(
            model=model,
            messages=[
                {"role": "user", "content": query},
            ],
        )
        responses.append(response.choices[0].message.content)

    content = f"""You are provided a user's query and an LLM's 5 different responses to that same query. Please select the best one and output it.
    
    User's query:
    {query}
    
    """
    for i, resp in enumerate(responses):
        content += f"""Response {i+1}:
        {resp}
        
        """
    final_response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant that judges other LLMs' responses to users' queries."},
            {"role": "user", "content": content},
        ],
    )
    return final_response.choices[0].message.content

@mcp.tool(
    title="Sequential Refinement Tool",
    description="Call the Mistral API 5 times sequentially to refine the original answer",
)
def sequential_refinement(query: str = Field(description="The user's query")) -> str:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    model = "mistral-large-latest"

    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "user", "content": query},
        ],
    )
    for _ in range(4):
        review = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": "You are an assistant that reviews other LLMs' responses to users' queries."},
                {"role": "user", "content": f"""You are provided a user's query and an LLM's response. Suggest improvements to the LLM's response if you have any. Otherwise, say that the response is good as is.
                
                {response.choices[0].message.content}
                """
                },
            ],
        )

        content = f"""You are provided a user's query, an LLM's response, and a review of that response. Please improve the LLM's response based on the review. If the review says the response is good as is, return the original response.

        User's query:
        {query}

        LLM's response:
        {response.choices[0].message.content}

        Review:
        {review.choices[0].message.content}
        """
        response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": "You are an assistant that improves other LLMs' responses based on reviews."},
                {"role": "user", "content": content},
            ],
        )
    return response.choices[0].message.content


@mcp.tool(
    title="LLM-as-a-Judge Tool",
    description="Call the Mistral API to review your response",
)
def call_mistral(exchange: str = Field(description="The user's query and the LLM's last response")) -> str:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    model = "mistral-large-latest"

    # Make a call to the Mistral API
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": "You are provided a user's query and an LLM's response. Your job is to review the LLM's response. If necessary, suggest improvements."},
            {"role": "user", "content": exchange},
        ],
    )
    return response.choices[0].message.content

@mcp.resource(
    uri="greeting://{name}",
    description="Get a personalized greeting",
    name="Greeting Resource",
)
def get_greeting(
    name: str,
) -> str:
    return f"Hello, {name}!"

@mcp.prompt("")
def greet_user(
    name: str = Field(description="The name of the person to greet"),
    style: str = Field(description="The style of the greeting", default="friendly"),
) -> str:
    """Generate a greeting prompt"""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }

    return f"{styles.get(style, styles['friendly'])} for someone named {name}."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")