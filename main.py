"""
MCP Server Template
"""

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from mistralai import Mistral
import mcp.types as types

mcp = FastMCP("Echo Server", port=3000, stateless_http=True, debug=True)


@mcp.tool(
    title="Echo Tool",
    description="Echo the input text",
)
def echo(text: str = Field(description="The text to echo")) -> str:
    return text

@mcp.tool(
    title="Call Mistral as a Judge",
    description="Call the Mistral API to review your response",
)
def call_mistral(exchange: str = Field(description="The user and Mistral's messages")) -> str:
    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)
    model = "mistral-large-latest"

    # Make a call to the Mistral API
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": "Below is a user's message and an LLM's response. Your job is to review the response for accuracy, completeness, and helpfulness. Provide constructive feedback and suggest improvements if necessary."},
            {"role": "user", "content": exchange},
        ],
        max_tokens=,
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
