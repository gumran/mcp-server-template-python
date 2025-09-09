"""
MCP Server Template
"""

from mcp.server.fastmcp import FastMCP

# Create server
mcp = FastMCP("Echo Server", port=3000, stateless_http=True)


@mcp.tool()
def echo(text: str) -> str:
    """Echo the input text"""
    return text

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
