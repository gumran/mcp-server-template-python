"""
MCP Server Template
"""

from mcp.server.fastmcp import FastMCP

# Create server
mcp = FastMCP("Echo Server", stateless_http=True)


@mcp.tool()
def echo(text: str) -> str:
    """Echo the input text"""
    return text


if __name__ == "__main__":
    mcp.run("streamable-http")
