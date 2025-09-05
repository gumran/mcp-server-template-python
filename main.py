"""
FastMCP Echo Server
"""

from fastmcp import FastMCP

# Create server
mcp = FastMCP("Echo Server", stateless_http=True)


@mcp.tool
def echo(text: str) -> str:
    """Echo the input text"""
    return text


if __name__ == "__main__":
    mcp.run(transport="http", host="localhost", port=3000)
