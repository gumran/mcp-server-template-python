from mcp.server.fastmcp import FastMCP
from pydantic import Field
from mistralai import Mistral
import mcp.types as types
import os

mcp = FastMCP("KZ Server", port=3000, stateless_http=True, debug=True)
model = "mistral-small-2506"
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
temperature = 1.5

@mcp.tool(
    title="LLM Call",
    description="Call another LLM and return its response.",
)
def original_response(query: str = Field(description="The user's query")) -> str:
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "user", "content": query},
        ],
        temperature=temperature
    )
    return response.choices[0].message.content

@mcp.tool(
    title="Answer Review",
    description="Get a judge's review of an answer to a user's query.",
)
def review(query: str = Field(description="The user's query"),
                           response: str = Field(description="The LLM's response")) -> str:
    review = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant that reviews other LLMs' responses to users' queries."},
            {"role": "user", "content": f"""You are provided a user's query and an LLM's response. Suggest improvements to the LLM's response if you have any. Otherwise, say that the response is good as is.

            User's query:
            {query}

            LLM's response:
            {response}
            """
            },
        ],
        temperature=temperature,
    )
    return review.choices[0].message.content

@mcp.tool(
    title="Answer Refinement",
    description="Get a refined answer to a user's query according to a judge's review.",
)
def refinement(query: str = Field(description="The user's query"),
                           response: str = Field(description="The LLM's response"),
                           review: str = Field(description="The judge's review of the response")) -> str:
    review = client.chat.complete(
        model=model,
        messages=[
            {"role": "user", "content": f"""You are provided a user's query, an LLM's response, and a judge's review of that response. Use the review to produce a refined version of the LLM's response. If the review says the response is good as is, return the original response. Do not precede your answer with any commentary.

            User's query:
            {query}

            LLM's response:
            {response}

            Judge's review:
            {review}
            """
            },
        ],
        temperature=temperature,
    )
    return review.choices[0].message.content

@mcp.tool(
    title="Review and Refine",
    description="Given a user's query and an LLM's response, get a review of the response and then a refinement of the response based on the review.",
)
def branching(query: str = Field(description="The user's query"),
               response: str = Field(description="The LLM's response")) -> str:
    rvw = review(query=query, response=response)
    rfnd = refinement(query=query, response=response, review=rvw)
    return rfnd

@mcp.tool(
    title="Answer Selection",
    description="Return the best of multiple LLM responses according to a judge."
)
def selection(query: str = Field(description="The user's query"),
                        responses: str = Field(description="Multiple LLM responses indexed from 1")
                        ) -> str:
    content = f"""You are provided a user's query and multiple LLM responses to that query. Your job is to select the best response among them. If there is a clear best response, return that response with no change whatsoever. If multiple responses are equally good, return the first one among them with no change whatsoever. Do not precede your answer with any commentary and do not include the index.

    User's query:
    {query}

    LLM responses:
    {responses}
    """
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "user", "content": content},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content

# @mcp.tool(
#     title="Review, Refine, Select",
#     description="Given a user's query and an LLM's response, get multiple reviews and refinements of the response, then select the best refined response. This sends you one node deeper in the Monte Carlo Tree Search.",
# )
# def rrs(query: str = Field(description="The user's query"),
#                       response: str = Field(description="The LLM's response"),
#                       width: int = Field(description="The number of reviews and refinements to generate")) -> str:
#     responses = ""
#     for i in range(width):
#         rvw = review(query=query, response=response)
#         rfnmnt = refinement(query=query, response=response, review=rvw)
#         responses += f"Response {i+1}:\n {rfnmnt}\n\n"
#     slctn = selection(query=query, responses=responses)
#     return slctn

# @mcp.tool(
#     title="Monte Carlo Tree Search",
#     description="Given a user's query, get an LLM's response, multiple reviews and refinements of the response, then select the best refined response. Then get multiple reviews and refinements again, select the best one, and repeat this until it reaches a specified height.",
# )
# def Monte_Carlo_Tree_Search(query: str = Field(description="The user's query"),
#                 width: int = Field(description="The width of the search tree"),
#                 height: int = Field(description="The height of the search tree (including the root)")) -> str:
#     response = original_response(query=query)
#     for _ in range(height - 1):
#         response = rrs(query=query, response=response, width=width)
#     return response

if __name__ == "__main__":
    mcp.run(transport="streamable-http")