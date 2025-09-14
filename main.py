from mcp.server.fastmcp import FastMCP
from pydantic import Field
from mistralai import Mistral
import mcp.types as types
import os

mcp = FastMCP("Alim Server", port=3000, stateless_http=True, debug=True)
model = "mistral-small-2506"
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
temperature = 1.3

@mcp.tool(
    title="LLM Call",
    description="Call another LLM and return its response.",
)
async def original_response(query: str = Field(description="The user's query")) -> str:
    response = await client.chat.complete_async(
        model=model,
        messages=[
            {"role": "user", "content": query},
        ],
        temperature=temperature
    )
    return response.choices[0].message.content

@mcp.tool(
    title="Answer Review",
    description="Get a review of an answer to a user's query. Expects the user's original query and the LLM's latest response to the query.",
)
async def review(query: str = Field(description="The user's query"),
                           response: str = Field(description="The LLM's response")) -> str:
    review = await client.chat.complete_async(
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
    description="Get a refined answer to a user's query. Expects the user's query, the LLM's response to the query, and the judge's review of that response.",
)
async def refinement(query: str = Field(description="The user's query"),
                           response: str = Field(description="The LLM's response"),
                           review: str = Field(description="The judge's review of the response")) -> str:
    review = await client.chat.complete_async(
        model=model,
        messages=[
            {"role": "system", "content": "You are an assistant that refines other LLMs' messages based on a judge's feedback."},
            {"role": "user", "content": f"""You are provided a user's query, an LLM's response, and a judge's review of that response. Use the review to produce a refined version of the LLM's response. Do not precede your answer with any commentary. If the review says the response is good as is, return the original response.

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
    title="Answer Selection",
    description="Get the index of the best of multiple LLM responses according to an external judge. Expects the user's original query and multiple LLM responses indexed from 1."
)
async def selection(query: str = Field(description="The user's query"),
                        responses: str = Field(description="Multiple LLM responses indexed from 1")) -> str:
    content = f"""You are provided a user's query and multiple LLM responses to that query. Your job is to select the best response among them. If there is a clear best response, return that response with no change whatsoever. If multiple responses are equally good, return the first one among them with no change whatsoever. Do not precede your answer with any commentary and do not include the index.

    User's query:
    {query}

    LLM responses:
    {responses}
    """
    response = await client.chat.complete_async(
        model=model,
        messages=[
            {"role": "user", "content": content},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content

@mcp.tool(
    title="Review, Refine, Select",
    description="Given a user's query and an LLM's response, get w reviews and refinements of the response, then select the best refined response. Expects the user's original query, the LLM's response to the query, and w.",
)
async def rrs(query: str = Field(description="The user's query"),
                      response: str = Field(description="The LLM's response"),
                      w: int = Field(description="The number of reviews and refinements to generate")) -> str:
    responses = ""
    for i in range(w):
        rvw = await review(query=query, response=response)
        rfnmnt = await refinement(query=query, response=response, review=rvw)
        responses += f"Response {i+1}:\n {rfnmnt}\n\n"
    slctn = await selection(query=query, responses=responses)
    return slctn

@mcp.tool(
    title="Monte Carlo Tree Search",
    description="Given a user's query, get an LLM's response, get w reviews and refinements of the response, then select the best refined response. Repeat this process for d iterations. Expects the user's original query, the LLM's response to the query, w, and d.",
)
async def Monte_Carlo_Tree_Search(query: str = Field(description="The user's query"),
                w: int = Field(description="The width of the search tree"),
                d: int = Field(description="The depth of the search tree")) -> str:
    response = await original_response(query=query)
    for _ in range(d - 1):
        response = await rrs(query=query, response=response, w=w)
    return response

if __name__ == "__main__":
    mcp.run(transport="streamable-http")