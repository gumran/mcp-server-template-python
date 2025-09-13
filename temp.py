from mistralai import Mistral
import os
import dotenv
import os
from pydantic import Field

dotenv.load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
model = "mistral-large-latest"

def sequential_refinement(
    query: str = Field(description="The user's query"),
    response: str = Field(description="The LLM's response")
    ) -> str:

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
        temperature=0.7
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


print(sequential_refinement("If you double the mass of an object but keep the same kinetic energy, what happens to its velocity?"))