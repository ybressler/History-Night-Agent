from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, AnyHttpUrl

from src.config import GEMINI_API_KEY


class ResponseFormatter(BaseModel):
    year: int = Field(description="answer to the question")
    citation: Optional[AnyHttpUrl] = Field(
        description="Citation to the Q if you have one"
    )


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20", google_api_key=GEMINI_API_KEY
)
model_with_tools = model.bind_tools([ResponseFormatter])


if __name__ == "__main__":
    # Simple text invocation
    result = model_with_tools.invoke(
        "What is the earliest year glass windows were used in a palace? (Starting in the year 850 AD)"
    )
    print(result.content)
