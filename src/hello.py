from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

from src.config import GEMINI_API_KEY


class HistoryNightAgent:
    model = GeminiModel(
        model_name="gemini-2.5-flash-preview-05-20",
        provider=GoogleGLAProvider(api_key=GEMINI_API_KEY),
    )

    agent = Agent(
        model,
        system_prompt="Be concise, reply with one sentence.",
    )


def main():
    hn_agent = HistoryNightAgent().agent
    result = hn_agent.run_sync(
        "What is the earliest year glass windows were used in a palace? (Starting in the year 850 AD)",
        output_type=int,
    )
    print(result.output)


if __name__ == "__main__":
    main()
