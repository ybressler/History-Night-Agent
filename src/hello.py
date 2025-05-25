from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel


class HistoryNightAgent:
    model = GeminiModel("gemini-2.5-pro-preview-05-06")
    agent = Agent(
        model,
        system_prompt="Be concise, reply with one sentence.",
    )


def main():
    hn_agent = HistoryNightAgent()
    result = hn_agent.run_sync(
        "What is the earliest year glass windows were used in a palace?"
    )
    print(result.output)


if __name__ == "__main__":
    main()
