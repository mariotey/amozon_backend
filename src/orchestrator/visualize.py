"""
Compile the recommendation graph and print its Mermaid diagram.

Usage (from src/):
    python -m orchestrator.visualize
"""

from orchestrator.graph import build_agent


def main() -> None:
    agent = build_agent()
    mermaid = agent.get_graph().draw_mermaid()
    print(mermaid)


if __name__ == "__main__":
    main()
