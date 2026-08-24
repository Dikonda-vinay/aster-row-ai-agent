#!/usr/bin/env python3
"""Aster & Row Customer Support Agent CLI."""

from __future__ import annotations

import argparse
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

from app.agent import Agent
from app.session import Session


def print_banner() -> None:
    print()
    print("=" * 68)
    print("                 ASTER & ROW")
    print("             CUSTOMER SUPPORT AI")
    print("=" * 68)
    print("  AI-powered support for orders, returns, shipping & warranties")
    print("-" * 68)
    print("  Type your question below.")
    print("  Type 'exit' or 'quit' to end the session.")
    print("=" * 68)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aster & Row customer support agent"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show retrieval and tool traces",
    )
    args = parser.parse_args()

    print()
    print("Initializing Aster & Row Support Agent...")
    print("Loading knowledge base...")

    try:
        agent = Agent()
        agent.retriever.build_or_load_index()
    except Exception as exc:
        print(f"\n[ERROR] Failed to initialize agent: {exc}")
        sys.exit(1)

    session = Session(session_id=str(uuid.uuid4()))

    print("\nKnowledge base ready.")
    print_banner()

    while True:
        try:
            user_text = input("  You  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended.")
            break

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            print("\nThank you for using Aster & Row Support.")
            print("Goodbye!\n")
            break

        try:
            result = agent.handle_turn(session, user_text)
        except Exception as exc:
            print(f"\n  [ERROR] {exc}\n", file=sys.stderr)
            continue

        print()
        print("  Agent")
        print("  " + "-" * 60)

        # Indent the response slightly for readability.
        for line in result.display_text.splitlines():
            print(f"  {line}")

        if result.sources:
            print()
            print("  Sources:")
            for source in result.sources:
                print(f"    • {source}")

        if result.handoff_recommended:
            print()
            print("  ⚠ Human support is recommended for this request.")

        print()

        if args.debug:
            print("  " + "=" * 60)
            print("  DEBUG TRACE")
            print("  " + "=" * 60)

            print("\n  Retrieved chunks:")
            for r in result.retrieved:
                print(
                    f"    • {r.chunk.filename} | "
                    f"{r.chunk.heading or '(intro)'} | "
                    f"score={r.score:.3f}"
                )

            if result.tool_calls:
                print("\n  Tool calls:")
                for tc in result.tool_calls:
                    print(
                        f"    • {tc.tool_name} "
                        f"args={tc.arguments}"
                    )

            print("  " + "=" * 60)
            print()


if __name__ == "__main__":
    main()