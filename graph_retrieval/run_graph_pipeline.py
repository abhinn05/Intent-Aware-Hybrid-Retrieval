#!/usr/bin/env python3
"""
run_graph_pipeline.py
=====================
CLI to build and query the Neo4j knowledge graph.

Usage
─────
  # 1. Build the graph from CSV
  python run_graph_pipeline.py build

  # 2. Search the graph
  python run_graph_pipeline.py search "Python developer with React and AWS experience"

  # 3. Interactive search REPL
  python run_graph_pipeline.py interactive

  # 4. Search with custom top-k
  python run_graph_pipeline.py search "data analyst" --top_k 20

Environment
───────────
  Copy .env.example → .env and fill in your Neo4j credentials.
"""

import argparse
import json
import os
import sys
import textwrap

from dotenv import load_dotenv

from graph_pipeline import build_graph, search_graph


def _get_neo4j_config():
    load_dotenv()
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    if not password:
        print("ERROR: NEO4J_PASSWORD is not set. Create a .env file (see .env.example).")
        sys.exit(1)
    return uri, user, password


def _get_csv_path():
    load_dotenv()
    return os.getenv("DATA_CSV_PATH", "data.csv")


def _format_result(idx: int, r: dict) -> str:
    """Pretty-print a single search result."""
    lines = [
        f"  ╔══ Rank #{idx} ══════════════════════════════════════",
        f"  ║ ID               : {r['id']}",
        f"  ║ Name             : {r['name']}",
        f"  ║ Experience (yrs) : {r['years_of_experience']}",
        f"  ║ Score            : {r['score']}",
        f"  ║ Matched Skills   : {', '.join(r['matched_skills']) if r['matched_skills'] else '—'}",
        f"  ║ Matched Roles    : {', '.join(r['matched_roles']) if r['matched_roles'] else '—'}",
        f"  ║ Explanation      : {r['explanation']}",
    ]
    # Wrap summary to 72 chars
    summary = r.get("skill_summary", "")
    if summary:
        wrapped = textwrap.fill(summary, width=72, initial_indent="  ║   ", subsequent_indent="  ║   ")
        lines.append(f"  ║ Summary          :")
        lines.append(wrapped)
    lines.append(f"  ╚═══════════════════════════════════════════════════")
    return "\n".join(lines)


def cmd_build(args):
    """Build the knowledge graph from CSV."""
    uri, user, pw = _get_neo4j_config()
    csv_path = _get_csv_path()
    stats = build_graph(csv_path, uri, user, pw, clear_existing=not args.append)
    print(f"\n✔ Graph built successfully.")
    print(f"  Candidates : {stats['candidates']}")
    print(f"  Skills     : {stats['skills']}")
    print(f"  Roles      : {stats['roles']}")
    print(f"  Relations  : {stats['relationships']}")


def cmd_search(args):
    """Search the graph for a query."""
    uri, user, pw = _get_neo4j_config()
    results = search_graph(args.query, uri, user, pw, top_k=args.top_k)

    if not results:
        print(f"\n⚠  No results found for: \"{args.query}\"")
        return

    print(f"\n🔍 Search results for: \"{args.query}\"  (top {args.top_k})")
    print(f"   Found {len(results)} result(s)\n")

    for i, r in enumerate(results, 1):
        print(_format_result(i, r))
        print()

    if args.json:
        print("\n── JSON output ──")
        print(json.dumps(results, indent=2, default=str))


def cmd_interactive(args):
    """Interactive search REPL."""
    uri, user, pw = _get_neo4j_config()
    top_k = args.top_k

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   Knowledge-Graph Search  –  Interactive Mode            ║")
    print("║   Type a query and press Enter. Type 'quit' to exit.     ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    while True:
        try:
            query = input("🔎 Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not query:
            continue

        results = search_graph(query, uri, user, pw, top_k=top_k)
        if not results:
            print(f"  ⚠  No results found.\n")
            continue

        print(f"\n  Found {len(results)} result(s):\n")
        for i, r in enumerate(results, 1):
            print(_format_result(i, r))
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Pipeline – Build & Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python run_graph_pipeline.py build
              python run_graph_pipeline.py search "Python developer"
              python run_graph_pipeline.py search "senior buyer procurement" --top_k 5
              python run_graph_pipeline.py interactive
        """),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── build ──
    p_build = sub.add_parser("build", help="Build the knowledge graph from CSV")
    p_build.add_argument("--append", action="store_true",
                         help="Append to existing graph instead of clearing first")
    p_build.set_defaults(func=cmd_build)

    # ── search ──
    p_search = sub.add_parser("search", help="Search the knowledge graph")
    p_search.add_argument("query", help="Free-text search query")
    p_search.add_argument("--top_k", type=int, default=10, help="Max results (default: 10)")
    p_search.add_argument("--json", action="store_true", help="Also output results as JSON")
    p_search.set_defaults(func=cmd_search)

    # ── interactive ──
    p_inter = sub.add_parser("interactive", help="Interactive search REPL")
    p_inter.add_argument("--top_k", type=int, default=10, help="Max results per query")
    p_inter.set_defaults(func=cmd_interactive)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
