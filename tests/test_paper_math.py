"""Check paper math survives GitHub Markdown rendering (requires gh and network).

Run from any directory: python3 tests/test_paper_math.py
"""

import html
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MATH = re.compile(
    r"```math\n(.*?)\n```|\$`(.*?)`\$|\$\$\n(.*?)\n\$\$"
    r"|```[^\n]*\n.*?\n```|`[^`\n]*`|\$([^$]+)\$",
    re.DOTALL,
)


def check(path):
    source = path.read_text()
    expected = [
        next(group for group in match.groups() if group is not None)
        for match in MATH.finditer(source)
        if any(group is not None for group in match.groups())
    ]
    result = subprocess.run(
        ["gh", "api", "markdown", "--input", "-"],
        input=json.dumps({"text": source, "mode": "gfm", "context": "oponite/lana"}),
        capture_output=True,
        text=True,
        check=True,
    )
    rendered = re.findall(
        r"<math-renderer\b[^>]*>(.*?)</math-renderer>", result.stdout, re.DOTALL
    )
    assert len(rendered) == len(expected), (
        f"{path.name}: expected {len(expected)} math expressions, got {len(rendered)}"
    )
    for index, (before, after) in enumerate(zip(expected, rendered), 1):
        after = html.unescape(html.unescape(after))
        # Markdown must preserve TeX escapes for sets, spacing, and identifiers.
        for escape in (r"\{", r"\}", r"\,", r"\;", r"\!", r"\_"):
            assert before.count(escape) == after.count(escape), (
                f"{path.name}: expression {index} lost {escape!r}: {before!r}"
            )
    print(f"{path.name}: {len(rendered)} expressions preserved by GitHub")


if __name__ == "__main__":
    for name in ("semantics.md", "semantics-2.md"):
        check(ROOT / "papers" / name)
