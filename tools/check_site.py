#!/usr/bin/env python3
"""Check the published site for learner-facing defects; exit 1 if any are found.

Checks: broken relative links, pages nothing links to, markdown with no rendered
page, markdown left unrendered in HTML, and (when the untracked file
.internal-terms exists, one regex per line) internal names in tracked files.

Usage: python3 tools/check_site.py
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "site", "docs"}
ORPHAN_ALLOWED = {"404.html"}
HREF = re.compile(r'href="([^"#?]+)')
CODE_BLOCKS = re.compile(r"<(pre|code|script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
RAW_MARKDOWN = re.compile(r"\*\*[A-Za-z][^*<\n]{1,60}\*\*|^\s*```", re.MULTILINE)


def load_builder():
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "tools" / "build_site.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def site_pages():
    return sorted(p for p in ROOT.rglob("*.html") if not SKIP_DIRS & set(p.relative_to(ROOT).parts))


def link_problems(pages):
    broken, inbound = [], set()
    for page in pages:
        for href in HREF.findall(page.read_text(encoding="utf-8", errors="ignore")):
            if re.match(r"^[a-z]+:|^//", href):
                continue
            target = (page.parent / href).resolve()
            if target.is_dir():
                target = target / "index.html"
            if target.exists():
                inbound.add(target)
            else:
                broken.append(f"{page.relative_to(ROOT)} -> {href}")
    orphans = [str(p.relative_to(ROOT)) for p in pages
               if p.resolve() not in inbound and p.name not in ORPHAN_ALLOWED and p != ROOT / "index.html"]
    return broken, orphans


def unrendered_markdown(pages):
    hits = []
    for page in pages:
        text = CODE_BLOCKS.sub("", page.read_text(encoding="utf-8", errors="ignore"))
        match = RAW_MARKDOWN.search(text)
        if match:
            hits.append(f"{page.relative_to(ROOT)}: {match.group(0).strip()[:40]}")
    return hits


def unrendered_sources(builder):
    pages, _ = builder.plan_pages()
    return [str(p.md.relative_to(ROOT)) for p in pages if not p.out.exists()]


def internal_term_hits():
    terms_file = ROOT / ".internal-terms"
    if not terms_file.exists():
        print("note: .internal-terms not found; internal-name scan skipped")
        return []
    patterns = [line.strip() for line in terms_file.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if not patterns:
        return []
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-co", "--exclude-standard"],
                             capture_output=True, text=True, check=True).stdout.split()
    hits = []
    for name in tracked:
        if name.startswith(("site/", "docs/design/")):
            continue
        for pattern in patterns:
            if re.search(pattern, name, re.IGNORECASE):
                hits.append(f"{name}: file name matches {pattern}")
        path = ROOT / name
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{name}:{line}: {match.group(0)}")
    return hits


def report(title, items, limit=40):
    print(f"{title}: {len(items)}")
    for item in items[:limit]:
        print("  " + item)
    if len(items) > limit:
        print(f"  ... and {len(items) - limit} more")


def main():
    pages = site_pages()
    broken, orphans = link_problems(pages)
    checks = [
        ("Broken links", broken),
        ("Orphan pages", orphans),
        ("Markdown with no rendered page", unrendered_sources(load_builder())),
        ("Unrendered markdown in HTML", unrendered_markdown(pages)),
        ("Internal names", internal_term_hits()),
    ]
    for title, items in checks:
        report(title, items)
    failed = sum(len(items) for _, items in checks)
    print("FAIL" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
