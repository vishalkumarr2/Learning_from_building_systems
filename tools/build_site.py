#!/usr/bin/env python3
"""Render every curriculum markdown file to a styled HTML page beside it.

Covers optimization/, llm-to-vla/ and robotics/learn/, writes the robotics
card pages and the lesson manifest (assets/lessons.js), and wires the shared
learner assets (assets/learn.css, assets/learn.js) into the hand-built pages,
including the HTML-only ml-compilers track.

Usage: python3 tools/build_site.py
"""

import html
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify, unique

SITE_ROOT = Path(__file__).resolve().parent.parent
SITE_NAME = "Learning From Building Systems"
GENERATED_MARKER = '<div class="md-content">'
NON_LESSON_STEMS = {
    "readme", "curriculum", "00-learning-plan", "00-mastery-plan",
    "study-plan", "resources", "gaps-roadmap",
}
WEEK_DIR = re.compile(r"week-\d+$")
HAND_MAINTAINED = {"optimization/curriculum.html"}


@dataclass(frozen=True)
class Track:
    key: str
    name: str
    accent: str
    math: str
    landing: str


@dataclass(frozen=True)
class RoboticsSection:
    slug: str
    title: str
    subtitle: str
    accent: str
    card_blurb: str
    subgroups: tuple = (("exercises", "Exercise"),)


@dataclass
class Page:
    md: Path
    out: Path
    track: Track
    title: str
    is_lesson: bool
    seq_key: Path
    prev: "Page | None" = None
    next: "Page | None" = None
    extra_meta: dict = field(default_factory=dict)


GENERIC_TRACKS = (
    Track("optimization", "Optimization", "cyan", "katex", "optimization/index.html"),
    Track("llm-to-vla", "LLM to VLA", "violet", "katex", "llm-to-vla/index.html"),
)
ROBOTICS_TRACK = Track("robotics", "Robotics", "blue", "mathjax", "robotics/index.html")
ML_COMPILERS = Track("ml-compilers", "ML Compilers", "emerald", "katex", "ml-compilers/index.html")

ROBOTICS_SECTIONS = (
    RoboticsSection("cpp-advanced", "Advanced C++ for Robotics",
                    "Real-time, safety-critical & production systems programming", "indigo",
                    "Real-time, safety-critical & production systems — 18 modules", ()),
    RoboticsSection("zephyr", "Zephyr RTOS",
                    "From basics to production firmware with Zephyr, DMA, SPI, CAN, and ROS2 integration",
                    "emerald", "Embedded firmware: Zephyr, DMA, SPI, CAN, ROS2 bridge — 17 tutorials",
                    (("exercises", "Exercise"), ("study-notes", "Notes"), ("deep-dive", "Deep Dive"))),
    RoboticsSection("electronics", "Electronics Fundamentals",
                    "Passive components, semiconductors, op-amps, communication protocols (UART, SPI, I2C, CAN)",
                    "amber", "Passive components through CAN bus — deep-dive lessons + exercises"),
    RoboticsSection("control-systems", "Control Systems",
                    "PID to advanced control: MCU inner loop to Jetson outer loop — 10 lessons + exercises + debugging",
                    "rose", "PID to advanced control: MCU inner loop to Jetson outer loop — 10 lessons + exercises",
                    (("exercises", "Exercise"), ("debugging", "Debug"), ("senior-questions", "Senior Q"))),
    RoboticsSection("navigation-estimator", "Navigation Estimator",
                    "Dead reckoning, Kalman filters, IMU fusion, failure modes, and Ceres optimization",
                    "sky", "Dead reckoning, Kalman filters, IMU fusion, and failure diagnosis"),
    RoboticsSection("ros2-handson", "ROS 2 Hands-On",
                    "Nodes, topics, actions, TF2, Nav2, and QoS — practical ROS 2 development",
                    "violet", "Nodes, topics, TF2, Nav2, QoS — practical ROS 2 development"),
    RoboticsSection("python-scripting", "Python Scripting",
                    "Type-safe Python, testing, CLI scripts, and timeseries analysis for robotics",
                    "teal", "Type-safe Python, testing, CLI tools, and timeseries analysis"),
    RoboticsSection("logic", "Logic & Critical Reasoning",
                    "Arguments, fallacies, propositional logic, deduction, scientific reasoning — 9 lessons + exercises",
                    "cyan", "Arguments, fallacies, propositional logic, deduction, scientific reasoning"),
)
ROBOTICS_LEARN = "robotics/learn"
ROBOTICS_EXTRA_CARDS = (
    ("learn/study-plan.html", "Unified Study Plan",
     "120–130 hour roadmap: Electronics → Protocols → Embedded Systems", "blue"),
    ("learn/resources.html", "External Resources",
     "Curated courses, papers, repos, and videos for each learning track", "cyan"),
    ("learn/gaps-roadmap.html", "Gaps & Roadmap",
     "Learning gap analysis and prioritised improvement roadmap", "orange"),
)


def section_track(section):
    return Track(f"{ROBOTICS_LEARN}/{section.slug}", section.title, section.accent, "mathjax",
                 f"{ROBOTICS_LEARN}/{section.slug}/index.html")


def esc(text):
    return html.escape(text, quote=True)


def rel(path):
    return path.relative_to(SITE_ROOT).as_posix()


def href_between(from_page, to_page):
    return Path(os.path.relpath(to_page, from_page.parent)).as_posix()


def root_prefix(page_path):
    depth = len(page_path.relative_to(SITE_ROOT).parts) - 1
    return "../" * depth


def plain_markdown(text):
    return re.sub(r"[*_`]|\[([^\]]*)\]\([^)]*\)", lambda m: m.group(1) or "", text).strip()


def extract_title(md_text):
    match = re.search(r"^#\s+(.+)", md_text, re.MULTILINE)
    if match:
        return plain_markdown(match.group(1))
    for line in md_text.splitlines():
        if line.strip():
            return plain_markdown(line.strip())[:80]
    return "Untitled"


def extract_subtitle(md_text):
    for line in md_text.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("###"):
            return plain_markdown(line.lstrip("#"))
        if line.startswith("**") or line[0].isalpha():
            return plain_markdown(line)[:120]
    return ""


def slug_to_title(slug):
    return re.sub(r"^\d+[-_]", "", slug).replace("-", " ").replace("_", " ").title()


def output_path(md, card_dirs):
    """Map a markdown source to the HTML file it renders to."""
    parent = md.parent
    if md.name == "notes.md":
        return parent / "index.html"
    if md.name == "README.md":
        if parent in card_dirs or (parent / "notes.md").exists():
            return parent / "readme.html"
        return parent / "index.html"
    if re.fullmatch(r"[A-Z0-9-]+", md.stem):
        return parent / (md.stem.lower() + ".html")
    return parent / (md.stem + ".html")


CODE_SPAN = re.compile(r"(```.*?```|~~~.*?~~~|`[^`\n]+`)", re.DOTALL)
DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE_MATH = re.compile(r"(?<![\\$\w])\$(?=\S)([^$\n]+?)(?<=\S)\$(?![\w$])")
MERMAID = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
PLACEHOLDER = re.compile(r"MATHPH(\d+)XEND")


def protect_math(md_text):
    """Swap math for placeholders so markdown emphasis cannot mangle it."""
    stash = []

    def keep(text):
        stash.append(text)
        return f"MATHPH{len(stash) - 1}XEND"

    def display(match):
        return f'\n<div class="math-display">{keep(match.group(0))}</div>\n'

    pieces = CODE_SPAN.split(md_text)
    for i in range(0, len(pieces), 2):
        piece = DISPLAY_MATH.sub(display, pieces[i])
        pieces[i] = INLINE_MATH.sub(lambda m: keep(m.group(0)), piece)
    return "".join(pieces), stash


def restore_math(body, stash):
    return PLACEHOLDER.sub(lambda m: esc(stash[int(m.group(1))]), body)


LIST_FENCE = re.compile(r"(?<=\S)\n( {1,3})```[^\n]*\n(.*?)\n\1```", re.DOTALL)


def indent_list_fences(md_text):
    """Turn fences nested in list items into indented code blocks, which python-markdown nests reliably."""
    def fix(match):
        indent = match.group(1)
        lines = (indent + match.group(2)).split("\n")
        return "\n\n" + "\n".join("        " + line[len(indent):] if line.strip() else "" for line in lines)
    return LIST_FENCE.sub(fix, md_text)


def preprocess_markdown(md_text):
    md_text = indent_list_fences(md_text)
    md_text = MERMAID.sub(lambda m: f'\n<div class="mermaid">\n{m.group(1).strip()}\n</div>\n', md_text)
    return re.sub(r"<details(?![^>]*markdown=)", '<details markdown="1"', md_text)


def retitle_headings(body):
    """Re-derive heading ids from their final text and record a plain label for the TOC."""
    used = set()

    def fix(match):
        level, inner = match.group(1), match.group(3)
        text = html.unescape(re.sub(r"<[^>]+>", "", inner))
        hid = unique(slugify(text, "-"), used)
        return f'<h{level} id="{hid}" data-plain="{esc(text)}">{inner}</h{level}>'

    return re.sub(r'<h([1-6]) id="([^"]*)">(.*?)</h\1>', fix, body, flags=re.DOTALL)


def render_markdown(md_text):
    md_text, stash = protect_math(preprocess_markdown(md_text))
    converter = markdown.Markdown(extensions=[
        "tables", "fenced_code", "codehilite", "toc", "sane_lists", "smarty", "md_in_html",
    ], extension_configs={"codehilite": {"css_class": "highlight", "guess_lang": False}})
    body = converter.convert(md_text)
    body = re.sub(r"<p>\s*(<summary>.*?</summary>)\s*", r"\1\n<p>", body, flags=re.DOTALL)
    body = re.sub(r"<p>\s*</p>", "", body)
    return retitle_headings(restore_math(body, stash))


def rewrite_links(body, page, md_to_html, card_dirs):
    """Point .md links at the HTML page each markdown file renders to."""
    pattern = re.compile(r'href="(?![a-z]+:|//|#)([^"#]+?\.md)(#[^"]*)?"', re.IGNORECASE)

    def replace(match):
        target = (page.md.parent / match.group(1)).resolve()
        out = md_to_html.get(target) or output_path(target, card_dirs)
        return f'href="{href_between(page.out, out)}{match.group(2) or ""}"'

    return pattern.sub(replace, body)


def math_head(engine):
    if engine == "katex":
        return """    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}], throwOnError: false});"></script>"""
    return """    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }
        };
    </script>
    <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>"""


def learner_head(page_path, meta):
    """Return the tags that load the shared learner assets plus page metadata."""
    prefix = root_prefix(page_path)
    tags = [f'<meta name="lfbs-root" content="{prefix or "./"}">',
            f'<meta name="lfbs-path" content="{esc(rel(page_path))}">']
    tags += [f'<meta name="lfbs-{k}" content="{esc(v)}">' for k, v in meta.items()]
    tags += [f'<link rel="stylesheet" href="{prefix}assets/learn.css">',
             f'<script defer src="{prefix}assets/lessons.js"></script>',
             f'<script defer src="{prefix}assets/learn.js"></script>']
    return "    <!-- lfbs:begin -->\n    " + "\n    ".join(tags) + "\n    <!-- lfbs:end -->"


PAGE_CSS = """        body { font-family: 'Inter', sans-serif; }
        .md-content h1 { font-size: 2rem; font-weight: 700; color: #111827; margin-top: 2rem; margin-bottom: 1rem; }
        .md-content h2 { font-size: 1.5rem; font-weight: 600; color: #1f2937; margin-top: 1.75rem; margin-bottom: 0.75rem; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.5rem; }
        .md-content h3 { font-size: 1.25rem; font-weight: 600; color: #374151; margin-top: 1.5rem; margin-bottom: 0.5rem; }
        .md-content h4 { font-size: 1.1rem; font-weight: 600; color: #4b5563; margin-top: 1.25rem; margin-bottom: 0.5rem; }
        .md-content p { margin-bottom: 1rem; line-height: 1.75; color: #374151; }
        .md-content ul { list-style: disc; margin-left: 1.5rem; margin-bottom: 1rem; color: #374151; }
        .md-content ol { list-style: decimal; margin-left: 1.5rem; margin-bottom: 1rem; color: #374151; }
        .md-content li { margin-bottom: 0.35rem; line-height: 1.6; }
        .md-content code { background-color: #f3f4f6; padding: 0.15rem 0.4rem; border-radius: 0.25rem; font-size: 0.875rem; font-family: 'Fira Code', 'Courier New', monospace; color: #be185d; }
        .md-content pre { background-color: #1f2937; color: #e5e7eb; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; margin-bottom: 1.25rem; font-size: 0.85rem; line-height: 1.6; }
        .md-content pre code { background: none; padding: 0; color: inherit; font-size: inherit; }
        .md-content blockquote { border-left: 4px solid #6366f1; background: #eef2ff; padding: 0.75rem 1rem; margin-bottom: 1rem; border-radius: 0 0.5rem 0.5rem 0; color: #374151; }
        .md-content table { width: 100%; border-collapse: collapse; margin-bottom: 1.25rem; }
        .md-content th { background: #f9fafb; padding: 0.5rem 0.75rem; border: 1px solid #e5e7eb; text-align: left; font-weight: 600; font-size: 0.875rem; }
        .md-content td { padding: 0.5rem 0.75rem; border: 1px solid #e5e7eb; font-size: 0.875rem; }
        .md-content tr:hover { background-color: #f9fafb; }
        .md-content hr { border: 0; border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }
        .md-content strong { font-weight: 600; }
        .md-content a { color: #2563eb; text-decoration: underline; }
        .md-content a:hover { color: #1d4ed8; }
        .md-content img { max-width: 100%; border-radius: 0.5rem; margin: 1rem 0; }
        .md-content details { margin-bottom: 1rem; border: 1px solid #e5e7eb; border-radius: 0.5rem; overflow: hidden; }
        .md-content summary { cursor: pointer; padding: 0.75rem 1rem; background: #f9fafb; font-weight: 600; color: #374151; }
        .md-content summary:hover { background: #f3f4f6; }
        .md-content details[open] summary { border-bottom: 1px solid #e5e7eb; }
        .md-content details > *:not(summary) { padding-left: 1rem; padding-right: 1rem; }
        .md-content .mermaid { margin: 1.5rem 0; overflow-x: auto; }
        .md-content .math-display { margin: 1.5rem 0; overflow-x: auto; }"""

CARD_CSS = """        body { font-family: 'Inter', sans-serif; }
        .card { background-color: white; border-radius: 0.75rem; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1); transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; }
        .card:hover { transform: translateY(-5px); box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1); }"""


def nav_link(href, label, accent, align):
    if not href:
        return "<span></span>"
    return f'<a href="{href}" class="lfbs-pager-{align} text-{accent}-600 hover:text-{accent}-800">{label}</a>'


def html_page(title, body_html, page_path, track, up, prev_link=None, next_link=None,
              meta=None, uses_mermaid=False):
    """Wrap rendered markdown in the site's standard page shell.

    up, prev_link and next_link are (href, label) pairs or None; meta becomes
    lfbs-* meta tags that drive assets/learn.js.
    """
    accent = track.accent
    up_href, up_label = up
    pager = ""
    if prev_link or next_link:
        prev_html = nav_link(prev_link and prev_link[0], f"&larr; {esc(prev_link[1])}" if prev_link else "", accent, "prev")
        next_html = nav_link(next_link and next_link[0], f"{esc(next_link[1])} &rarr;" if next_link else "", accent, "next")
        pager = f"""
        <nav class="lfbs-pager mt-6 flex justify-between gap-4 text-sm" aria-label="Lesson navigation">
            {prev_html}
            {next_html}
        </nav>"""
    mermaid_head = '\n    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>' if uses_mermaid else ""
    mermaid_init = "\n    <script>mermaid.initialize({ startOnLoad: true });</script>" if uses_mermaid else ""
    suffix = SITE_NAME if track.key.startswith("robotics") else f"{track.name} - {SITE_NAME}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{esc(title)} - {esc(suffix)}</title>
    <script src="https://cdn.tailwindcss.com"></script>
{math_head(track.math)}{mermaid_head}
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
{PAGE_CSS}
    </style>
{learner_head(page_path, meta or {})}
</head>
<body class="bg-gray-100 text-gray-800">
    <div class="container mx-auto p-4 sm:p-6 lg:p-8 max-w-4xl">
        <a href="{up_href}" class="text-{accent}-600 hover:text-{accent}-800 mb-4 inline-block">&larr; {esc(up_label)}</a>

        <div class="bg-white rounded-xl shadow-md p-5 sm:p-8 md:p-10">
            <div class="md-content">
{body_html}
            </div>
        </div>
{pager}
    </div>{mermaid_init}
</body>
</html>
"""


def card_html(href, title, description, accent):
    desc = f'\n                    <p class="text-gray-600">{esc(description)}</p>' if description else ""
    return f"""                <a href="{href}" class="card block p-6">
                    <h3 class="font-bold text-xl mb-2 text-{accent}-700">{esc(title)}</h3>{desc}
                </a>"""


def index_page(title, subtitle, cards, page_path, up, accent, meta=None):
    """Generate a landing page made of cards."""
    up_href, up_label = up
    meta = {"generated": "cards", **(meta or {})}
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{esc(title)} - {SITE_NAME}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
{CARD_CSS}
    </style>
{learner_head(page_path, meta)}
</head>
<body class="bg-gray-100 text-gray-800">
    <div class="container mx-auto p-4 sm:p-6 lg:p-8">
        <a href="{up_href}" class="text-{accent}-600 hover:text-{accent}-800 mb-4 inline-block">&larr; {esc(up_label)}</a>

        <header class="text-center mb-12">
            <h1 class="text-3xl sm:text-4xl md:text-5xl font-bold text-gray-900">{esc(title)}</h1>
            <p class="mt-4 text-lg text-gray-600 max-w-3xl mx-auto">{esc(subtitle)}</p>
        </header>

        <div id="lfbs-progress"></div>

        <main>
            <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
{chr(10).join(cards)}
            </div>
        </main>
    </div>
</body>
</html>
"""


def is_module_index(md, out, card_dirs):
    return out.name == "index.html" and md.parent.parent in card_dirs


def is_lesson(md, out, card_dirs):
    if is_module_index(md, out, card_dirs):
        return True
    return out.name != "index.html" and out.stem.lower() not in NON_LESSON_STEMS


def sequence_key(md, out, card_dirs):
    if WEEK_DIR.match(md.parent.name) or is_module_index(md, out, card_dirs):
        return md.parent.parent
    return md.parent


def markdown_files(track_dir):
    return sorted(p for p in track_dir.rglob("*.md")
                  if not any(part.startswith(".") for part in p.relative_to(SITE_ROOT).parts))


def robotics_track_for(md):
    parts = md.relative_to(SITE_ROOT / ROBOTICS_LEARN).parts
    for section in ROBOTICS_SECTIONS:
        if len(parts) > 1 and parts[0] == section.slug:
            return section_track(section)
    return ROBOTICS_TRACK


def plan_pages():
    """Decide the output path, title, track and lesson order of every markdown page."""
    card_dirs = {SITE_ROOT / ROBOTICS_LEARN / s.slug for s in ROBOTICS_SECTIONS}
    sources = [(md, track) for track in GENERIC_TRACKS for md in markdown_files(SITE_ROOT / track.key)]
    sources += [(md, robotics_track_for(md)) for md in markdown_files(SITE_ROOT / ROBOTICS_LEARN)]
    pages = []
    for md, track in sources:
        out = output_path(md, card_dirs)
        title = extract_title(md.read_text(encoding="utf-8"))
        pages.append(Page(md, out, track, title, is_lesson(md, out, card_dirs), sequence_key(md, out, card_dirs)))
    link_sequences(pages)
    return pages, card_dirs


def link_sequences(pages):
    groups = {}
    for page in pages:
        if page.is_lesson:
            groups.setdefault(page.seq_key, []).append(page)
    for group in groups.values():
        group.sort(key=lambda p: rel(p.md))
        for before, after in zip(group, group[1:]):
            before.next, after.prev = after, before


def existing_title(index_html):
    match = re.search(r"<title>(.*?)</title>", index_html.read_text(encoding="utf-8"), re.DOTALL)
    return html.unescape(match.group(1).split(" - ")[0].strip()) if match else slug_to_title(index_html.parent.name)


def up_link(page, titles):
    """Find the nearest index page above this one and its title."""
    directory = page.out.parent.parent if page.out.name == "index.html" else page.out.parent
    while directory != SITE_ROOT.parent:
        index = directory / "index.html"
        if index in titles:
            return href_between(page.out, index), f"Back to {titles[index]}"
        if index.exists() and index != page.out:
            return href_between(page.out, index), f"Back to {existing_title(index)}"
        directory = directory.parent
    return href_between(page.out, SITE_ROOT / "index.html"), "Back to Home"


def is_hand_built(index):
    return index.exists() and 'name="lfbs-generated"' not in index.read_text(encoding="utf-8")


def collection_pages(pages):
    """Plan card indexes for generic-track folders that have pages but no index."""
    planned = {p.out for p in pages}
    by_dir = {}
    for page in pages:
        if page.track.key not in {t.key for t in GENERIC_TRACKS}:
            continue
        owner = page.out.parent.parent if page.out.name == "index.html" else page.out.parent
        by_dir.setdefault(owner, []).append(page)
    collections = {}
    for directory, members in by_dir.items():
        index = directory / "index.html"
        if index in planned or is_hand_built(index) or len(members) < 2 or rel(directory) in {t.key for t in GENERIC_TRACKS}:
            continue
        collections[index] = (members[0].track, sorted(members, key=lambda p: rel(p.out)))
    return collections


def page_meta(page):
    meta = {"track": page.track.key}
    if page.is_lesson:
        meta["lesson"] = "1"
    return meta


def write_markdown_pages(pages, card_dirs, titles):
    md_to_html = {p.md.resolve(): p.out for p in pages}
    for page in pages:
        body = render_markdown(page.md.read_text(encoding="utf-8"))
        body = rewrite_links(body, page, md_to_html, card_dirs)
        prev_link = page.prev and (href_between(page.out, page.prev.out), page.prev.title)
        next_link = page.next and (href_between(page.out, page.next.out), page.next.title)
        document = html_page(page.title, body, page.out, page.track, up_link(page, titles),
                             prev_link, next_link, page_meta(page), 'class="mermaid"' in body)
        page.out.write_text(document, encoding="utf-8")


def write_collections(collections, titles):
    for index, (track, members) in collections.items():
        cards = [card_html(href_between(index, m.out), m.title, "", track.accent) for m in members]
        stub = Page(index, index, track, titles[index], False, index.parent)
        document = index_page(titles[index], f"{track.name} — {len(members)} pages", cards, index,
                              up_link(stub, titles), track.accent)
        index.write_text(document, encoding="utf-8")


def section_cards(section, pages_by_out):
    base = SITE_ROOT / ROBOTICS_LEARN / section.slug
    track = section_track(section)
    top = sorted((p for p in pages_by_out.values() if p.out.parent == base and p.out.name != "index.html"),
                 key=lambda p: (p.out.stem not in {"readme", "study-plan"}, p.out.stem))
    cards = [card_html(p.out.name, "Overview" if p.out.stem == "readme" else p.title, "", track.accent) for p in top]
    modules = [p for p in pages_by_out.values() if p.out.name == "index.html" and p.out.parent.parent == base]
    for page in sorted(modules, key=lambda p: p.out):
        subtitle = extract_subtitle(page.md.read_text(encoding="utf-8"))
        cards.append(card_html(href_between(base / "index.html", page.out), page.title,
                               subtitle[:100] or slug_to_title(page.md.parent.name), track.accent))
    for subdir, label in section.subgroups:
        members = sorted((p for p in pages_by_out.values() if p.out.parent == base / subdir), key=lambda p: p.out.name)
        cards += [card_html(f"{subdir}/{p.out.name}", f"{label}: {p.title}", "", track.accent) for p in members]
    return cards


def write_robotics_indexes(pages):
    pages_by_out = {p.out: p for p in pages}
    for section in ROBOTICS_SECTIONS:
        index = SITE_ROOT / ROBOTICS_LEARN / section.slug / "index.html"
        cards = section_cards(section, pages_by_out)
        document = index_page(section.title, section.subtitle, cards, index, ("../../index.html", "Back to Robotics"),
                              section.accent, {"landing": section_track(section).key})
        index.write_text(document, encoding="utf-8")
    cards = [card_html(href, title, blurb, accent) for href, title, blurb, accent in ROBOTICS_EXTRA_CARDS]
    cards += [card_html(f"learn/{s.slug}/index.html", s.title, s.card_blurb, s.accent) for s in ROBOTICS_SECTIONS]
    landing = SITE_ROOT / "robotics" / "index.html"
    keys = ",".join(section_track(s).key for s in ROBOTICS_SECTIONS)
    document = index_page("Robotics", "Embedded systems, real-time C++, Zephyr RTOS, electronics, and protocol deep-dives",
                          cards, landing, ("../index.html", "Back to Home"), "blue", {"landing": keys})
    landing.write_text(document, encoding="utf-8")


ML_LINK_FIXES = (
    (re.compile(r'href="README\.html"'), 'href="index.html"'),
    (re.compile(r'href="\.\./\.\./README\.html"'), 'href="../../index.html"'),
    (re.compile(r'href="\.\./\.\./STUDY-PLAN\.html"'), 'href="../../curriculum.html"'),
)


def inject_learner_head(path, meta):
    """Insert or refresh the learner asset block in a hand-built page."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\s*<!-- lfbs:begin -->.*?<!-- lfbs:end -->", "", text, flags=re.DOTALL)
    text = text.replace("</head>", learner_head(path, meta) + "\n</head>", 1)
    path.write_text(text, encoding="utf-8")


def ml_compiler_lessons():
    """Fix the HTML-only ml-compilers pages and return their lessons in order."""
    lessons = []
    for page in sorted((SITE_ROOT / ML_COMPILERS.key / "weeks").glob("week-*/day-*.html")):
        text = page.read_text(encoding="utf-8")
        for pattern, replacement in ML_LINK_FIXES:
            text = pattern.sub(replacement, text)
        page.write_text(text, encoding="utf-8")
        inject_learner_head(page, {"track": ML_COMPILERS.key, "lesson": "1"})
        lessons.append((rel(page), existing_title(page)))
    for page in (SITE_ROOT / ML_COMPILERS.key / "weeks").glob("week-*/index.html"):
        inject_learner_head(page, {"track": ML_COMPILERS.key})
    inject_learner_head(SITE_ROOT / ML_COMPILERS.key / "curriculum.html", {"track": ML_COMPILERS.key})
    return lessons


def lesson_manifest(pages, ml_lessons):
    """Map each progress key to its ordered (path, title) lessons; daily tracks count only days."""
    manifest = {}
    for page in sorted(pages, key=lambda p: rel(p.md)):
        if page.is_lesson:
            manifest.setdefault(page.track.key, []).append((rel(page.out), page.title))
    for key, lessons in manifest.items():
        days = [entry for entry in lessons if Path(entry[0]).name.startswith("day-")]
        if days:
            manifest[key] = days
    manifest[ML_COMPILERS.key] = ml_lessons
    return manifest


def write_manifest(manifest):
    names = {t.key: t.name for t in GENERIC_TRACKS + (ML_COMPILERS, ROBOTICS_TRACK)}
    names.update({section_track(s).key: s.title for s in ROBOTICS_SECTIONS})
    payload = {"names": names, "lessons": manifest}
    (SITE_ROOT / "assets").mkdir(exist_ok=True)
    (SITE_ROOT / "assets" / "lessons.js").write_text(
        "window.LFBS = " + json.dumps(payload, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")


def stale_generated_pages(pages, collections):
    """List generated pages whose markdown source no longer exists."""
    expected = {p.out for p in pages} | set(collections)
    roots = [SITE_ROOT / t.key for t in GENERIC_TRACKS] + [SITE_ROOT / ROBOTICS_LEARN]
    return sorted(h for root in roots for h in root.rglob("*.html")
                  if h not in expected and "code" not in h.relative_to(root).parts
                  and rel(h) not in HAND_MAINTAINED
                  and GENERATED_MARKER in h.read_text(encoding="utf-8"))


def main():
    pages, card_dirs = plan_pages()
    titles = {p.out: p.title for p in pages}
    collections = collection_pages(pages)
    titles.update({index: f"{slug_to_title(index.parent.name)}" for index in collections})
    titles.update({SITE_ROOT / ROBOTICS_LEARN / s.slug / "index.html": s.title for s in ROBOTICS_SECTIONS})
    write_markdown_pages(pages, card_dirs, titles)
    write_collections(collections, titles)
    write_robotics_indexes(pages)
    ml_lessons = ml_compiler_lessons()
    write_manifest(lesson_manifest(pages, ml_lessons))
    inject_learner_head(SITE_ROOT / "index.html", {"home": "1"})
    for track in GENERIC_TRACKS + (ML_COMPILERS,):
        inject_learner_head(SITE_ROOT / track.landing, {"landing": track.key})
    print(f"Rendered {len(pages)} markdown pages, {len(collections)} collection indexes, "
          f"{len(ROBOTICS_SECTIONS) + 1} robotics indexes, {len(ml_lessons)} ml-compilers lessons.")
    stale = stale_generated_pages(pages, collections)
    if stale:
        print("Generated pages with no markdown source (review, then delete):")
        for path in stale:
            print("  " + rel(path))


if __name__ == "__main__":
    main()
