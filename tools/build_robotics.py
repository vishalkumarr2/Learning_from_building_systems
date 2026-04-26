#!/usr/bin/env python3
"""
Convert robotics/ markdown files to HTML matching the site's existing style.

Generates:
  - robotics/index.html           (landing page with cards for each section)
  - robotics/<section>/index.html (section pages with cards for each topic)
  - robotics/<path>/notes.html    (individual content pages)

Uses the same Tailwind CDN + Inter font + card styling as the rest of the site.
"""

import os
import re
import sys
from pathlib import Path

import markdown

SITE_ROOT = Path(__file__).resolve().parent.parent
ROBOTICS = SITE_ROOT / "robotics"
LEARN = ROBOTICS / "learn"

# Color scheme per section (matches the existing site's use of colored headings)
SECTION_COLORS = {
    "cpp-advanced": {"accent": "indigo", "hex": "#4338ca"},
    "zephyr": {"accent": "emerald", "hex": "#047857"},
    "electronics": {"accent": "amber", "hex": "#b45309"},
    "control-systems": {"accent": "rose", "hex": "#be123c"},
    "logic": {"accent": "cyan", "hex": "#0891b2"},
}

# ── HTML template for content pages ──────────────────────────────────────────

def html_page(title, body_html, back_href, back_label, accent_class="blue"):
    """Wrap rendered markdown in the site's standard page shell."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{esc(title)} - Learning From Building Systems</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }}
        }};
    </script>
    <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; }}
        .card {{
            background-color: white;
            border-radius: 0.75rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }}
        /* Markdown content styling */
        .md-content h1 {{ font-size: 2rem; font-weight: 700; color: #111827; margin-top: 2rem; margin-bottom: 1rem; }}
        .md-content h2 {{ font-size: 1.5rem; font-weight: 600; color: #1f2937; margin-top: 1.75rem; margin-bottom: 0.75rem; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.5rem; }}
        .md-content h3 {{ font-size: 1.25rem; font-weight: 600; color: #374151; margin-top: 1.5rem; margin-bottom: 0.5rem; }}
        .md-content h4 {{ font-size: 1.1rem; font-weight: 600; color: #4b5563; margin-top: 1.25rem; margin-bottom: 0.5rem; }}
        .md-content p {{ margin-bottom: 1rem; line-height: 1.75; color: #374151; }}
        .md-content ul {{ list-style: disc; margin-left: 1.5rem; margin-bottom: 1rem; color: #374151; }}
        .md-content ol {{ list-style: decimal; margin-left: 1.5rem; margin-bottom: 1rem; color: #374151; }}
        .md-content li {{ margin-bottom: 0.35rem; line-height: 1.6; }}
        .md-content code {{
            background-color: #f3f4f6; padding: 0.15rem 0.4rem; border-radius: 0.25rem;
            font-size: 0.875rem; font-family: 'Fira Code', 'Courier New', monospace; color: #be185d;
        }}
        .md-content pre {{
            background-color: #1f2937; color: #e5e7eb; padding: 1rem; border-radius: 0.5rem;
            overflow-x: auto; margin-bottom: 1.25rem; font-size: 0.85rem; line-height: 1.6;
        }}
        .md-content pre code {{ background: none; padding: 0; color: inherit; font-size: inherit; }}
        .md-content blockquote {{
            border-left: 4px solid #6366f1; background: #eef2ff; padding: 0.75rem 1rem;
            margin-bottom: 1rem; border-radius: 0 0.5rem 0.5rem 0; color: #374151;
        }}
        .md-content table {{ width: 100%; border-collapse: collapse; margin-bottom: 1.25rem; }}
        .md-content th {{ background: #f9fafb; padding: 0.5rem 0.75rem; border: 1px solid #e5e7eb; text-align: left; font-weight: 600; font-size: 0.875rem; }}
        .md-content td {{ padding: 0.5rem 0.75rem; border: 1px solid #e5e7eb; font-size: 0.875rem; }}
        .md-content tr:hover {{ background-color: #f9fafb; }}
        .md-content hr {{ border: 0; border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }}
        .md-content strong {{ font-weight: 600; }}
        .md-content a {{ color: #2563eb; text-decoration: underline; }}
        .md-content a:hover {{ color: #1d4ed8; }}
        .md-content img {{ max-width: 100%; border-radius: 0.5rem; margin: 1rem 0; }}
        .md-content .mermaid {{ margin: 1.5rem 0; overflow-x: auto; }}
        .md-content .math-display {{ margin: 1.5rem 0; overflow-x: auto; }}
    </style>
</head>
<body class="bg-gray-100 text-gray-800">
    <div class="container mx-auto p-4 sm:p-6 lg:p-8 max-w-4xl">
        <a href="{back_href}" class="text-{accent_class}-600 hover:text-{accent_class}-800 mb-4 inline-block">&larr; {esc(back_label)}</a>

        <div class="bg-white rounded-xl shadow-md p-6 sm:p-8 md:p-10">
            <div class="md-content">
{body_html}
            </div>
        </div>

        <footer class="text-center mt-16 py-6 border-t border-gray-300">
            <p class="text-gray-500">Robotics Learning Resources</p>
        </footer>
    </div>
    <script>
        mermaid.initialize({{ startOnLoad: true }});
    </script>
</body>
</html>"""


def index_page(title, subtitle, cards_html, back_href, back_label, accent="blue"):
    """Generate a section landing page with cards."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{esc(title)} - Learning From Building Systems</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; }}
        .card {{
            background-color: white;
            border-radius: 0.75rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }}
    </style>
</head>
<body class="bg-gray-100 text-gray-800">
    <div class="container mx-auto p-4 sm:p-6 lg:p-8">
        <a href="{back_href}" class="text-blue-600 hover:text-blue-800 mb-4 inline-block">&larr; {esc(back_label)}</a>

        <header class="text-center mb-12">
            <h1 class="text-4xl md:text-5xl font-bold text-gray-900">{esc(title)}</h1>
            <p class="mt-4 text-lg text-gray-600 max-w-3xl mx-auto">{esc(subtitle)}</p>
        </header>

        <main>
            <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
{cards_html}
            </div>
        </main>

        <footer class="text-center mt-16 py-6 border-t border-gray-300">
            <p class="text-gray-500">Robotics Learning Resources</p>
        </footer>
    </div>
</body>
</html>"""


def card_html(href, title, description, accent="green"):
    return f"""                <a href="{href}" class="card block p-6">
                    <h3 class="font-bold text-xl mb-2 text-{accent}-700">{esc(title)}</h3>
                    <p class="text-gray-600">{esc(description)}</p>
                </a>"""


# ── Utilities ────────────────────────────────────────────────────────────────

def esc(text):
    """Escape HTML entities in text."""
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


def extract_title(md_text):
    """Extract the first H1 from markdown, or fall back to first line."""
    match = re.match(r"^#\s+(.+)", md_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    # fall back to first non-empty line
    for line in md_text.splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return "Untitled"


def extract_subtitle(md_text):
    """Extract the first H3 or paragraph as a subtitle."""
    for line in md_text.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("###"):
            return line.lstrip("#").strip()
        if line.startswith("**") or line[0].isalpha():
            return line[:120]
    return ""


def slug_to_title(slug):
    """Convert directory slug like '01-move-semantics-value-categories' to title."""
    # Strip leading number prefix
    slug = re.sub(r"^\d+-", "", slug)
    slug = re.sub(r"^\d+_", "", slug)
    return slug.replace("-", " ").replace("_", " ").title()


def preprocess_markdown(md_text):
    """Expand markdown features the plain renderer doesn't handle well."""
    display_math_pattern = re.compile(r"^\$\$\s*\n(.*?)\n\$\$\s*$", re.MULTILINE | re.DOTALL)
    mermaid_pattern = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)

    def display_math_replacer(match):
        expression = match.group(1).strip()
        return f'\n<div class="math-display">\n$$\n{expression}\n$$\n</div>\n'

    def mermaid_replacer(match):
        diagram = match.group(1).strip()
        return f'\n<div class="mermaid">\n{diagram}\n</div>\n'

    md_text = display_math_pattern.sub(display_math_replacer, md_text)
    return mermaid_pattern.sub(mermaid_replacer, md_text)


def rewrite_markdown_links(body_html):
    """Rewrite internal markdown links to generated HTML paths."""
    href_pattern = re.compile(r'href="(?![a-z]+://|//|#)([^"]+?)\.md(#.*?)?"', re.IGNORECASE)

    def href_replacer(match):
        target = match.group(1)
        fragment = match.group(2) or ""
        return f'href="{target}.html{fragment}"'

    return href_pattern.sub(href_replacer, body_html)


def convert_md_to_html(md_path, out_path, back_href, back_label, accent="blue"):
    """Convert a single markdown file to a styled HTML page."""
    md_text = md_path.read_text(encoding="utf-8")
    md_text = preprocess_markdown(md_text)
    title = extract_title(md_text)

    md_ext = markdown.Markdown(extensions=[
        "tables", "fenced_code", "codehilite", "toc",
        "sane_lists", "smarty",
    ], extension_configs={
        "codehilite": {"css_class": "highlight", "guess_lang": False},
    })
    body = md_ext.convert(md_text)
    body = rewrite_markdown_links(body)

    html = html_page(title, body, back_href, back_label, accent)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return title


# ── Section builders ─────────────────────────────────────────────────────────

def build_cpp_advanced():
    """Build C++ Advanced section pages."""
    section = LEARN / "cpp-advanced"
    accent = "indigo"
    cards = []

    # README and STUDY-PLAN as top-level pages
    for name, label, desc in [
        ("README.md", "Overview", "Course overview and structure"),
        ("STUDY-PLAN.md", "Study Plan", "8-week study plan with daily schedule"),
    ]:
        md = section / name
        if md.exists():
            out_name = name.replace(".md", ".html").lower()
            out = section / out_name
            convert_md_to_html(md, out, "index.html", "Back to C++ Advanced", accent)
            cards.append(card_html(out_name, label, desc, accent))

    # Module directories (01-*, 02-*, ...)
    modules = sorted([d for d in section.iterdir()
                      if d.is_dir() and re.match(r"\d{2}-", d.name)])

    for mod in modules:
        notes = mod / "notes.md"
        readme = mod / "README.md"
        md = notes if notes.exists() else (readme if readme.exists() else None)
        if md is None:
            continue

        md_text = md.read_text(encoding="utf-8")
        title = extract_title(md_text)
        subtitle = extract_subtitle(md_text)
        out = mod / "index.html"
        convert_md_to_html(md, out, "../index.html", "Back to C++ Advanced", accent)
        rel = f"{mod.name}/index.html"
        cards.append(card_html(rel, title, subtitle[:100] if subtitle else slug_to_title(mod.name), accent))

    # Section index
    idx = index_page(
        "Advanced C++ for Robotics",
        "Real-time, safety-critical & production systems programming",
        "\n".join(cards),
        "../index.html", "Back to Robotics",
        accent,
    )
    (section / "index.html").write_text(idx, encoding="utf-8")
    return len(cards)


def build_zephyr():
    """Build Zephyr RTOS section pages."""
    section = LEARN / "zephyr"
    accent = "emerald"
    cards = []

    # Top-level files
    for name, label, desc in [
        ("README.md", "Overview", "Zephyr learning path overview"),
        ("00-mastery-plan.md", "Mastery Plan", "Complete learning roadmap"),
    ]:
        md = section / name
        if md.exists():
            out_name = name.replace(".md", ".html").lower()
            out = section / out_name
            convert_md_to_html(md, out, "index.html", "Back to Zephyr", accent)
            cards.append(card_html(out_name, label, desc, accent))

    # Numbered tutorials (01_zephyr_intro.md, etc.)
    tutorials = sorted([f for f in section.iterdir()
                        if f.is_file() and f.suffix == ".md"
                        and re.match(r"\d{2}_", f.name)])

    for md in tutorials:
        md_text = md.read_text(encoding="utf-8")
        title = extract_title(md_text)
        out_name = md.stem + ".html"
        out = section / out_name
        convert_md_to_html(md, out, "index.html", "Back to Zephyr", accent)
        cards.append(card_html(out_name, title, "", accent))

    # Exercises subdirectory
    exercises_dir = section / "exercises"
    if exercises_dir.exists():
        ex_cards = []
        for md in sorted(exercises_dir.glob("*.md")):
            md_text = md.read_text(encoding="utf-8")
            title = extract_title(md_text)
            out = exercises_dir / (md.stem + ".html")
            convert_md_to_html(md, out, "../index.html", "Back to Zephyr", accent)
            ex_cards.append(card_html(f"exercises/{md.stem}.html", f"Exercise: {title}", "", accent))
        cards.extend(ex_cards)

    # Study notes subdirectory
    notes_dir = section / "study-notes"
    if notes_dir.exists():
        for md in sorted(notes_dir.glob("*.md")):
            md_text = md.read_text(encoding="utf-8")
            title = extract_title(md_text)
            out = notes_dir / (md.stem + ".html")
            convert_md_to_html(md, out, "../index.html", "Back to Zephyr", accent)
            cards.append(card_html(f"study-notes/{md.stem}.html", f"Notes: {title}", "", accent))

    # Deep-dive subdirectory
    deepdive_dir = section / "deep-dive"
    if deepdive_dir.exists():
        for md in sorted(deepdive_dir.glob("*.md")):
            md_text = md.read_text(encoding="utf-8")
            title = extract_title(md_text)
            out = deepdive_dir / (md.stem + ".html")
            convert_md_to_html(md, out, "../index.html", "Back to Zephyr", accent)
            cards.append(card_html(f"deep-dive/{md.stem}.html", f"Deep Dive: {title}", "", accent))

    idx = index_page(
        "Zephyr RTOS",
        "From basics to production firmware with Zephyr, DMA, SPI, CAN, and ROS2 integration",
        "\n".join(cards),
        "../index.html", "Back to Robotics",
        accent,
    )
    (section / "index.html").write_text(idx, encoding="utf-8")
    return len(cards)


def build_electronics():
    """Build Electronics section pages."""
    section = LEARN / "electronics"
    accent = "amber"
    cards = []

    # Learning plan
    md = section / "00-learning-plan.md"
    if md.exists():
        out = section / "00-learning-plan.html"
        convert_md_to_html(md, out, "index.html", "Back to Electronics", accent)
        cards.append(card_html("00-learning-plan.html", "Learning Plan", "Electronics learning roadmap", accent))

    # Numbered lessons
    lessons = sorted([f for f in section.iterdir()
                      if f.is_file() and f.suffix == ".md"
                      and re.match(r"\d{2}-", f.name)])

    for md in lessons:
        md_text = md.read_text(encoding="utf-8")
        title = extract_title(md_text)
        out_name = md.stem + ".html"
        out = section / out_name
        convert_md_to_html(md, out, "index.html", "Back to Electronics", accent)
        cards.append(card_html(out_name, title, "", accent))

    # Exercises
    exercises_dir = section / "exercises"
    if exercises_dir.exists():
        for md in sorted(exercises_dir.glob("*.md")):
            md_text = md.read_text(encoding="utf-8")
            title = extract_title(md_text)
            out = exercises_dir / (md.stem + ".html")
            convert_md_to_html(md, out, "../index.html", "Back to Electronics", accent)
            cards.append(card_html(f"exercises/{md.stem}.html", f"Exercise: {title}", "", accent))

    idx = index_page(
        "Electronics Fundamentals",
        "Passive components, semiconductors, op-amps, communication protocols (UART, SPI, I2C, CAN)",
        "\n".join(cards),
        "../index.html", "Back to Robotics",
        accent,
    )
    (section / "index.html").write_text(idx, encoding="utf-8")
    return len(cards)


def build_control_systems():
    """Build Control Systems section pages."""
    section = LEARN / "control-systems"
    accent = "rose"
    cards = []

    # Learning plan
    md = section / "00-learning-plan.md"
    if md.exists():
        out = section / "00-learning-plan.html"
        convert_md_to_html(md, out, "index.html", "Back to Control Systems", accent)
        cards.append(card_html("00-learning-plan.html", "Learning Plan", "Full curriculum overview — 10 lessons + exercises + debugging", accent))

    # Numbered lessons (01-*.md through 10-*.md)
    lessons = sorted([f for f in section.iterdir()
                      if f.is_file() and f.suffix == ".md"
                      and re.match(r"\d{2}-", f.name)
                      and f.name != "00-learning-plan.md"])

    for md in lessons:
        md_text = md.read_text(encoding="utf-8")
        title = extract_title(md_text)
        out_name = md.stem + ".html"
        out = section / out_name
        convert_md_to_html(md, out, "index.html", "Back to Control Systems", accent)
        cards.append(card_html(out_name, title, "", accent))

    # Exercises
    exercises_dir = section / "exercises"
    if exercises_dir.exists():
        for md in sorted(exercises_dir.glob("*.md")):
            md_text = md.read_text(encoding="utf-8")
            title = extract_title(md_text)
            out = exercises_dir / (md.stem + ".html")
            convert_md_to_html(md, out, "../index.html", "Back to Control Systems", accent)
            cards.append(card_html(f"exercises/{md.stem}.html", f"Exercise: {title}", "", accent))

    # Debugging sessions
    debugging_dir = section / "debugging"
    if debugging_dir.exists():
        for md in sorted(debugging_dir.glob("*.md")):
            md_text = md.read_text(encoding="utf-8")
            title = extract_title(md_text)
            out = debugging_dir / (md.stem + ".html")
            convert_md_to_html(md, out, "../index.html", "Back to Control Systems", accent)
            cards.append(card_html(f"debugging/{md.stem}.html", f"Debug: {title}", "", accent))

    # Senior questions
    senior_dir = section / "senior-questions"
    if senior_dir.exists():
        for md in sorted(senior_dir.glob("*.md")):
            md_text = md.read_text(encoding="utf-8")
            title = extract_title(md_text)
            out = senior_dir / (md.stem + ".html")
            convert_md_to_html(md, out, "../index.html", "Back to Control Systems", accent)
            cards.append(card_html(f"senior-questions/{md.stem}.html", f"Senior Q: {title}", "", accent))

    idx = index_page(
        "Control Systems",
        "PID to advanced control: MCU inner loop to Jetson outer loop — 10 lessons + exercises + debugging",
        "\n".join(cards),
        "../index.html", "Back to Robotics",
        accent,
    )
    (section / "index.html").write_text(idx, encoding="utf-8")
    return len(cards)


def build_logic():
    """Build Logic & Critical Reasoning section pages."""
    section = LEARN / "logic"
    accent = "cyan"
    cards = []

    # Learning plan
    md = section / "00-learning-plan.md"
    if md.exists():
        out = section / "00-learning-plan.html"
        convert_md_to_html(md, out, "index.html", "Back to Logic", accent)
        cards.append(card_html("00-learning-plan.html", "Learning Plan", "9-lesson curriculum overview with engineering connections", accent))

    # Numbered lessons
    lessons = sorted([f for f in section.iterdir()
                      if f.is_file() and f.suffix == ".md"
                      and re.match(r"\d{2}-", f.name)
                      and f.name != "00-learning-plan.md"])

    for md in lessons:
        md_text = md.read_text(encoding="utf-8")
        title = extract_title(md_text)
        out_name = md.stem + ".html"
        out = section / out_name
        convert_md_to_html(md, out, "index.html", "Back to Logic", accent)
        cards.append(card_html(out_name, title, "", accent))

    # Exercises
    exercises_dir = section / "exercises"
    if exercises_dir.exists():
        for md in sorted(exercises_dir.glob("*.md")):
            md_text = md.read_text(encoding="utf-8")
            title = extract_title(md_text)
            out = exercises_dir / (md.stem + ".html")
            convert_md_to_html(md, out, "../index.html", "Back to Logic", accent)
            cards.append(card_html(f"exercises/{md.stem}.html", f"Exercise: {title}", "", accent))

    idx = index_page(
        "Logic & Critical Reasoning",
        "Arguments, fallacies, propositional logic, deduction, scientific reasoning — 9 lessons + exercises",
        "\n".join(cards),
        "../index.html", "Back to Robotics",
        accent,
    )
    (section / "index.html").write_text(idx, encoding="utf-8")
    return len(cards)


def build_study_plan():
    """Build the top-level STUDY-PLAN page."""
    md = LEARN / "STUDY-PLAN.md"
    if md.exists():
        out = LEARN / "study-plan.html"
        convert_md_to_html(md, out, "../index.html", "Back to Robotics", "blue")
        return True
    return False


def build_navigation_estimator():
    """Build Navigation Estimator section pages."""
    section = LEARN / "navigation-estimator"
    accent = "sky"
    cards = []

    # Numbered lessons
    lessons = sorted([f for f in section.iterdir()
                      if f.is_file() and f.suffix == ".md"
                      and re.match(r"\d{2}-", f.name)])
    for md in lessons:
        md_text = md.read_text(encoding="utf-8")
        title = extract_title(md_text)
        out = section / (md.stem + ".html")
        convert_md_to_html(md, out, "index.html", "Back to Nav Estimator", accent)
        cards.append(card_html(md.stem + ".html", title, "", accent))

    # Exercises subdirectory
    exercises_dir = section / "exercises"
    if exercises_dir.exists():
        for md in sorted(exercises_dir.glob("*.md")):
            md_text = md.read_text(encoding="utf-8")
            title = extract_title(md_text)
            out = exercises_dir / (md.stem + ".html")
            convert_md_to_html(md, out, "../index.html", "Back to Nav Estimator", accent)
            cards.append(card_html(f"exercises/{md.stem}.html", f"Exercise: {title}", "", accent))

    idx = index_page(
        "Navigation Estimator",
        "Dead reckoning, Kalman filters, IMU fusion, failure modes, and Ceres optimization",
        "\n".join(cards),
        "../index.html", "Back to Robotics",
        accent,
    )
    (section / "index.html").write_text(idx, encoding="utf-8")
    return len(cards)


def build_ros2_handson():
    """Build ROS 2 Hands-On section pages."""
    section = LEARN / "ros2-handson"
    accent = "violet"
    cards = []

    lessons = sorted([f for f in section.iterdir()
                      if f.is_file() and f.suffix == ".md"
                      and re.match(r"\d{2}-", f.name)])
    for md in lessons:
        md_text = md.read_text(encoding="utf-8")
        title = extract_title(md_text)
        out = section / (md.stem + ".html")
        convert_md_to_html(md, out, "index.html", "Back to ROS 2", accent)
        cards.append(card_html(md.stem + ".html", title, "", accent))

    exercises_dir = section / "exercises"
    if exercises_dir.exists():
        for md in sorted(exercises_dir.glob("*.md")):
            md_text = md.read_text(encoding="utf-8")
            title = extract_title(md_text)
            out = exercises_dir / (md.stem + ".html")
            convert_md_to_html(md, out, "../index.html", "Back to ROS 2 Hands-on", accent)
            cards.append(card_html(f"exercises/{md.stem}.html", f"Exercise: {title}", "", accent))

    idx = index_page(
        "ROS 2 Hands-On",
        "Nodes, topics, actions, TF2, Nav2, and QoS — practical ROS 2 development",
        "\n".join(cards),
        "../index.html", "Back to Robotics",
        accent,
    )
    (section / "index.html").write_text(idx, encoding="utf-8")
    return len(cards)


def build_python_scripting():
    """Build Python Scripting section pages."""
    section = LEARN / "python-scripting"
    accent = "teal"
    cards = []

    lessons = sorted([f for f in section.iterdir()
                      if f.is_file() and f.suffix == ".md"
                      and re.match(r"\d{2}-", f.name)])
    for md in lessons:
        md_text = md.read_text(encoding="utf-8")
        title = extract_title(md_text)
        out = section / (md.stem + ".html")
        convert_md_to_html(md, out, "index.html", "Back to Python Scripting", accent)
        cards.append(card_html(md.stem + ".html", title, "", accent))

    # Exercises subdirectory
    exercises_dir = section / "exercises"
    if exercises_dir.exists():
        for md in sorted(exercises_dir.glob("*.md")):
            md_text = md.read_text(encoding="utf-8")
            title = extract_title(md_text)
            out = exercises_dir / (md.stem + ".html")
            convert_md_to_html(md, out, "../index.html", "Back to Python Scripting", accent)
            cards.append(card_html(f"exercises/{md.stem}.html", f"Exercise: {title}", "", accent))

    idx = index_page(
        "Python Scripting",
        "Type-safe Python, testing, CLI scripts, and timeseries analysis for robotics",
        "\n".join(cards),
        "../index.html", "Back to Robotics",
        accent,
    )
    (section / "index.html").write_text(idx, encoding="utf-8")
    return len(cards)


def build_robotics_index():
    """Build the top-level robotics/index.html landing page."""
    cards = []

    cards.append(card_html("learn/study-plan.html",
                           "Unified Study Plan",
                           "120–130 hour roadmap: Electronics → Protocols → Embedded Systems",
                           "blue"))
    cards.append(card_html("learn/resources.html",
                           "External Resources",
                           "Curated courses, papers, repos, and videos for each learning track",
                           "cyan"))
    cards.append(card_html("learn/gaps-roadmap.html",
                           "Gaps & Roadmap",
                           "Learning gap analysis and prioritised improvement roadmap",
                           "orange"))
    cards.append(card_html("learn/cpp-advanced/index.html",
                           "Advanced C++ for Robotics",
                           "Real-time, safety-critical & production systems — 18 modules",
                           "indigo"))
    cards.append(card_html("learn/zephyr/index.html",
                           "Zephyr RTOS",
                           "Embedded firmware: Zephyr, DMA, SPI, CAN, ROS2 bridge — 17 tutorials",
                           "emerald"))
    cards.append(card_html("learn/electronics/index.html",
                           "Electronics Fundamentals",
                           "Passive components through CAN bus — 7 deep-dive lessons + exercises",
                           "amber"))
    cards.append(card_html("learn/control-systems/index.html",
                           "Control Systems",
                           "PID to advanced control: MCU inner loop to Jetson outer loop — 10 lessons + exercises",
                           "rose"))
    cards.append(card_html("learn/navigation-estimator/index.html",
                           "Navigation Estimator",
                           "Dead reckoning, Kalman filters, IMU fusion, and failure diagnosis",
                           "sky"))
    cards.append(card_html("learn/ros2-handson/index.html",
                           "ROS 2 Hands-On",
                           "Nodes, topics, TF2, Nav2, QoS — practical ROS 2 development",
                           "violet"))
    cards.append(card_html("learn/python-scripting/index.html",
                           "Python Scripting",
                           "Type-safe Python, testing, CLI tools, and timeseries analysis",
                           "teal"))
    cards.append(card_html("learn/logic/index.html",
                           "Logic & Critical Reasoning",
                           "Arguments, fallacies, propositional logic, deduction, scientific reasoning — 9 lessons + exercises",
                           "cyan"))

    idx = index_page(
        "Robotics",
        "Embedded systems, real-time C++, Zephyr RTOS, electronics, and protocol deep-dives",
        "\n".join(cards),
        "../index.html", "Back to Home",
        "blue",
    )
    (ROBOTICS / "index.html").write_text(idx, encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building robotics HTML pages...")

    n1 = build_cpp_advanced()
    print(f"  C++ Advanced: {n1} pages")

    n2 = build_zephyr()
    print(f"  Zephyr RTOS:  {n2} pages")

    n3 = build_electronics()
    print(f"  Electronics:  {n3} pages")

    n4 = build_control_systems()
    print(f"  Control Sys:  {n4} pages")

    n5 = build_navigation_estimator()
    print(f"  Nav Estimator: {n5} pages")

    n6 = build_ros2_handson()
    print(f"  ROS 2:        {n6} pages")

    n7 = build_python_scripting()
    print(f"  Python:       {n7} pages")

    n8 = build_logic()
    print(f"  Logic:        {n8} pages")

    build_study_plan()
    print("  Study Plan:   1 page")

    build_robotics_index()
    print("  Robotics index: 1 page")

    total = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + 2  # +2 for study plan + robotics index
    print(f"\nDone! Generated {total} HTML files.")
    print("Next: add Robotics link to index.html sidebar and category grid.")


if __name__ == "__main__":
    main()
