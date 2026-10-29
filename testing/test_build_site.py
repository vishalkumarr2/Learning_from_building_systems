import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_site.py"
SPEC = importlib.util.spec_from_file_location("build_site", MODULE_PATH)
build_site = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = build_site
SPEC.loader.exec_module(build_site)

ROOT = build_site.SITE_ROOT


def render_robotics_page(md_text):
    body = build_site.render_markdown(md_text)
    page_path = ROOT / "robotics/learn/zephyr/sample.html"
    return build_site.html_page("Sample", body, page_path, build_site.ROBOTICS_TRACK, ("index.html", "Back"))


class RenderTests(unittest.TestCase):
    def test_robotics_page_includes_mathjax_support(self):
        html = render_robotics_page("# Sample\n\nInline math $x$.\n\n$$\nq = (x, y, \\theta)\n$$\n")

        self.assertIn("MathJax", html)
        self.assertIn("tex-mml-chtml.js", html)
        self.assertIn("inlineMath", html)
        self.assertIn("displayMath", html)
        self.assertIn("<div class=\"math-display\">", html)
        self.assertIn("$x$", html)
        self.assertIn("$$", html)
        self.assertNotIn("<p>$$", html)

    def test_markdown_inside_details_is_rendered(self):
        body = build_site.render_markdown(
            "<details><summary>Answer</summary>\n\nUse **west** and `west update`.\n\n</details>\n")

        self.assertIn("<strong>west</strong>", body)
        self.assertIn("<code>west update</code>", body)
        self.assertNotIn("**west**", body)
        self.assertRegex(body, r"<details[^>]*>\s*<summary>Answer</summary>")

    def test_underscores_in_math_are_not_turned_into_emphasis(self):
        body = build_site.render_markdown("Compare $a_i$ with $b_i$ here.\n")

        self.assertIn("$a_i$", body)
        self.assertNotIn("<em>", body)

    def test_dollar_amounts_in_code_are_left_alone(self):
        body = build_site.render_markdown("Run `echo $HOME $PATH` now.\n")

        self.assertIn("<code>echo $HOME $PATH</code>", body)

    def test_heading_ids_ignore_math_placeholders(self):
        body = build_site.render_markdown("## Why scale by $\\sqrt{d_k}$?\n")

        self.assertNotIn("mathph", body.lower())
        self.assertIn('data-plain="Why scale by $\\sqrt{d_k}$?"', body)

    def test_fence_nested_in_list_item_renders_as_code(self):
        body = build_site.render_markdown("1. **Try it:** run this\n   ```python\n   x = 1\n   ```\n")

        self.assertIn("<code", body)
        self.assertNotIn("```", body)

    def test_fence_nested_in_list_inside_details_renders_as_code(self):
        body = build_site.render_markdown(
            "<details><summary>Answer</summary>\n\n1. **Detect:**\n   ```python\n   # Check:\n   x = 1\n\n   y = 2\n   ```\n\n</details>\n")

        self.assertIn("<pre", body)
        self.assertNotIn("```", body)
        self.assertNotIn("<h1", body)


class OutputPathTests(unittest.TestCase):
    def test_readme_becomes_folder_index(self):
        md = ROOT / "llm-to-vla/weeks/week-02/README.md"
        self.assertEqual(build_site.output_path(md, set()), md.parent / "index.html")

    def test_readme_in_card_section_keeps_its_own_name(self):
        md = ROOT / "robotics/learn/zephyr/README.md"
        self.assertEqual(build_site.output_path(md, {md.parent}), md.parent / "readme.html")

    def test_upper_case_names_are_lowercased(self):
        md = ROOT / "llm-to-vla/CURRICULUM.md"
        self.assertEqual(build_site.output_path(md, set()), md.parent / "curriculum.html")

    def test_hyphenated_zephyr_tutorial_keeps_its_name(self):
        md = ROOT / "robotics/learn/zephyr/01-zephyr-intro.md"
        self.assertEqual(build_site.output_path(md, set()), md.parent / "01-zephyr-intro.html")


class LinkRewriteTests(unittest.TestCase):
    def test_links_to_readme_and_curriculum_point_at_generated_pages(self):
        lesson = ROOT / "llm-to-vla/weeks/week-02/day-11.md"
        page = build_site.Page(lesson, lesson.with_suffix(".html"), build_site.GENERIC_TRACKS[1],
                               "Day 11", True, lesson.parent)
        body = '<a href="../README.md">Week</a> <a href="../../CURRICULUM.md#phase-2">All</a>'

        rewritten = build_site.rewrite_links(body, page, {}, set())

        self.assertIn('href="../index.html"', rewritten)
        self.assertIn('href="../../curriculum.html#phase-2"', rewritten)


if __name__ == "__main__":
    unittest.main()
