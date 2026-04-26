import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_robotics.py"
SPEC = importlib.util.spec_from_file_location("build_robotics", MODULE_PATH)
build_robotics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_robotics)


class BuildRoboticsTests(unittest.TestCase):
    def test_convert_md_to_html_includes_mathjax_support(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            md_path = temp_dir / "sample.md"
            out_path = temp_dir / "sample.html"

            md_path.write_text(
                "# Sample\n\nInline math $x$.\n\n$$\nq = (x, y, \\theta)\n$$\n",
                encoding="utf-8",
            )

            build_robotics.convert_md_to_html(
                md_path,
                out_path,
                "index.html",
                "Back",
                "blue",
            )

            html = out_path.read_text(encoding="utf-8")

            self.assertIn("MathJax", html)
            self.assertIn("tex-mml-chtml.js", html)
            self.assertIn("inlineMath", html)
            self.assertIn("displayMath", html)
            self.assertIn("<div class=\"math-display\">", html)
            self.assertIn("$x$", html)
            self.assertIn("$$", html)
            self.assertNotIn("<p>$$", html)


if __name__ == "__main__":
    unittest.main()