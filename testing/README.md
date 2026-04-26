Regression checks for the robotics site builder.

Run the focused builder regression test with:

python3 -m unittest testing/test_build_robotics.py

Current coverage includes generated-page math rendering support for robotics lessons.
This guards against silent regressions where markdown leaves LaTeX content as plain paragraph text instead of a dedicated math block.
