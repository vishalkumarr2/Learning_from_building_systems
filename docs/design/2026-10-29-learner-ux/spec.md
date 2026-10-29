status: for_review

open_items: |
  - Owner to delete five files (deletion was blocked for the agent): the two
    duplicate navigation-estimator pages whose file names carry internal names
    (exercises/05-*-specific.* and 06-ceres-*-deep-dive.* duplicates, md + html),
    llm-to-vla/learning-plan.html (superseded by 00-learning-plan.html) and
    robotics/learn/zephyr/deep-dive/README.html (superseded by deep-dive/index.html).
    Then rerun build_site.py and check_site.py; both must report OK.
  - Optimization weeks 5-10 (42 lessons) are not written; links to them now say so.
  - Git history still holds the internal names; rewriting it is the owner's call.

charter: |
  Learners hit broken navigation, unpublished lessons, raw markdown inside answer
  blocks and a layout that breaks on phones. Internal product references that an
  earlier scrub (14a5d42) removed were re-added by a later sync (7cd7424).

contract: |
  - Every .md under a curriculum tree has a rendered .html sibling, built by one
    script (tools/build_site.py) from the markdown, so the two cannot drift.
  - Zero broken relative links and zero orphan pages across the site.
  - Markdown inside <details> renders (md_in_html).
  - Every lesson page: back link, prev/next footer, auto table of contents on long
    pages, "mark as done" progress saved in the browser; track landing pages show
    progress and a "continue where you left off" link.
  - Homepage and lesson pages have no horizontal scroll at 390 px width.
  - Homepage cards lead to real content; empty areas are labelled as such.
  - Repo-wide scan for the internal product, hardware and customer names finds none.

source: |
  tools/build_robotics.py (existing generator, page shell, link rewriting)
  optimization/ and llm-to-vla/ generated pages (same shell, KaTeX, footer prev/next)
  14a5d42 diff (replacement vocabulary for the scrub)

history: |
  14a5d42 vishalkumarr2 2026-04-22: scrub internal references from public content.
  7cd7424 vishalkumarr2 2026-05-13: "sync all markdown content" re-added them.
  85b15b1 mentions a converter script for optimization; it was never committed,
  which is why only robotics HTML can be rebuilt today.

verify: |
  python3 tools/build_site.py && python3 -m unittest discover -s testing
  python3 tools/check_site.py   -> 0 broken links, 0 orphans, 0 md without html,
                                   0 raw markdown, 0 internal-term hits
  Playwright at 390 px: document scrollWidth <= viewport on homepage and a lesson.

packages:
  - A scrub: replace internal names, strip internal source identifiers; check_site term scan is clean.
  - B generator: tools/build_site.py renders all curriculum markdown; tests pass.
  - C learner assets: assets/learn.css + assets/learn.js (TOC, progress, resume, overflow).
  - D html-only pages: ml-compilers link fixes and asset injection.
  - E homepage, area stubs, README.
  - F check_site.py link/coverage/term audit, runnable before every push.

decisions:
  - context: how markdown becomes HTML
    options: generalise build_robotics.py | MkDocs | patch HTML in place
    decision: generalise build_robotics.py (user, 2026-10-29)
    consequences: no new dependency, current look kept; ml-compilers stays HTML-only.
  - context: scrub scope
    options: re-apply April vocabulary | also strip internal source identifiers | leave
    decision: user delegated ("whatever is the best"); chose re-apply vocabulary and
      replace internal source identifiers with generic descriptions. Git history is
      not rewritten; that remains a separate decision for the owner.
    consequences: older commits still contain the terms.
