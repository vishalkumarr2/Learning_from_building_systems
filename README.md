# Learn Module

Self-study curriculum covering electronics, embedded systems, robotics, control theory, optimization, and ML — built from real OKS AMR engineering work.

Served as a [Docsify](https://docsify.js.org/) site locally via `index.html`.

## Publishing to GitHub

The public-facing version of this module lives in a separate repo:

**Repo:** [vishalkumarr2/Learning_from_building_systems](https://github.com/vishalkumarr2/Learning_from_building_systems)

### Content mapping

| Source (this repo)             | Target (public repo)                   |
|-------------------------------|----------------------------------------|
| `learn/*` (root files)         | `robotics/learn/*`                     |
| `learn/<track>/`               | `robotics/learn/<track>/`              |
| `learn/llm-to-vla/`           | `llm-to-vla/` (repo root)             |
| `learn/optimization/`         | `optimization/` (repo root)           |

### How to sync

```bash
# 1. Clone the target repo
git clone git@github.com:vishalkumarr2/Learning_from_building_systems.git /tmp/lbs
cd /tmp/lbs

# 2. Sync modules that live under robotics/learn/
for track in control-systems cpp-advanced electronics navigation-estimator ros2-handson zephyr; do
  rsync -av --include='*.md' --include='*/' --exclude='*.html' \
    /home/viku/ai/issueanalyser/learn/$track/ robotics/learn/$track/
done

# 3. Sync root-level markdown files
cp /home/viku/ai/issueanalyser/learn/{GAPS-ROADMAP,STUDY-PLAN,RESOURCES}.md robotics/learn/

# 4. Sync modules that live at repo root
rsync -av --include='*.md' --include='*/' --exclude='_planning/' --exclude='*.html' \
  /home/viku/ai/issueanalyser/learn/llm-to-vla/ llm-to-vla/
rsync -av --include='*.md' --include='*/' --exclude='_planning/' --exclude='*.html' \
  /home/viku/ai/issueanalyser/learn/optimization/ optimization/

# 5. Commit and push
git add -A
git commit -m "fix: sync learn module content"
git push origin main

# 6. Clean up
rm -rf /tmp/lbs
```

### What NOT to sync

- `_planning/` directories — internal planning artifacts, not for public repo
- `*.html` files — the target repo generates its own HTML via MkDocs
- `build/` directories — compiled artifacts from cpp-advanced exercises
- `_sidebar.md`, `index.html` — Docsify config specific to this repo's local site

### Naming conventions

- **Zephyr files** use hyphens: `01-zephyr-intro.md` (not underscores)
- **All tracks** use `00-learning-plan.md` or `00-mastery-plan.md` as the entry point
- **Exercises** go in `<track>/exercises/`
- **Study notes** go in `<track>/study-notes/`
- **Projects** go in `<track>/projects/`
