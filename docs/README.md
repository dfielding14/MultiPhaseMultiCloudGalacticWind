# Documentation

This directory contains the source files for the project documentation, built with [MkDocs](https://www.mkdocs.org/) and the [Material theme](https://squidfunk.github.io/mkdocs-material/).

## Building Locally

1. Install dependencies:
```bash
pip install -r docs/requirements.txt
```

2. Serve documentation locally:
```bash
mkdocs serve
```

Then visit http://127.0.0.1:8000

3. Build static site:
```bash
mkdocs build
```

The site will be generated in the `site/` directory.

## Deployment

Documentation is automatically deployed to GitHub Pages when pushing to the main branch via GitHub Actions.

## Structure

```
docs/
├── index.md                 # Homepage
├── getting_started/         # Installation, quickstart, examples
├── physics/                 # Physics documentation
├── api/                     # API reference
├── guide/                   # User guides
├── development/             # Development docs
└── about/                   # References, license, changelog
```

## Adding Content

1. Create markdown files in appropriate directories
2. Update `mkdocs.yml` navigation section
3. Use standard markdown with these extensions:
   - LaTeX math: `$...$` for inline, `$$...$$` for blocks
   - Code blocks with syntax highlighting
   - Admonitions for notes/warnings
   - Tables and footnotes

## LaTeX Math

Math rendering is enabled via MathJax. Examples:

Inline: `$E = mc^2$` renders as $E = mc^2$

Block:
```markdown
$$
\frac{dv}{dr} = \frac{v/r}{1 - 1/\mathcal{M}^2}
$$
```