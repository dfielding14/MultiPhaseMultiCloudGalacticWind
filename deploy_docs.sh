#!/bin/bash
# Deploy documentation to GitHub Pages

echo "Building documentation..."
mkdocs build

echo "Documentation built successfully!"
echo ""
echo "To deploy to GitHub Pages:"
echo "1. Commit and push your changes to the main branch"
echo "2. GitHub Actions will automatically deploy the documentation"
echo ""
echo "Or manually deploy with:"
echo "  mkdocs gh-deploy"
echo ""
echo "To serve locally for testing:"
echo "  mkdocs serve"
echo "  Then visit http://127.0.0.1:8000"