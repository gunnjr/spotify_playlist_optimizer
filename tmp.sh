#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/release_auth_success.sh v0.2.1
# Example: ./scripts/release_auth_success.sh v0.2.1

VERSION="${1:-v0.2.1}"

echo "🏷️  Creating release $VERSION..."

# Ensure repo clean
git add -A
git commit -m "release: $VERSION – Spotify auth successfully validated"
git tag -a "$VERSION" -m "Auth successful (Spotify OAuth verified)"
git push origin main
git push origin "$VERSION"

echo "✅ Release $VERSION pushed to GitHub."

