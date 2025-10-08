# Contributing

Thanks for helping improve `spotify_playlist_optimizer`!

## Dev setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install black ruff pre-commit
pre-commit install
```

## Workflow

1. Fork and create a feature branch
2. Keep commits small and focused
3. Ensure `black` and `ruff` pass
4. Add tests or artifacts where it helps (plots, CSVs)
5. Open a PR with a clear before/after description

Good first issues: visualization polish, alternative cycle constructors, report templates.
