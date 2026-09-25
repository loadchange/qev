# Homebrew packaging

The qev repository is its own tap: `Formula/qev.rb` at the repository root.

```bash
brew tap loadchange/qev https://github.com/loadchange/qev
brew install loadchange/qev/qev
```

The formula creates a Python 3.12 virtual environment and installs pinned,
checksummed wheels (`requirements.txt`, plus `mlx-vlm` without its optional
dependencies). No torch, OpenCV or audio packages are installed; images and
video use numpy processors. Homebrew fetches every wheel before the build, so
the install step itself runs offline.

A formula without a bottle is built from source, and Homebrew then requires
current developer tools (for example Xcode 27 on macOS 27) even though nothing
is compiled. Releases therefore ship a bottle built on macOS 14 arm64, which
Homebrew pours on macOS 14 and every newer version without that check.

## Updating dependencies

```bash
uv pip freeze --python .venv/bin/python | grep -v '^-e' > /tmp/tested.txt
MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile packaging/homebrew/requirements.in \
  --python-platform aarch64-apple-darwin --python-version 3.12 \
  --constraints /tmp/tested.txt --no-header --no-annotate -o packaging/homebrew/requirements.txt
```

The constraints keep the Homebrew environment on the versions the test suite
ran with.

## Release

1. Bump `version` in `pyproject.toml` and `qev/__init__.py`, commit, then tag and
   push: `git tag v0.4.0 && git push origin v0.4.0`.
2. The `Homebrew bottle` workflow builds, tests and bottles the formula and
   attaches the bottle and its JSON to the `v0.4.0` release.
3. Write the formula for the tagged source and the bottle:

   ```bash
   curl -sL https://github.com/loadchange/qev/archive/refs/tags/v0.4.0.tar.gz | shasum -a 256
   gh release download v0.4.0 --pattern '*.bottle.json' --dir /tmp/qev-bottle
   python3 packaging/homebrew/generate_formula.py \
     --source-url https://github.com/loadchange/qev/archive/refs/tags/v0.4.0.tar.gz \
     --source-sha256 <sha256> --bottle-json /tmp/qev-bottle/*.json --output Formula/qev.rb
   ```

4. Commit `Formula/qev.rb` to `main`. `brew update && brew upgrade qev` picks it up.

Manual runs of the workflow (`gh workflow run homebrew-bottle.yml`) build the
same bottle as a workflow artifact without publishing anything.
