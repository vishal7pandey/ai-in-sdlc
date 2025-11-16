<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I want you to document everything about uv, and how uv is used for libary management.

Now I have extensive information about uv. Let me create a comprehensive documentation guide covering all aspects of uv and library management.

# Complete uv Documentation: Python Package and Library Management

## Overview

**uv** is an extremely fast Python package and project manager written in Rust, designed to replace and unify tools like pip, pip-tools, pipx, poetry, pyenv, twine, and virtualenv. Created by Astral (the team behind Ruff), uv offers 10-100x faster performance than pip with a unified CLI interface.[^1][^2][^3][^4]

## Installation

### Standalone Installer (Recommended)

**macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with wget
wget -qO- https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Specific Version:**

```bash
curl -LsSf https://astral.sh/uv/0.9.9/install.sh | sh
```


### Alternative Installation Methods

**Via pip or pipx:**

```bash
pip install uv
# or
pipx install uv
```

**Via Homebrew:**

```bash
brew install uv
```

**Via WinGet (Windows):**

```bash
winget install --id=astral-sh.uv -e
```

**Via Cargo:**

```bash
cargo install --git https://github.com/astral-sh/uv uv
```


### Upgrading uv

**Self-update (standalone installer):**

```bash
uv self update
```

**Via package manager:**

```bash
pip install --upgrade uv
```


### Uninstalling

**Remove binaries:**

```bash
# Default location (0.5.0+)
rm ~/.local/bin/uv
rm ~/.local/bin/uvx
rm ~/.local/bin/uvw

# Old location (pre-0.5.0)
rm ~/.cargo/bin/uv
```


***

## Core Concepts

### What uv Replaces

| Traditional Tool | uv Equivalent | Purpose |
| :-- | :-- | :-- |
| pip | `uv pip` / `uv add` | Package installation |
| pip-tools | `uv pip compile` / `uv lock` | Dependency resolution |
| venv/virtualenv | `uv venv` / automatic | Virtual environments |
| pyenv | `uv python` | Python version management |
| pipx | `uv tool` / `uvx` | CLI tool installation |
| poetry | `uv` (project mode) | Project management |
| twine | `uv publish` | Package publishing |

### Key Performance Features

**Speed Benchmarks:**

- 10-100x faster than pip for package installation
- 115x faster with caching enabled
- 80x faster than `python -m venv` for environment creation
- 10x faster than poetry install[^3][^6][^1]

**How It's Fast:**

1. **Rust Implementation**: No Python overhead
2. **Parallel Downloads**: Concurrent package fetching
3. **Global Cache**: Deduplicated storage with hard links
4. **Binary Format Metadata**: Memory-mapped data (no parsing)
5. **Lock-free Data Structures**: Concurrent cache access[^7]

***

## Python Version Management

uv manages Python installations similar to pyenv.[^2][^8][^9]

### Installing Python Versions

```bash
# Install specific version
uv python install 3.12

# Install multiple versions
uv python install 3.10 3.11 3.12

# Install latest patch of a minor version
uv python install 3.11

# Install specific patch
uv python install 3.11.7
```


### Listing Python Versions

```bash
# List installed versions
uv python list

# List all available versions
uv python list --all-versions
```


### Pinning Python Version

```bash
# Pin version for project (creates .python-version)
uv python pin 3.12

# Pin specific patch
uv python pin 3.11.7
```

The `.python-version` file is automatically recognized and respected by uv.[^8][^9]

### Upgrading Python Versions

```bash
# Upgrade to latest patch of 3.12
uv python upgrade 3.12

# Upgrade all installed versions
uv python upgrade
```

**Auto-upgrade Virtual Environments:**

```bash
# Enable auto-upgrade for new installs
uv python install 3.12 --preview-features python-upgrade

# Virtual environments using this version will auto-upgrade
```


### Switching from pyenv to uv

**Steps:**

1. Remove pyenv from shell configuration:
```bash
# Edit ~/.bashrc or ~/.zshrc
# Comment out:
# eval "$(pyenv init -)"
# eval "$(pyenv virtualenv-init -)"

source ~/.bashrc
```

2. Transition commands:
| pyenv | uv |
|-------|-----|
| `pyenv install 3.12` | `uv python install 3.12` |
| `pyenv versions` | `uv python list` |
| `pyenv local 3.12` | `uv python pin 3.12` |
| `pyenv global 3.12` | Not supported (by design) |[^8][^10]
3. Recreate virtual environments:
```bash
# Delete old environment
rm -rf .venv

# Create new with uv
uv venv
```

**Key Differences:**

- uv automatically downloads Python on-demand (no explicit install needed)
- `.python-version` file still works
- No global Python shims (project-focused)[^11][^8]

***

## Project Management

### Creating Projects

**Initialize new project:**

```bash
# Application (default)
uv init my-app

# Library
uv init my-lib --lib

# Package-less (scripts only)
uv init --bare

# With options
uv init example \
  --bare \
  --description "Hello world" \
  --author-from git \
  --vcs git \
  --python-pin
```

**Project Templates:**

uv supports two built-in templates:

1. **Application**: Includes `__main__.py` for execution
2. **Library**: Includes module structure for distribution[^12]

**Community Templates:**

For advanced scaffolding, use external tools:

```bash
# Using Copier
uvx copier copy gh:pawamoy/copier-uv /path/to/project

# Using uvinit (wrapper around copier)
uvx uvinit

# Using pytemplate-uv
uvx pytemplate-uv --template pyproject --name my-project
```


### Project Structure

**Generated structure:**

```
my-project/
├── .python-version      # Pinned Python version
├── pyproject.toml       # Project configuration
├── README.md
├── uv.lock             # Lockfile (auto-generated)
├── .venv/              # Virtual environment (created on first use)
└── src/
    └── my_project/
        ├── __init__.py
        └── __main__.py  # (for applications)
```


***

## Dependency Management

### Adding Dependencies

**Basic usage:**

```bash
# Add runtime dependency
uv add requests

# Add with version constraint
uv add "flask>=2.0,<3.0"

# Add multiple packages
uv add numpy pandas

# Add from git
uv add git+https://github.com/user/repo

# Add local package
uv add --editable ../my-local-package
```


### Development Dependencies

```bash
# Add to dev group (PEP 735)
uv add --dev pytest
uv add --dev ruff mypy

# Result in pyproject.toml:
# [dependency-groups]
# dev = ["pytest>=8.1.1"]
```


### Optional Dependencies

```bash
# Add optional dependency
uv add --optional docs sphinx

# Result in pyproject.toml:
# [project.optional-dependencies]
# docs = ["sphinx>=8.1.3"]
```


### Custom Dependency Groups

```bash
# Create custom groups
uv add --group test pytest pytest-cov
uv add --group lint ruff

# Result:
# [dependency-groups]
# test = ["pytest", "pytest-cov"]
# lint = ["ruff"]
```

**Installing groups:**

```bash
# Sync specific group
uv sync --group test

# Sync multiple groups
uv sync --group test --group lint

# Sync all groups
uv sync --all-groups

# Exclude dev group
uv sync --no-dev

# Only install specific group
uv sync --only-group test
```


### Removing Dependencies

```bash
# Remove dependency
uv remove requests

# Remove from specific group
uv remove --dev pytest
uv remove --group test pytest-cov
```


### Difference: optional-dependencies vs dependency-groups

| Feature | `optional-dependencies` | `dependency-groups` |
| :-- | :-- | :-- |
| Standard | PEP 621 | PEP 735 |
| Published | Yes | No |
| Install syntax | `pip install pkg[extra]` | `uv sync --group name` |
| Use case | User-facing features | Development tools |
| Compatibility | All tools | uv-specific |

**Best Practices:**

- Use **optional-dependencies** for library features users can opt into
- Use **dependency-groups** for dev tools (testing, linting, docs)[^14]

***

## Lock Files and Reproducibility

### How Locking Works

uv automatically generates `uv.lock` when you add dependencies.[^16][^17]

**Lock file features:**

- Universal: Works across all platforms (Windows, macOS, Linux)
- Includes transitive dependencies
- Contains exact versions and hashes
- Should be committed to version control[^17][^16]


### Working with Lock Files

**Generate/update lock:**

```bash
# Auto-generated when adding deps
uv add requests

# Manually update lock
uv lock

# Lock specific Python version
uv lock --python 3.12
```

**Syncing environment:**

```bash
# Install from lockfile
uv sync

# Strict mode (fail if lock outdated)
uv sync --locked

# Check lock status
uv lock --check
```


### Upgrading Dependencies

```bash
# Upgrade all to latest compatible
uv lock --upgrade

# Upgrade specific package
uv lock --upgrade-package requests

# Upgrade with version constraint
uv lock --upgrade-package "pandas==2.1.0"

# Combine with sync
uv sync --upgrade-package requests
```

**Important**: Never manually edit `uv.lock`. Always use uv commands.[^16]

### Migrating from requirements.txt

```bash
# 1. Create pyproject.toml
uv init --bare

# 2. Import dependencies
uv add -r requirements.txt

# 3. Import dev dependencies
uv add --dev -r requirements-dev.txt

# 4. Verify
uv pip freeze

# 5. Remove old files
rm requirements.txt requirements-dev.txt
```


***

## Virtual Environments

### Creating Environments

```bash
# Create in .venv (default)
uv venv

# Create with specific Python version
uv venv --python 3.12
uv venv --python 3.11.7

# Create in custom location
uv venv /path/to/env

# Create with name
uv venv my-env
```

**Performance**: 80x faster than `python -m venv`[^1]

### Activation

uv automatically detects and uses environments, so activation is often unnecessary.[^18][^2]

**Manual activation (if needed):**

```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```


### Automatic Environment Detection

uv searches for `.venv` in current directory and parent directories.[^19]

```bash
# No activation needed
uv pip install requests
uv run python script.py
```


***

## The pip Interface

uv provides a drop-in replacement for pip with enhanced performance.[^20][^21][^19]

### Basic Commands

```bash
# Create virtual environment
uv venv

# Install packages
uv pip install requests
uv pip install -r requirements.txt
uv pip install -e .  # editable install

# Install with extras
uv pip install "mypackage[dev,test]"

# Uninstall
uv pip uninstall requests

# List installed
uv pip list

# Show package info
uv pip show requests

# Freeze
uv pip freeze > requirements.txt
```


### Compiling Requirements

```bash
# Generate requirements.txt from .in file
uv pip compile requirements.in

# Universal resolution (cross-platform)
uv pip compile --universal requirements.in

# With Python version
uv pip compile --python-version 3.12 requirements.in

# Output to specific file
uv pip compile requirements.in -o requirements.txt
```


### Syncing Environments

```bash
# Install exact versions from requirements.txt
uv pip sync requirements.txt

# Removes packages not in requirements.txt
```


### Compatibility with pip

**Fully compatible:**

- `requirements.txt` files
- PyPI and custom indexes
- Editable installs
- Constraints files[^3][^19]

**Differences from pip:**

- Virtual environments used by default (not system Python)
- PEP 517 build isolation enabled by default
- Stricter spec enforcement
- Some command-line options differ[^19]

**System Python (use with caution):**

```bash
uv pip install --system requests
```


***

## Running Scripts

uv makes running Python scripts seamless with automatic dependency management.[^22][^23][^24]

### Simple Scripts

```bash
# Run script (no dependencies)
uv run script.py

# Run in project context
uv run python -m mymodule

# Run with Python flags
uv run python -c "print('hello')"
```


### Scripts with Inline Dependencies

**PEP 723 inline script metadata:**

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests",
#     "rich",
# ]
# ///

import requests
from rich import print

response = requests.get("https://api.github.com")
print(response.json())
```

**Running:**

```bash
# Via uv run
uv run script.py

# As executable (after chmod +x)
./script.py
```


### Temporary Dependencies

```bash
# Add packages for this run only
uv run --with requests --with rich script.py

# From requirements file
uv run --with-requirements requirements.txt script.py

# Isolated (no project dependencies)
uv run --no-project script.py
```


### Signal Handling

uv forwards signals to child processes on Unix systems (except SIGKILL, SIGCHLD, SIGIO, SIGPOLL).[^23]

**Windows**: uv ignores Ctrl-C, deferring to child process for clean exit.[^23]

***

## Tool Management (pipx Replacement)

uv replaces pipx for installing and running CLI tools.[^25][^5][^26]

### Running Tools Ephemerally

```bash
# Using uvx (alias for uv tool run)
uvx pycowsay 'hello world!'

# Using uv tool run
uv tool run ruff check .

# With specific version
uvx black@24.0.0 script.py
```


### Installing Tools

```bash
# Install globally
uv tool install ruff

# Install specific version
uv tool install black==24.0.0

# Install from git
uv tool install git+https://github.com/user/repo

# Editable install
uv tool install --editable ~/my-tool
```

**Installed location**: `~/.local/bin`[^25]

### Managing Installed Tools

```bash
# List installed tools
uv tool list

# Upgrade tool
uv tool upgrade ruff

# Upgrade all tools
uv tool upgrade --all

# Uninstall tool
uv tool uninstall ruff
```


### Differences from pipx

| Feature | pipx | uv tool |
| :-- | :-- | :-- |
| Install location | `~/.local/pipx/venvs/` | Tool cache |
| Editable installs | `pipx install -e .` | `uv tool install --editable .` |
| Include deps | `--include-deps` | Not yet supported |


***

## Building and Publishing Packages

### Building Distributions

**Build both source and wheel:**

```bash
uv build

# Output:
# dist/
# ├── example-0.1.0.tar.gz (source)
# └── example-0.1.0-py3-none-any.whl (wheel)
```

**Build specific format:**

```bash
# Source distribution only
uv build --sdist

# Wheel only
uv build --wheel

# Both from source
uv build --sdist --wheel
```

**Build different project:**

```bash
uv build path/to/project
```


### Build Backend

uv uses the build backend specified in `pyproject.toml` (default: hatchling).[^28]

**Using uv's build backend (stable):**

```toml
[build-system]
requires = ["uv-build-backend"]
build-backend = "uv_build_backend"
```


### Publishing to PyPI

```bash
# Publish to PyPI
uv publish

# Publish to Test PyPI
uv publish --publish-url https://test.pypi.org/legacy/

# With token
uv publish --token pypi-...

# Multiple indexes (configuration needed)
```

**Authentication:**

Set environment variables:

```bash
export UV_PUBLISH_TOKEN=pypi-...
export UV_PUBLISH_USERNAME=__token__
```

**Multi-index workflow:**

```toml
# pyproject.toml
[tool.uv.publish]
index-url = "https://test.pypi.org/legacy/"
token = "${TEST_PYPI_TOKEN}"

[[tool.uv.publish.indexes]]
name = "pypi"
url = "https://upload.pypi.org/legacy/"
token = "${PYPI_TOKEN}"
```


***

## Workspaces (Monorepos)

uv workspaces enable managing multiple Python packages in a single repository.[^29][^30][^31]

### Setting Up a Workspace

**Root `pyproject.toml`:**

```toml
[project]
name = "my-monorepo"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["package-a", "package-b"]

[tool.uv.sources]
package-a = { workspace = true }
package-b = { workspace = true }

[tool.uv.workspace]
members = [
    "packages/package-a",
    "packages/package-b",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "ruff>=0.6.0",
]
package = false  # Root is not a package
```

**Directory structure:**

```
my-monorepo/
├── .venv/                  # Shared virtual environment
├── pyproject.toml          # Root configuration
├── uv.lock                 # Single lockfile for all packages
└── packages/
    ├── package-a/
    │   ├── pyproject.toml
    │   └── src/package_a/
    └── package-b/
        ├── pyproject.toml
        └── src/package_b/
```


### Package Configuration

**Individual package `pyproject.toml`:**

```toml
[project]
name = "package-a"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "requests>=2.31.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build_meta"
```


### Workspace Commands

```bash
# Sync all workspace members
uv sync

# Run command in workspace context
uv run python -c "import package_a; import package_b"

# Add dependency to specific member
cd packages/package-a
uv add numpy

# Build specific member
uv build packages/package-a
```


### Workspace Features

**Shared virtual environment**: All packages use single `.venv`[^29]

**Single lockfile**: `uv.lock` resolves all workspace dependencies together[^31][^29]

**Cross-references**: Packages can depend on each other[^31][^29]

**Unified Python version**: `requires-python` is intersection of all members[^32]

### Building Workspace Members

**Challenge**: uv doesn't natively build workspace members with local dependencies.[^33]

**Solution - Una**:

```bash
# Install una
uv add --dev una

# Build with local dependencies included
una build packages/package-a
```


***

## Caching

uv uses aggressive caching for performance.[^34][^35][^7]

### Cache Location

**Default locations:**

- Linux: `~/.cache/uv/`
- macOS: `~/Library/Caches/uv/`
- Windows: `%LOCALAPPDATA%\uv\cache\`[^35][^34]


### Cache Structure

```
~/.cache/uv/
├── wheels/         # Extracted wheel contents
├── built-wheels/   # Wheels built from source
├── git-db/         # Git repositories
├── archive-v0/     # Downloaded archives
└── simple-v4/      # Package metadata
```


### Changing Cache Directory

**Temporary:**

```bash
uv pip install --cache-dir /custom/cache requests
```

**Permanent:**

```bash
# Environment variable
export UV_CACHE_DIR=/custom/cache

# Or in pyproject.toml/uv.toml
cache-dir = "/custom/cache"
```

**Multiple drives (Windows):**

```toml
# D:/uv.toml
cache-dir = "D:/.uv/cache"
```


### Cache Operations

```bash
# View cache directory
uv cache dir

# Clean entire cache
uv cache clean

# Prune (remove outdated)
uv cache prune

# CI-optimized prune (keep source-built wheels)
uv cache prune --ci
```


### Cache Efficiency

**Hard Links**: uv uses hard links from cache to environments (zero-copy)[^36][^7]

**Copy-on-Write**: On supporting filesystems, efficient copy operations[^6][^7]

**Global Deduplication**: Same package version shared across all environments[^7]

**Warning**: Cache directory must be on same filesystem as project for optimal performance (hard links requirement)[^36][^34]

### Cache Safety

- Thread-safe and append-only
- Safe for concurrent uv commands
- File-based locks on environments
- **Not safe** to modify cache directly or run `uv cache clean` during other operations[^34]

***

## Advanced Features

### uv vs Traditional Tools Comparison

| Feature | pip + venv | poetry | uv |
| :-- | :-- | :-- | :-- |
| Speed | Baseline | 10x faster | 10-100x faster |
| Python management | External (pyenv) | External | Built-in |
| Lock files | No | Yes | Yes |
| Universal resolution | No | No | Yes |
| Build backend | setuptools | poetry-core | Multiple |
| Publishing | twine | Built-in | Built-in |

### uv add vs uv pip install

| Aspect | `uv add` | `uv pip install` |
| :-- | :-- | :-- |
| Updates `pyproject.toml` | ✅ Yes | ❌ No |
| Updates `uv.lock` | ✅ Yes | ❌ No |
| Universal resolution | ✅ Yes | ❌ No (platform-specific) |
| Requires project | ✅ Yes | ❌ No |
| Best for | Project dependencies | Ad-hoc installs |

**Universal Resolution Example:**

Package has versions: 1.0.0, 1.1.0 (Windows/Linux), 1.2.0 (Linux only)

- `uv add`: Selects 1.1.0 (works everywhere)
- `uv pip install`: Selects 1.2.0 on Linux (breaks portability)[^38]


### Configuration Files

**Priority order:**

1. Command-line flags
2. `pyproject.toml`
3. `uv.toml` (project-specific)
4. `~/.config/uv/uv.toml` (global)
5. Environment variables[^39]

**Example `uv.toml`:**

```toml
cache-dir = "/custom/cache"
python-preference = "only-managed"

[pip]
index-url = "https://pypi.org/simple"
extra-index-url = ["https://custom.pypi.org/simple"]
```


### Environment Variables

```bash
# Cache
export UV_CACHE_DIR=/custom/cache
export UV_NO_CACHE=1

# Python
export UV_PYTHON_PREFERENCE=only-managed
export UV_PYTHON=python3.12

# Installation
export UV_INDEX_URL=https://pypi.org/simple
export UV_LINK_MODE=copy  # Instead of hardlink

# Project
export UV_PROJECT_ENVIRONMENT=/usr/local/
```


### Docker Integration

**Minimal Dockerfile:**

```dockerfile
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set environment
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# Copy source
COPY . .

# Run
CMD ["uv", "run", "python", "-m", "myapp"]
```


### CI/CD Optimization

**GitHub Actions example:**

```yaml
- name: Set up uv
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Restore cache
  uses: actions/cache@v4
  with:
    path: ~/.cache/uv
    key: ${{ runner.os }}-uv-${{ hashFiles('uv.lock') }}

- name: Install dependencies
  run: uv sync --frozen

- name: Run tests
  run: uv run pytest

- name: Prune cache
  run: uv cache prune --ci
```


***

## Best Practices

### Project Workflow

**1. Start new project:**

```bash
uv init my-project
cd my-project
uv add requests pandas
```

**2. Development:**

```bash
uv add --dev pytest ruff mypy
uv sync
uv run pytest
```

**3. Lock dependencies:**

```bash
uv lock
git add uv.lock pyproject.toml
git commit -m "Lock dependencies"
```

**4. Collaborate:**

```bash
git clone repo
cd repo
uv sync  # Installs exact versions
```


### When to Use Each Interface

**Use `uv` (project mode):**

- New projects
- Long-term applications
- Team collaboration
- Reproducible builds

**Use `uv pip`:**

- Migrating from pip
- Quick testing
- System-wide tools
- CI/CD compatibility

**Use `uvx`:**

- One-off tools
- Script execution
- No installation needed


### Dependency Management

**Runtime dependencies:**

```bash
uv add requests  # Always use uv add
```

**Development tools:**

```bash
uv add --dev pytest ruff mypy
```

**Optional features:**

```bash
uv add --optional docs sphinx
# Users install with: pip install mypackage[docs]
```

**Internal dev groups:**

```bash
uv add --group ci coverage
# Not published; only for developers
```


### Lock File Management

**Do:**

- Commit `uv.lock` to version control
- Use `uv sync --locked` in CI/CD
- Run `uv lock --upgrade` periodically

**Don't:**

- Edit `uv.lock` manually
- Delete `uv.lock` (regenerate with `uv lock`)
- Ignore lock file changes in PRs


### Performance Tips

**1. Use cache on same filesystem:**

```bash
# Good: cache and project on same drive
UV_CACHE_DIR=/home/user/.cache/uv

# Bad: cache on different drive (slow copies)
UV_CACHE_DIR=/mnt/otherdrive/.cache/uv
```

**2. Enable hard links:**

```bash
# Avoid UV_LINK_MODE=copy unless necessary
# Hard links are 100x faster
```

**3. Reuse environments:**

```bash
# Don't recreate .venv unnecessarily
uv sync  # Updates existing environment
```

**4. CI optimization:**

```bash
# Cache uv directory
# Use uv cache prune --ci at end
```


### Python Version Strategy

**For libraries:**

```toml
[project]
requires-python = ">=3.9"  # Support wide range
```

**For applications:**

```toml
[project]
requires-python = ">=3.12"  # Use latest
```

```bash
uv python pin 3.12  # Lock exact version
```


***

## Troubleshooting

### Common Issues

**1. Hard link warnings on Windows:**

```
Failed to hardlink files; falling back to full copy.
```

**Solution**: Create cache on same drive

```bash
# Set in D:/uv.toml
cache-dir = "D:/.uv/cache"
```

**2. Package build failures:**

```bash
# Use --no-build-isolation for problematic packages
uv pip install --no-build-isolation biopython==1.77

# Install build dependencies first
uv pip install wheel setuptools
```

**3. Virtual environment not found:**

```bash
# uv searches for .venv automatically
# Create if missing:
uv venv

# Or specify location:
uv venv /custom/path
```

**4. Dependency resolution conflicts:**

```bash
# Check which packages conflict
uv lock --verbose

# Try upgrading
uv lock --upgrade

# Use constraints
uv add "package>=1.0,<2.0"
```

**5. Import errors after uv add:**

```bash
# Ensure environment is synced
uv sync

# Or run directly
uv run python script.py
```


### Compatibility Issues

**pipx --include-deps not supported:**

Use `uv tool install` without `--include-deps`. Feature not yet implemented.[^27]

**Conda integration:**

uv and conda don't integrate well. Choose one:

- Use `uv` for everything (recommended)
- Use `uv pip` within conda environment (manual)[^29]

**System Python modification:**

uv prefers virtual environments. To modify system Python (not recommended):

```bash
uv pip install --system package
```


***

## Cheat Sheet

### Quick Reference

| Task | Command |
| :-- | :-- |
| **Install uv** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **New project** | `uv init my-project` |
| **Add dependency** | `uv add requests` |
| **Add dev dependency** | `uv add --dev pytest` |
| **Install Python** | `uv python install 3.12` |
| **Create venv** | `uv venv` |
| **Sync deps** | `uv sync` |
| **Run script** | `uv run script.py` |
| **Run tool** | `uvx black .` |
| **Build package** | `uv build` |
| **Publish** | `uv publish` |
| **Lock deps** | `uv lock` |
| **Upgrade deps** | `uv lock --upgrade` |
| **Clean cache** | `uv cache clean` |

### Migration Paths

**From pip + requirements.txt:**

```bash
uv init --bare
uv add -r requirements.txt
uv add --dev -r requirements-dev.txt
```

**From poetry:**

```bash
# Convert manually or use poetry export
poetry export -f requirements.txt | uv add -r /dev/stdin
uv add --dev pytest  # Add dev deps individually
```

**From pipx:**

```bash
# List pipx tools
pipx list

# Install with uv
uv tool install tool-name
```


***

This comprehensive guide covers everything about uv for Python package and library management, from installation and basic usage to advanced features like workspaces, caching, and publishing. Use it as a complete reference for transitioning to or mastering uv.[^4][^2][^1][^3]
<span style="display:none">[^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65][^66][^67][^68][^69][^70][^71][^72][^73][^74][^75][^76][^77][^78][^79][^80][^81][^82][^83][^84][^85][^86][^87]</span>

<div align="center">⁂</div>

[^1]: https://www.digitalocean.com/community/conceptual-articles/uv-python-package-manager

[^2]: https://realpython.com/python-uv/

[^3]: https://www.datacamp.com/tutorial/python-uv

[^4]: https://astral.sh/blog/uv

[^5]: https://pypi.org/project/uv/

[^6]: https://envelope.dev/blog/poetry-vs-uv-vs-pip-choosing-the-right-package-installer

[^7]: https://xebia.com/blog/uv-the-engineering-secrets-behind-pythons-speed-king/

[^8]: https://pydevtools.com/handbook/how-to/how-to-switch-from-pyenv-to-uv-for-managing-python-versions/

[^9]: https://realpython.com/intro-to-pyenv/

[^10]: https://github.com/astral-sh/uv/issues/12263

[^11]: https://rob.cogit8.org/posts/2024-09-19-pyenv-to-uv/

[^12]: https://docs.astral.sh/uv/concepts/projects/init/

[^13]: https://github.com/astral-sh/uv/issues/9011

[^14]: https://pydevtools.com/handbook/explanation/what-are-optional-dependencies-and-dependency-groups/

[^15]: https://discuss.python.org/t/pep-735-dependency-groups-in-pyproject-toml/39233?page=14

[^16]: https://jakubk.cz/posts/uv_lock/

[^17]: https://pydevtools.com/handbook/how-to/how-to-use-a-uv-lockfile-for-reproducible-python-environments/

[^18]: https://blog.appsignal.com/2025/09/24/switching-from-pip-to-uv-in-python-a-comprehensive-guide.html

[^19]: https://docs.astral.sh/uv/pip/compatibility/

[^20]: https://pydevtools.com/handbook/reference/uv/

[^21]: https://docs.astral.sh/uv/pip/

[^22]: https://mathspp.com/blog/til/standalone-executable-python-scripts-with-uv

[^23]: https://docs.astral.sh/uv/concepts/projects/run/

[^24]: https://docs.astral.sh/uv/guides/scripts/

[^25]: https://www.reddit.com/r/learnpython/comments/1mwv8il/is_there_a_uv_equivalent_of_pipx_install_e/

[^26]: https://github.com/astral-sh/uv/issues/8244

[^27]: https://github.com/astral-sh/uv/issues/6922

[^28]: https://pydevtools.com/blog/uv-build-backend/

[^29]: https://matekole.com/tils/monorepo-uv-workspaces/

[^30]: https://gafni.dev/blog/cracking-the-python-monorepo/

[^31]: https://github.com/JasperHG90/uv-monorepo

[^32]: https://docs.astral.sh/uv/concepts/projects/workspaces/

[^33]: https://pypi.org/project/una/

[^34]: https://docs.astral.sh/uv/concepts/cache/

[^35]: https://stackoverflow.com/questions/79664325/how-to-change-the-uv-cache-directory

[^36]: https://github.com/astral-sh/uv/issues/6613

[^37]: https://jinaldesai.com/python-pip-vs-pdm-vs-poetry-vs-uv/

[^38]: https://github.com/astral-sh/uv/issues/9219

[^39]: https://docs.astral.sh/uv/getting-started/installation/

[^40]: https://pydevtools.com/handbook/how-to/how-to-install-uv/

[^41]: https://mac.install.guide/python/install-uv

[^42]: https://sudhanva.me/conda-vs-poetry-vs-uv-vs-pip/

[^43]: https://deepnote.com/blog/ultimate-guide-to-uv-library-in-python

[^44]: https://www.youtube.com/watch?v=AMdG7IjgSPM

[^45]: https://igorstechnoclub.com/uv-python-package-manager-beginners-guide/

[^46]: https://www.linkedin.com/posts/damienbenveniste_pip-vs-uv-vs-poetry-nowadays-i-feel-it-activity-7351658451695042561-CQmd

[^47]: https://www.loopwerk.io/articles/2024/python-poetry-vs-uv/

[^48]: https://github.com/astral-sh/uv

[^49]: https://www.youtube.com/watch?v=BDkr0HH_QAQ

[^50]: https://docs.astral.sh/uv/guides/install-python/

[^51]: https://www.reddit.com/r/Python/comments/1gqh4te/uv_after_050_might_be_worth_replacing/

[^52]: https://www.reddit.com/r/Python/comments/1isv37n/is_uv_package_manager_taking_over/

[^53]: https://github.com/astral-sh/uv/issues/7019

[^54]: https://pydevtools.com/handbook/how-to/migrate-requirements.txt/

[^55]: https://docs.astral.sh/uv/guides/projects/

[^56]: https://docs.astral.sh/uv/concepts/resolution/

[^57]: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/

[^58]: https://docs.astral.sh/uv/concepts/projects/sync/

[^59]: https://www.reddit.com/r/Python/comments/1iy4h5k/cracking_the_python_monorepo_build_pipelines_with/

[^60]: https://docs.astral.sh/uv/pip/packages/

[^61]: https://www.reddit.com/r/learnpython/comments/1kkw9ou/installing_dependencies_from_a_project_using_uv/

[^62]: https://stackoverflow.com/questions/62408719/download-dependencies-declared-in-pyproject-toml-using-pip

[^63]: https://stackoverflow.com/questions/78902565/how-do-i-install-python-dev-dependencies-using-uv

[^64]: https://docs.astral.sh/uv/concepts/python-versions/

[^65]: https://www.reddit.com/r/Python/comments/1j9g0ii/uv_or_pyenv_for_student_python_teaching_python/

[^66]: https://pypi.org/project/uv/0.2.26/

[^67]: https://python.plainenglish.io/stop-using-pyenv-give-uv-a-chance-4e1ba6645d53

[^68]: https://docs.astral.sh/uv/concepts/projects/build/

[^69]: https://blog.stephenturner.us/p/uv-part-2-building-and-publishing-packages

[^70]: https://calmcode.io/course/uv/run

[^71]: https://github.com/astral-sh/uv/issues/8729

[^72]: https://github.com/astral-sh/uv/issues/7963

[^73]: https://docs.astral.sh/uv/guides/package/

[^74]: https://discuss.python.org/t/uploading-to-pypi-without-twine/14075

[^75]: https://adver.tools/python/tutorial/running-python-scripts-uv/

[^76]: https://thisdavej.com/packaging-python-command-line-apps-the-modern-way-with-uv/

[^77]: https://pydevtools.com/handbook/tutorial/publishing-your-first-python-package-to-pypi/

[^78]: https://hexmos.com/freedevtools/tldr/uv/uv-run/

[^79]: https://stackoverflow.com/questions/26059111/build-a-wheel-egg-and-all-dependencies-for-a-python-project

[^80]: https://pawamoy.github.io/copier-uv/generate/

[^81]: https://www.pyopensci.org/python-package-guide/package-structure-code/declare-dependencies.html

[^82]: https://docs.astral.sh/uv/concepts/projects/dependencies/

[^83]: https://github.com/jlevy/simple-modern-uv

[^84]: https://realpython.com/uv-vs-pip/

[^85]: https://pypi.org/project/pytemplate-uv/2.0.0/

[^86]: https://github.com/astral-sh/uv/issues/9754

[^87]: https://gotofritz.net/blog/2025-08-10-copier-python-template-with-uv/
