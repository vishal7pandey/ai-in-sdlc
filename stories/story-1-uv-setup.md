# Subtask: UV Package Manager Setup and Virtual Environment Configuration

## Subtask Overview

**Subtask ID:** STORY-001-UV-SETUP
**Subtask Title:** Configure UV Package Manager, Create Virtual Environment, and Install All Dependencies
**Priority:** P0 - Critical (Prerequisite for all development)
**Estimated Effort:** 2-3 hours
**Prerequisites:** Git repository cloned, uv installed on system
**Dependencies:** Must complete before AC1-AC8 of Story 1

---

## Objectives

This subtask configures **uv** as the unified package and environment manager for the entire project, ensuring:

- ✅ `uv` replaces pip, venv, and pipenv for all package management
- ✅ Virtual environment is created and configured
- ✅ All project dependencies (runtime + dev) are installed
- ✅ Lock file is generated for reproducible builds
- ✅ Python version is pinned to 3.11
- ✅ Development tooling (pytest, ruff, mypy) is properly configured
- ✅ Team members can bootstrap the project in < 5 minutes

---

## Task Breakdown

### Task 1: Install uv on the System

**Status:** Verify first; only proceed if not installed

```bash
# Check if uv is already installed
which uv
uv --version

# If NOT installed, proceed with installation
```

**For macOS/Linux (Recommended - Standalone Installer):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**For Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (via pip/pipx):**
```bash
pip install uv
# or
pipx install uv
```

**Verification:**
```bash
uv --version
# Expected output: uv {VERSION}

which uv
# Expected: Path to uv executable
```

---

### Task 2: Create Project pyproject.toml Configuration

**File: `pyproject.toml`** (Project Root - Create/Update)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "req-eng-platform"
version = "1.0.0"
description = "Multi-agent conversational requirements engineering platform"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [
    { name = "Your Organization", email = "dev@example.com" }
]
keywords = ["ai", "agents", "requirements", "langgraph", "langchain"]

[project.urls]
repository = "https://github.com/your-org/req-eng-platform"
documentation = "https://github.com/your-org/req-eng-platform/docs"

# Core Runtime Dependencies
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.12.0",
    "psycopg[binary]>=3.17.0",
    "redis>=5.0.0",
    "chromadb>=0.4.15",
    "langchain>=0.1.0",
    "langchain-openai>=0.0.1",
    "langgraph>=0.0.28",
    "langsmith>=0.1.0",
    "openai>=1.3.0",
    "python-dotenv>=1.0.0",
    "pydantic-ai>=0.0.1",
    "python-multipart>=0.0.6",
    "jinja2>=3.1.0",
    "pyyaml>=6.0.0",
    "tenacity>=8.2.0",
    "structlog>=23.2.0",
    "python-json-logger>=2.0.7",
]

# Development Dependencies
[dependency-groups]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.5.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0",
    "black>=23.12.0",
    "isort>=5.13.0",
    "pre-commit>=3.5.0",
    "ipython>=8.18.0",
    "ipdb>=0.13.0",
]

test = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.5.0",
    "httpx>=0.25.0",
    "faker>=20.1.0",
]

lint = [
    "ruff>=0.1.0",
    "mypy>=1.7.0",
    "black>=23.12.0",
    "isort>=5.13.0",
]

docs = [
    "sphinx>=7.2.0",
    "sphinx-rtd-theme>=2.0.0",
    "myst-parser>=2.0.0",
]

[tool.uv]
# Ensure we use managed Python only
python-preference = "managed"
compile-bytecode = true

[tool.uv.sources]
# If using local packages, define them here

[tool.uv.workspace]
# For future monorepo support
members = []

# Black formatter configuration
[tool.black]
line-length = 100
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

# Ruff linter configuration
[tool.ruff]
line-length = 100
target-version = "py311"
exclude = [
    ".bzr",
    ".direnv",
    ".eggs",
    ".git",
    ".git-rewrite",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pants.d",
    ".pytype",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pypackages__",
    "_build",
    "buck-out",
    "build",
    "dist",
    "node_modules",
    "venv",
]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
    "ARG",  # flake8-unused-arguments
    "SIM",  # flake8-simplify
    "TCH",  # flake8-type-checking
    "RUF",  # Ruff-specific rules
]
ignore = [
    "E501",   # line too long (handled by black)
    "B008",   # do not perform function calls in argument defaults
    "C901",   # too complex
    "ARG001", # unused lambda argument
]

[tool.ruff.lint.isort]
known-third-party = ["fastapi", "pydantic", "sqlalchemy"]

# MyPy configuration
[tool.mypy]
python_version = "3.11"
disallow_untyped_defs = false
disallow_incomplete_defs = false
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_configs = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = [
    "langchain.*",
    "langsmith.*",
    "langgraph.*",
    "chromadb.*",
]
ignore_missing_imports = true

# Pytest configuration
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_mode = "auto"
addopts = "-v --tb=short --strict-markers"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow running tests",
    "api: API endpoint tests",
]

# Coverage configuration
[tool.coverage.run]
source = ["src"]
omit = [
    "*/__init__.py",
    "*/tests/*",
    "*/__pycache__/*",
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
```

---

### Task 3: Create .python-version File

**File: `.python-version`** (Project Root - Create)

```
3.11
```

**Why:** uv automatically detects and respects this file, pinning the project to Python 3.11.

**Verification:**
```bash
cat .python-version
# Expected output: 3.11
```

---

### Task 4: Pin Python Version Using uv

```bash
# From project root
uv python pin 3.11

# Verification
cat .python-version
# Expected output: 3.11

uv python list
# Expected: Shows Python 3.11 available/installed
```

---

### Task 5: Create Virtual Environment

```bash
# From project root, create virtual environment
uv venv

# Verification - environment should be created
ls -la .venv/
# Expected: bin/, lib/, pyvenv.cfg present

# Check activation script exists
ls -la .venv/bin/activate  # Linux/macOS
# or
ls .venv\Scripts\activate  # Windows
```

**Note:** The virtual environment is created in `.venv/` directory (uv default).

---

### Task 6: Add Dependencies Using uv

**Add all runtime dependencies:**
```bash
uv add fastapi uvicorn pydantic pydantic-settings sqlalchemy alembic psycopg redis chromadb langchain langchain-openai langgraph langsmith openai python-dotenv pydantic-ai python-multipart jinja2 pyyaml tenacity structlog python-json-logger
```

**Add development dependencies:**
```bash
uv add --dev pytest pytest-asyncio pytest-cov pytest-xdist ruff mypy black isort pre-commit ipython ipdb
```

**Add test dependencies (as separate group):**
```bash
uv add --group test httpx faker
```

**Add lint dependencies (as separate group):**
```bash
uv add --group lint
```

**Add docs dependencies (optional):**
```bash
uv add --group docs sphinx sphinx-rtd-theme myst-parser
```

---

### Task 7: Generate and Review Lock File

```bash
# The lock file is auto-generated when dependencies are added
# Verify it exists
ls -la uv.lock
# Expected: File exists with size > 50KB

# Review lock file (first few lines)
head -20 uv.lock
# Expected: TOML format with version and dependency info
```

**Important:** The `uv.lock` file is **automatically generated** and should be **committed to git**.

---

### Task 8: Sync Virtual Environment

```bash
# Sync the virtual environment with locked dependencies
uv sync

# Verification - check installation
uv pip list
# Expected: All packages from pyproject.toml listed

# Verify critical packages
uv pip show fastapi
uv pip show langgraph
uv pip show sqlalchemy

# All three should return package information
```

---

### Task 9: Verify All Dependency Groups

```bash
# Sync with all groups (including dev, test, docs, lint)
uv sync --all-groups

# Verification
uv pip list | grep -E "pytest|ruff|mypy|black"
# Expected: All dev tools listed

# Count total packages
uv pip list | wc -l
# Expected: 50+ packages installed
```

---

### Task 10: Configure .gitignore for uv

**File: `.gitignore`** (Add/Update)

```gitignore
# Virtual environment
.venv/
venv/
env/
ENV/

# uv cache
.uv/

# Python compiled files
__pycache__/
*.pyc
*.pyo
*.pyd
*.so
*.egg
*.egg-info/
dist/
build/

# Environment variables
.env
.env.local
.env.*.local

# IDE/Editor
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Type checking
.mypy_cache/
.dmypy.json
dmypy.json

# Pre-commit
.pre-commit-framework-dir/

# Database
*.db
*.sqlite3
postgres-data/
redis-data/
chromadb-data/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
```

---

### Task 11: Configure Pre-commit Hooks (Optional but Recommended)

**File: `.pre-commit-config.yaml`** (Project Root - Create)

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.8
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: detect-private-key

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
        additional_dependencies: [types-all]
```

**Install pre-commit hooks:**
```bash
uv run pre-commit install

# Verification
cat .git/hooks/pre-commit
# Expected: hook script present
```

---

### Task 12: Create Quick Development Setup Script

**File: `scripts/setup-dev.sh`** (Create - Linux/macOS)

```bash
#!/bin/bash
set -e

echo "🚀 Setting up development environment..."

# Check uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "✅ uv is installed: $(uv --version)"

# Pin Python version
echo "📌 Pinning Python to 3.11..."
uv python pin 3.11

# Create virtual environment
echo "🔧 Creating virtual environment..."
uv venv

# Install all dependencies
echo "📦 Installing all dependencies..."
uv sync --all-groups

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
uv run pre-commit install

# Verify installation
echo ""
echo "✅ Setup complete!"
echo ""
echo "🔍 Verification:"
uv pip list | head -10
echo ""
echo "📝 To activate environment manually:"
echo "  source .venv/bin/activate  (Linux/macOS)"
echo "  .venv\Scripts\activate     (Windows)"
echo ""
echo "🏃 To run commands:"
echo "  uv run python script.py"
echo "  uv run pytest"
echo "  uv run ruff check ."
```

**File: `scripts/setup-dev.bat`** (Create - Windows)

```batch
@echo off
setlocal enabledelayedexpansion

echo 🚀 Setting up development environment...

REM Check uv is installed
where uv >nul 2>nul
if errorlevel 1 (
    echo ❌ uv is not installed. Installing...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
)

for /f "tokens=*" %%i in ('uv --version') do echo ✅ uv is installed: %%i

echo 📌 Pinning Python to 3.11...
call uv python pin 3.11

echo 🔧 Creating virtual environment...
call uv venv

echo 📦 Installing all dependencies...
call uv sync --all-groups

echo 🪝 Installing pre-commit hooks...
call uv run pre-commit install

echo.
echo ✅ Setup complete!
echo.
echo 🔍 Verification:
call uv pip list | findstr /R "^[a-zA-Z]" | for /l %%i in (1,1,10) do echo %%i

echo.
echo 📝 To activate environment manually:
echo   .venv\Scripts\activate
echo.
echo 🏃 To run commands:
echo   uv run python script.py
echo   uv run pytest
echo   uv run ruff check .

endlocal
```

**Make executable and document:**
```bash
chmod +x scripts/setup-dev.sh
chmod +x scripts/setup-dev.bat
```

---

### Task 13: Verify Complete Setup

**Run comprehensive verification:**

```bash
# 1. Check Python version
python --version
# Expected: Python 3.11.x

# 2. Check uv version
uv --version
# Expected: uv {VERSION}

# 3. Verify .venv exists and is active
which python
# Expected: path includes .venv

# 4. List all packages (first 20)
uv pip list | head -20

# 5. Check critical packages version
uv pip show fastapi | grep Version
uv pip show langgraph | grep Version
uv pip show sqlalchemy | grep Version

# 6. Verify lock file
ls -lh uv.lock

# 7. Test importing critical packages
python -c "import fastapi; import pydantic; import sqlalchemy; print('✅ All imports successful')"

# 8. Verify pytest works
uv run pytest --version

# 9. Verify ruff works
uv run ruff --version

# 10. Verify mypy works
uv run mypy --version
```

**Expected Output Summary:**
```
✅ Python 3.11.x is active
✅ uv {VERSION} is installed
✅ .venv/bin/python is being used
✅ 50+ packages installed
✅ fastapi, langgraph, sqlalchemy versions confirmed
✅ uv.lock file exists (size > 50KB)
✅ All imports successful
✅ pytest, ruff, mypy functional
```

---

### Task 14: Update README with Setup Instructions

**File: `README.md`** (Update - Add Quick Start Section)

```markdown
## 🚀 Quick Start

### Prerequisites
- Git
- `uv` package manager (install via: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Setup (5 minutes)

**Option 1: Automatic Setup (Recommended)**
```bash
bash scripts/setup-dev.sh  # macOS/Linux
scripts/setup-dev.bat      # Windows
```

**Option 2: Manual Setup**
```bash
# Clone repository
git clone <repo-url>
cd req-eng-platform

# Pin Python version
uv python pin 3.11

# Create virtual environment
uv venv

# Install all dependencies
uv sync --all-groups

# Verify installation
python --version
uv pip list
```

### Running the Application

```bash
# Start Docker services
docker-compose up -d

# Wait for services to be healthy
sleep 30

# Run database migrations
python scripts/migrate.py

# Start API server
uvicorn src.main:app --reload

# Access API documentation
open http://localhost:8000/docs
```

### Common Commands

```bash
# Run tests
uv run pytest

# Check code style
uv run ruff check .

# Format code
uv run ruff format .

# Type check
uv run mypy src/

# Add new dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Update lock file
uv lock

# Sync environment
uv sync
```

## 📦 Dependency Management

This project uses **uv** for fast, reliable package management:

- **Add runtime dependency**: `uv add package-name`
- **Add dev dependency**: `uv add --dev package-name`
- **Install all dependencies**: `uv sync --all-groups`
- **Update lock file**: `uv lock --upgrade`
- **See all dependencies**: `uv pip list`

All dependencies are specified in `pyproject.toml` and locked in `uv.lock`.
```

---

## Acceptance Criteria

### ✅ AC1: uv Installed and Verified
**Given** system has curl or package manager
**When** I run `uv --version`
**Then** uv responds with version number >= 0.1.0

```bash
uv --version
# Expected: uv 0.x.x or higher
```

---

### ✅ AC2: pyproject.toml Properly Configured
**Given** project root directory
**When** I read `pyproject.toml`
**Then** it contains:
- `[project]` section with name and dependencies
- `[dependency-groups]` with dev, test, lint, docs groups
- `[tool.black]`, `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`
- All required runtime dependencies listed
- Python requirement >= 3.11

```bash
grep "requires-python" pyproject.toml
# Expected: requires-python = ">=3.11"

grep "name = " pyproject.toml
# Expected: name = "req-eng-platform"
```

---

### ✅ AC3: Python Version Pinned to 3.11
**Given** project configured
**When** I check `.python-version` file
**Then** it contains exactly `3.11`

```bash
cat .python-version
# Expected: 3.11

uv python list
# Expected: Shows Python 3.11 available
```

---

### ✅ AC4: Virtual Environment Created
**Given** `.venv` does not exist initially
**When** I run `uv venv`
**Then** `.venv/` directory exists with proper structure

```bash
ls -la .venv/
# Expected: bin/, lib/, pyvenv.cfg present

file .venv/bin/python
# Expected: symlink or executable to Python 3.11
```

---

### ✅ AC5: All Dependencies Installed via uv
**Given** virtual environment exists
**When** I run `uv sync --all-groups`
**Then** all packages are installed with correct versions

```bash
uv pip list | wc -l
# Expected: 50+ packages

uv pip show fastapi
# Expected: Package information displayed

python -c "import fastapi, pydantic, sqlalchemy; print('✅ All critical imports successful')"
# Expected: ✅ All critical imports successful
```

---

### ✅ AC6: Lock File Generated
**Given** dependencies added via uv
**When** I check for `uv.lock`
**Then** file exists with all dependency hashes

```bash
ls -lh uv.lock
# Expected: File exists, size > 50KB

head -5 uv.lock
# Expected: TOML format with version info

grep "fastapi" uv.lock
# Expected: fastapi with pinned version found
```

---

### ✅ AC7: Development Tools Functional
**Given** all dependencies synced
**When** I run dev tools
**Then** each returns version information without errors

```bash
uv run pytest --version
# Expected: pytest {VERSION}

uv run ruff --version
# Expected: ruff {VERSION}

uv run mypy --version
# Expected: mypy {VERSION}

uv run black --version
# Expected: black, {VERSION}
```

---

### ✅ AC8: Git Properly Configured
**Given** project repository
**When** I check git configuration
**Then** `.gitignore` includes `.venv/` and `.uv/`

```bash
grep -E "\.venv|\.uv" .gitignore
# Expected: Both .venv/ and .uv/ in .gitignore

git status
# Expected: .venv/ and uv.lock not shown as untracked
```

---

### ✅ AC9: Quick Setup Works
**Given** fresh clone of repository
**When** developer runs `bash scripts/setup-dev.sh` (or `.bat` on Windows)
**Then** complete environment is set up in < 5 minutes with no errors

```bash
bash scripts/setup-dev.sh

# Expected output:
# 🚀 Setting up development environment...
# ✅ uv is installed: uv {VERSION}
# 📌 Pinning Python to 3.11...
# 🔧 Creating virtual environment...
# 📦 Installing all dependencies...
# 🪝 Installing pre-commit hooks...
# ✅ Setup complete!
```

---

### ✅ AC10: Environment Auto-Activation Works
**Given** `.venv/` exists in project root
**When** I run `uv run python --version`
**Then** uv automatically uses `.venv` Python without manual activation

```bash
which python
# Expected: /path/to/project/.venv/bin/python

uv run python -c "import sys; print(sys.prefix)" | grep .venv
# Expected: Path includes .venv
```

---

## Definition of Done

- [ ] `uv` installed and version verified
- [ ] `pyproject.toml` created with all dependencies properly categorized
- [ ] `.python-version` file created with `3.11`
- [ ] `.venv/` virtual environment exists and is functional
- [ ] All runtime, dev, test, lint, and docs dependencies installed
- [ ] `uv.lock` file generated and committed to git
- [ ] `uv sync --all-groups` completes without errors
- [ ] All dev tools (pytest, ruff, mypy, black) are functional
- [ ] `.gitignore` properly configured for uv project
- [ ] Pre-commit hooks installed (optional but recommended)
- [ ] Setup scripts created and tested
- [ ] README updated with quick start instructions
- [ ] All 10 acceptance criteria verified and passing
- [ ] Fresh clone can complete setup in < 5 minutes

---

## Windsurf Implementation Notes

### Priority Order:
1. **Create `pyproject.toml`** - This is the configuration source of truth
2. **Install dependencies** - Use `uv add` commands in sequence
3. **Generate lock file** - Auto-generated, then verify
4. **Verify installation** - Run test commands
5. **Update documentation** - README and .gitignore

### Key uv Commands for Implementation:

```bash
# Create venv
uv venv

# Add dependencies (one command per group, or all at once)
uv add fastapi uvicorn pydantic ...  # Runtime
uv add --dev pytest ruff mypy ...    # Development
uv add --group test httpx faker      # Testing
uv add --group lint                  # Linting
uv add --group docs sphinx ...       # Documentation

# Sync environment
uv sync --all-groups

# Verify
uv pip list
python -c "import fastapi; import pydantic"
```

### Why uv vs pip:
- **10-100x faster** than pip
- **Unified tool** replaces pip + venv + pip-tools
- **Lock file support** for reproducibility
- **Universal resolution** works across all platforms
- **Python version management** built-in
- **Pre-commit friendly** - faster checks

### Potential Issues & Solutions:

| Issue | Solution |
|-------|----------|
| `command not found: uv` | Install uv first (see Task 1) |
| `Python 3.11 not found` | Run `uv python install 3.11` |
| `.venv not auto-detected` | Ensure `.venv` in project root, try `uv run python --version` |
| Lock file conflicts | Delete `uv.lock` and regenerate: `uv lock` |
| Slow installation | First run is slower; subsequent runs are cached |

---

## References

- **uv Official Documentation**: https://docs.astral.sh/uv/
- **Python 3.11 Release**: https://www.python.org/downloads/
- **PEP 735 (Dependency Groups)**: https://peps.python.org/pep-0735/
- **pyproject.toml Format**: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/

---

**End of UV Setup Subtask Document**

**Next Steps After Completion:**
- Proceed with Docker services setup (AC2 of Story 1)
- Database migration setup (AC3 of Story 1)
- FastAPI application development (AC5 of Story 1)
