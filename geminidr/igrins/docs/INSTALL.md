# Installing IGRINSDR

## Prerequisites
- DRAGONS v4.1 (see [installation instructions](https://dragons.readthedocs.io/projects/recipe-system-users-manual/en/v4.1.0/install.html))

## Dependencies

IGRINSDR has only one required dependency:
- DRAGONS v4.1 or higher (already listed in Prerequisites)

All other dependencies are optional or will be automatically installed as part of DRAGONS.

## Installation Methods

### 1. Install  from pypi (recommended)
```bash
pip install igrinsdr
```

### 2. Install directly from GitHub (recommended)

The easiest way to install IGRINSDR is directly from the GitHub repository. This will install the latest development version from the `dev` branch:

```bash
pip install git+https://github.com/GeminiDRSoftware/IGRINSDR.git@dev
```

### 3. Install from a local clone

If you want to modify the source code or contribute to the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/GeminiDRSoftware/IGRINSDR.git
   cd IGRINSDR
   ```

2. Switch to the `dev` branch (recommended):
   ```bash
   git checkout dev
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```
   or for a regular installation:
   ```bash
   pip install .
   ```

## Verifying the Installation

After installation, you can verify that IGRINSDR was installed correctly by running:

```bash
python -c "import igrinsdr; print('IGRINSDR version:', igrinsdr.__version__)"
```

## Updating IGRINSDR

If you installed from GitHub and want to update to the latest version:

```bash
pip install --upgrade git+https://github.com/GeminiDRSoftware/IGRINSDR.git@dev
```

For a local installation, pull the latest changes and reinstall:

```bash
git pull origin dev
pip install -e .  # for development mode
# or
pip install .     # for regular installation
```
