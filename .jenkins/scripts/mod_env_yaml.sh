:
item='[ \t]*-[ \t]*'

# Extract full git+https URLs needed from tox.ini:
fits_storage=$(grep "FitsStorage.git" tox.ini | xargs)
pytest_dragons=$(grep "pytest_dragons.git" tox.ini | xargs)

# Substitute full URLs for git+https deps, omit packages installed by special
# means (dragons/AstroFaker) and remove specific install prefix:
sed -i.bak \
    -e "/^prefix:/d" \
    -e "/^${item}\(astrofaker\|dragons\)=/d" \
    -e "s|^\(${item}\)fits-storage=.*|\1${fits_storage}|" \
    -e "s|^\(${item}\)pytest-dragons=.*|\1${pytest_dragons}|" \
    jenkins_env.yaml

# Override the deps in tox.ini using the above YAML spec:
sed -i '/\[testenv\]/a\conda_env = jenkins_env.yaml' tox.ini

