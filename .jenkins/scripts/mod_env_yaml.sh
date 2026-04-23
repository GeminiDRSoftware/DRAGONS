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

# Replace the deps in tox.ini with the above YAML spec:
python -c '
import configparser as cp
conf = cp.ConfigParser()
conf.read("tox.ini")
try:
    del conf["testenv"]["conda_deps"], conf["testenv"]["deps"]
except:
    pass
else:
    with open("tox.ini", "w") as cf:
        conf.write(cf)
'
sed -i '/\[testenv\]/a\conda_env = jenkins_env.yaml' tox.ini

