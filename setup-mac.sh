brew install python pipenv
python3 -m venv .venv ||:
source .venv/bin/activate ||:
python -m pip install -e ./git_petals

