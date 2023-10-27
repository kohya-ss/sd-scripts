.PHONY: all
all:

.PHONY: style
style: 
	black --exclude venv --exclude .venv .

.PHONY: test_style
test_style:
	find . -type d \( -name venv -o -name .venv \) -prune -o -type f -name "*.py" -print | xargs black --diff | diff /dev/null -

.PHONY: test
test: test_style


.DELETE_ON_ERROR:
SHELL=/bin/bash
