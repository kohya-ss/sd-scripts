.PHONY: all
all:

.PHONY: style
style: 
	ruff format --respect-gitignore

.PHONY: test_style
test_style:
	ruff format --respect-gitignore --check

.PHONY: test
test: test_style


.DELETE_ON_ERROR:
SHELL=/bin/bash
