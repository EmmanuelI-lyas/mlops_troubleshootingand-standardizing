#!/bin/bash

echo "Running lint checks"
flake8 .

echo "Running tests"
pytest

echo "Running type checks"
mypy .

echo "Validation complete"