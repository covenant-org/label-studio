#!/bin/bash

set -e
if [[ ! -d "label_studio" ]]; then
	echo "Run this script in the root of the project"
	exit 1
fi

VENV_FOLDER=".venv"
if [[ ! -d "$VENV_FOLDER" ]]; then
	VENV_FOLDER="venv"
	if [[ ! -d "$VENV_FOLDER" ]]; then
		echo "The virtual python enviroment does not exist in .venv"
		exit 1
	fi
fi

source "${VENV_FOLDER}/bin/activate"
ENV_FILE="$HOME/.local/share/label-studio/.env"
if [[ -f "${ENV_FILE}" ]]; then
	export $(cat "${ENV_FILE}" | xargs)
fi
python label_studio/manage.py runserver
