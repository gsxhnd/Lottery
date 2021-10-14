SHELL := /bin/bash

download_data:
	python3 ./get_train_data.py

.PHONY: download_data