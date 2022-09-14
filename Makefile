SHELL := /bin/bash

download_data:
	if [ ! -d input ]; then mkdir input; fi
	python3 ./src/download_train_data.py

mod:
	pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

.PHONY: download_data