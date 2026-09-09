## Install project dependencies
install:
	pip install -r requirements.txt

## Filter raw NTRS metadata into candidates, then fetch download links
links:
	python filter_ntrs_metadata.py
	python fetch_ntrs_download_links.py

## Download all files concurrently - faster, more load on NTRS
download-fast:
	python pdf_downloader_fast.py

## Download all files sequentially with a delay between requests - slower, gentler on NTRS
download-slow:
	python pdf_downloader.py

## Train the GPT model on the downloaded corpus
train:
	python train.py

## Generate text from a trained checkpoint
generate:
	python generate.py