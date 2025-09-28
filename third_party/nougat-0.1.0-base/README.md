<div align="center">
<h1>Nougat: Neural Optical Understanding for Academic Documents</h1>

[![Paper](https://img.shields.io/badge/Paper-arxiv.230x.xxxxx-red)](https://arxiv.org/abs/230x.xxxxx)
[![GitHub](https://img.shields.io/github/license/facebookresearch/nougat)](https://github.com/facebookresearch/nougat)
[![PyPI](https://img.shields.io/pypi/v/nougat-ocr?logo=pypi)](https://pypi.org/project/nougat-ocr)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

This is the official repository for Nougat, the academic document PDF parser that understands LaTeX math and tables.

Project page: https://facebookresearch.github.io/nougat/

## Install

From pip:
```
pip install nougat-ocr
```

From repository:
```
pip install git+https://github.com/facebookresearch/nougat
```

There are extra dependencies if you want to call the model from an API or generate a dataset.
Install via

`pip install "nougat-ocr[api]"` or `pip install "nougat-ocr[dataset]"`

### Get prediction for a PDF
#### CLI

To get predictions for a PDF run 

```$ nougat path/to/file.pdf```

```
usage: nougat [-h] [--batchsize BATCHSIZE] [--checkpoint CHECKPOINT] [--out OUT] pdf [pdf ...]

positional arguments:
  pdf                   PDF(s) to process.

optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE, -b BATCHSIZE
                        Batch size to use. Defaults to 6 which runs on 24GB VRAM.
  --checkpoint CHECKPOINT, -c CHECKPOINT
                        Path to checkpoint directory
  --out OUT, -o OUT     Output directory.
```

In the output directory every PDF will be saved as a `.mmd` file, the lightweight markup language, mostly compatible with [Mathpix Markdown](https://github.com/Mathpix/mathpix-markdown-it) (we make use of the LaTeX tables).

#### API

With the extra dependencies you use `app.py` to start an API. Call

```
$ nougat_api
```

To get a prediction of a PDF file by making a POST request to http://127.0.0.1:8503/predict/. It also accepts parameters `start` and `stop` to limit the computation to select page numbers (boundaries are included).

## Dataset
### Generate dataset

To generate a dataset you need 

1. A directory containing the PDFs
2. A directory containing the `.html` files (processed `.tex` files by [LaTeXML](https://math.nist.gov/~BMiller/LaTeXML/)) with the same folder structure
3. A binary file of [pdffigures2](https://github.com/allenai/pdffigures2) and a corresponding environment variable `export PDFFIGURES_PATH="/path/to/binary.jar"`

Next run

```
python -m nougat.dataset.split_htmls_to_pages --html path/html/root --pdfs path/pdf/root --out path/paired/output --figure path/pdffigures/outputs
```

Additional arguments include

| Argument              | Description                                |
| --------------------- | ------------------------------------------ |
| `--recompute`         | recompute all splits                       |
| `--markdown MARKDOWN` | Markdown output dir                        |
| `--workers WORKERS`   | How many processes to use                  |
| `--dpi DPI`           | What resoultion the pages will be saved at |
| `--timeout TIMEOUT`   | max time per paper in seconds              |
| `--tesseract`         | Tesseract OCR prediction for each page     |

Finally create a `jsonl` file that contains all the image paths, markdown text and meta information.

```
python -m nougat.dataset.create_index --dir path/paired/output --out index.jsonl
```

For each `jsonl` file you also need to generate a seek map for faster data loading:

```
python -m nougat.dataset.gen_seek file.jsonl
```

The resulting directory structure can look as follows:

```
root/
├── images
├── train.jsonl
├── train.seek.map
├── test.jsonl
├── test.seek.map
├── validation.jsonl
└── validation.seek.map
```

Note that the `.mmd` and `.json` files in the `path/paired/output` (here `images`) are no longer required.
This can be useful for pushing to a S3 bucket by halving the amount of files.

## Training

To train or fine tune a Nougat model, run 

```
python train.py --config config/train_nougat.yaml
```

## Evaluation

Run 

```
python test.py --checkpoint path/to/checkpoint --dataset path/to/test.jsonl --save_path path/to/results.json
```

To get the results for the different text modalities, run

```
python -m nougat.metrics path/to/results.json
```

## Citation

```
@misc{blecher_nougat_2023,
      doi = {10.48550/ARXIV.2303.xxxxx},
      url = {https://arxiv.org/abs/2303.xxxxx},
      author = {Blecher, Lukas and Cucurull, Guillem and Scialom, Thomas and Stojnic, Robert},
      title = {Nougat: Neural Optical Understanding for Academic Documents},
      publisher = {arXiv},
      year = {2023},
      copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Acknowledgments

This repository builds on top of the [Donut](https://github.com/clovaai/donut/) repository.

## License

Nougat is licensed under CC-BY-NC.
