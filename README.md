# Statement Extractor
### 💡 Info:
Statement Extractor is a python library to detect statements in texts automatically.The main modules can be found in the [src](https://github.com/jueri/statement_extractor/src) directory. The [example notebook](https://github.com/jueri/statement_extractor/blob/master/example.ipynb) in the root directory interface the modules and guide through the extraction process.

This repository is part of my bachelor theses with the title **Automated statement extraction from press briefings**.

### ⚙️ Setup:
This repository uses Pipenv to manage a virtual environment with all python packages. Information about how to install Pipenv can be found [here](https://pipenv.pypa.io/en/latest/).
To create a virtual environment and install all packages needed, call `pipenv install` from the root directory.

The [transformers](https://github.com/huggingface/transformers) library used for the BERT models needs a current [Rust](https://www.rust-lang.org/) installation.

Default directory and parameter can be defined in [config.py](https://github.com/jueri/statement_extractor/tree/master/config.py).
To set up, the system and download the necessary data, please run the [config.py](https://github.com/jueri/statement_extractor/tree/master/config.py) script from the root.

The wikification module relies on two wikification services, [Dandelion](https://dandelion.eu/) and [TagMe](https://sobigdata.d4science.org/web/tagme). API keys for these services can be created for free. The wikify module expects the environment variables `DANDELION_TOKEN` and `TAGME_TOKEN`. 

### Usage:
Besides from the different modules, a basic CLI interface is available to extract statements from press briefings. The following options are available.
```
Options:
  -i, --pdf_in TEXT        Path to the input pdf file.
  -o, --pdf_out TEXT       Output path.
  -n, --length INTEGER     Number of sentences per statement.
  -m, --main_concept TEXT  Method to detect the main concept for similarity
                           score calculation. Options are "embedding",
                           "wikify_title", "wikify_intro".
  --intro TEXT             Path to the introduction text to be used for
                           wikification.
  --main_concept_th FLOAT  Threshold for a minimum main concept similarity
  --claim_th FLOAT         Minimal claim confidence betweene 0.1 and 1.0, by
                           default the max confidence class is choosen.
  --help                   Show this message and exit.
```

### 📋 Content:
- [example.ipynb](https://github.com/jueri/statement_extractor/tree/master/example.ipynb) holds an example usecase and guides through the modules.
- [pre_predict_claims.ipynb](https://github.com/jueri/statement_extractor/tree/master/pre_predict_claims.ipynb) can be used to predict sentence based claim probabilitys for data annotation.

### Bugs:
After installation, please run `pip install -U PyMuPDF`
