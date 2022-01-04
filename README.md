# Statement Extractor
# The SMC claim dataset
### Info 💡
Statement Extractor is a python library to automatically detect statements in texts. The main modules can be found in the [src](https://github.com/jueri/statement_extractor/src) directory. Three notebooks in the root directory interface these modules and guide through the dataset creation process.

This repository is part of my bachelore theses with the title **Automated statement extractionfrom press briefings**.

### Setup 🎛
This repo holds a [Visual Studio Code (VS Code)](https://code.visualstudio.com/) [.devcontainer](https://github.com/jueri/statement_extractor/tree/master/.devcontainer). The docker development container can easily be recreated using VS Code.
Alternatively, can the dependencies be installed using with the following command:
`pip install -r .devcontainer/requirements.txt`

Default directorys and parameter can be defined in [config.py](https://github.com/jueri/statement_extractor/tree/master/config.py).
To setup the system and download necessary data please run the [config.py](https://github.com/jueri/statement_extractor/tree/master/config.py) script from root.

The wikification module relies on two wikification services, [Dandelion](https://dandelion.eu/) and [TagMe](https://sobigdata.d4science.org/web/tagme). API keys for these services can be created for free. The wikify module expects the environment variables `DANDELION_TOKEN` and `TAGME_TOKEN`.

### Run

### Content 📋
- [example.ipynb](https://github.com/jueri/statement_extractor/tree/master/example.ipynb) holds an example usecase and guides through the modules.
- [pre_predict_claims.ipynb](https://github.com/jueri/statement_extractor/tree/master/pre_predict_claims.ipynb) can be used to predict sentence based claim probabilitys for data annotation.

### Models
The `data/models` directory contains trained models.
|Name|Usecase|train_data|test_accuracy|
|---|---|---|---|
|`CM_gbert-base_full_g`|pre_label the SMC_claim_dataset|Google translated datasets: `IBM Debater® - Claims and Evidence`  and `IBM Debater® - Claim Sentences Search`|0.7274|