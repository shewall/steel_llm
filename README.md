## Steel CPP Literature Mining Pipeline

MSc dissertation project by Sevval Isikirik, supervised by Dr Ishwar Kapoor.

The idea of this project is to pull composition, processing and property (CPP) data out of steel research papers, using a large language model to help read the papers, and then use that data to train regression models that predict tensile strength, yield strength and elongation from the steel composition.

## How the pipeline works

The papers go through a few stages before they turn into a usable dataset:

1. **Collecting papers** — papers are found through CrossRef, and full text is downloaded from Elsevier and Springer.
2. **Turning papers into a common format** — the XML files from each publisher are parsed into the same JSON structure, so it does not matter which publisher a paper came from.
3. **Screening** — each paper is checked to see if it is actually about steel composition/processing/property data, and if that data can be pulled from the text or tables. Papers that are not relevant get dropped here.
4. **Finding evidence** — for papers that pass screening, the pipeline finds candidate pieces of evidence (numbers in tables, sentences in the text) and works out which material and which experimental condition each piece belongs to.
5. **Building observations** — each group of evidence gets checked again, against the original paper, before it is allowed to become a real data row. If a value cannot be traced back to the source, it does not go in.
6. **Building the dataset** — the checked observations get turned into one flat table, one row per experiment.
7. **Modelling** — regression models are trained on the final dataset to predict UTS, yield strength and elongation from composition.

The model used to help read the papers is `qwen3-next:80b-a3b-instruct-q4_K_M`, run through Ollama. It was run on a rented GPU (RunPod) because the model is too big to run on a normal laptop.

## Where things stand

The full pipeline has been run on 871 papers. Of those:
- 615 were dropped at the screening stage (not relevant, or nothing usable to extract)
- 198 went all the way through but did not end up with a usable strength/elongation value
- 51 papers produced at least one usable observation

That gave an initial dataset of 297 observations. After checking these papers by hand and removing the ones that were not actually steel (some papers turned out to be about aluminium, nickel alloys, etc.), 256 observations from 44 papers were left. This is the dataset used for the final results.

Five models were compared for each of the three targets (UTS, yield strength, elongation): Ridge regression, Random Forest, Gradient Boosting, SVR, and a simple "always predict the mean" baseline. They were evaluated with 5-fold cross-validation, grouped by paper (DOI), so that no paper appears in both the training and test set at the same time.

## Project layout

```
steel-llm-project/
├── data/
│   ├── raw/                  source XML files from Elsevier/Springer (not tracked in git)
│   ├── processed/            one JSON file per paper, common format
│   ├── screening_v2/         Stage 1 screening results
│   ├── evidence_v2/          Stage 2 evidence/grouping results
│   ├── extracted_v2/         Stage 3 observation results
│   ├── datasets_v2/          the final flat dataset (cpp_dataset_v2.csv)
│   ├── cloud_runs/           logs from running the pipeline on the full corpus
│   └── modelling_results/    saved model predictions, used by the notebooks
├── literature_pipeline/
│   ├── v2/                   the pipeline itself: screening, evidence, extraction, dataset building
│   ├── parsing/               code that turns publisher XML into the common JSON format
│   └── modelling/            shared code used by the modelling notebooks
├── notebooks/
│   └── modelling/            01–06, the notebooks that check, clean and model the dataset
├── scripts/                   small helper scripts for checking dataset/run status
├── docs/                       tables and a data file with the final numbers used in the write-up
└── figures/                   the figures used in the dissertation
```

## Running it

```bash
conda activate steel_llm
pip install -r requirements.txt
ollama serve
python -m literature_pipeline.v2.run_one_paper data/processed/<paper>.json
```

To run the whole corpus instead of one paper:

```bash
python -m literature_pipeline.v2.run_production_corpus
```

API keys for downloading papers go in a `.env` file (not tracked in git): `ELSEVIER_API_KEY`, `SPRINGER_API_KEY`. These are only needed if you want to download more papers — they are not needed to just run the pipeline on papers that are already downloaded.

Tests for the pipeline can be run with `pytest` (each module under `literature_pipeline/v2/` has its own `test_*.py` file, and none of them call the LLM or need a GPU).
