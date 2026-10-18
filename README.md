# Used Car Price Estimator

Describe a used car in plain English and get an estimated asking price with a range. For example:

> 2016 Toyota Camry, 90k miles, sedan, front wheel drive, gas, automatic

The app answers with a price of about 12,300 dollars, a likely range of about 9,700 to 16,400 dollars, a short explanation, and a "What moves the price" breakdown (more miles, a newer year, a rebuilt title, and so on).

I built this as my capstone for the TripleTen AI/ML program. It combines a trained machine learning model with an LLM chat interface, and it covers the rest of the production path too: experiment tracking, tests, a config file, and a Docker image.

## How it works

```
your message
   -> LLM pulls out the car details as JSON (it leaves blanks blank, it never guesses)
   -> Python checks the details (required fields, ranges, allowed values)
   -> missing something?  ask a clarifying question
   -> out of scope?       say so and explain why
   -> trained LightGBM pipeline predicts the price
   -> Python writes the price line and the range
   -> LLM explains the numbers Python computed
```

The main design rule is that **Python makes the decisions and the LLM only handles language**.

- The LLM turns a message into fields, and it turns finished numbers into a plain explanation. It never decides whether a car can be priced.
- The price line is always written by Python.
- If the LLM's explanation contains a number that is not in the facts Python gave it, the app throws the explanation away and shows a plain summary written by Python instead. The same happens if the LLM call fails.
- Off-topic messages get a fixed reply.

Details gather across turns, so you can answer a follow-up question with just "automatic". If you mention a different make or model, the app starts over with a new car.

## The data

- Dataset: [Used Cars Dataset (Craigslist listings)](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) on Kaggle, CC0 license.
- 426,880 US listings, 26 columns (I load 15 of them).
- What I predict: `price`, the asking price on the listing.

Cleaning rules (all in `config.yaml`):

| Rule | Why |
| --- | --- |
| Keep price between 500 and 100,000 dollars | The raw data has zero prices and a 3.7 billion dollar listing |
| Keep years 1990 and newer, drop missing years | Older cars are rare and mostly collectibles |
| Cap odometer at 300,000 miles, drop missing | Typos like 1,111,111 miles |
| Drop readings under 100 miles for cars older than 2020 | Placeholders on older cars (new cars really do have low readings) |
| Drop `parts only` and `missing` title rows | Not everyday sales |

This takes 426,880 rows down to 363,909. Model names are free text (about 24,000 unique values), so I combine make and model into one feature and let the encoder group rare ones (fewer than 25 listings). That keeps an unseen model from crashing the app.

## Results

All numbers are on a held-out test set, in dollars. I tracked six runs on a 100,000 row sample with MLflow, then trained the final model on all 363,909 rows.

| Run | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| Ridge, raw price | 4,952 | 7,581 | 0.712 |
| Ridge, log price | 4,445 | 7,469 | 0.721 |
| Ridge, log price, with condition | 4,405 | 7,426 | 0.724 |
| Ridge, log price, rare models grouped at 100 | 4,882 | 7,936 | 0.685 |
| LightGBM, log price | 3,619 | 6,214 | 0.807 |
| LightGBM, log price, deeper | 3,370 | 5,941 | 0.823 |
| **Final: same LightGBM settings, all data** | **3,133** | **5,697** | **0.839** |

What I took from it:

- Predicting log price beat predicting raw price.
- Ridge underfits (train error is close to test error). LightGBM cuts the typical error by about 1,000 dollars.
- Grouping rare models at 100 listings hurt, so I kept 25.
- The median absolute percent error on the test set is 12.5 percent.

The price range the app shows is the 10th to 90th percentile of actual price divided by predicted price on the test set, so about 80 percent of test cars fell inside it. Rare and unseen models get a wider range because the model is less reliable on them.

## Quick start

The trained model is included in the repo, so you do not need the 1.4 GB dataset to run the app. You do need Python 3.12 and a free LLM API key.

```bash
git clone https://github.com/TB5698/used-car-price-estimator.git
cd used-car-price-estimator
python -m venv .venv
```

Activate the environment (PowerShell: `.venv\Scripts\Activate.ps1`, Mac or Linux: `source .venv/bin/activate`), then:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill it in:

```
LLM_API_KEY=your key
LLM_MODEL=model id
# LLM_BASE_URL=   (only for a provider other than Groq)
```

I use [Groq](https://console.groq.com/keys) (free, no card) with `openai/gpt-oss-120b`. Any OpenAI-compatible provider works if you set `LLM_BASE_URL`. Once your key is in `.env`, `python -m src.llm` lists the models it can use. Free limits change, so check the provider's docs. Your key stays in `.env`, which Git ignores.

Run the chat page:

```bash
streamlit run streamlit_app.py
```

Or chat in the terminal:

```bash
python -m src.app
```

## Run with Docker

```bash
docker build -t used-car-estimator .
docker run --rm -p 8501:8501 --env-file .env used-car-estimator
```

Then open http://localhost:8501. The first build takes a few minutes, mostly installing packages. The image installs only `requirements-app.txt`, runs as a normal user, and has a health check. The API key is passed in when the container runs and is never baked into the image.

## Run the tests

```bash
pytest tests/ -v
```

No API key or network is needed. The LLM is replaced with a scripted fake in the tests, including the three demo cases (a normal question, a missing-information case, and an off-topic question).

## Reproduce the training

You only need this if you want to retrain. It needs the full dataset.

1. Create a Kaggle API token (Kaggle Settings, API Tokens) and save it in your `.kaggle` folder in your home directory.
2. Download the data:
   ```bash
   kaggle datasets download -d austinreese/craigslist-carstrucks-data -p data/raw --unzip
   ```
   The file is tracked with DVC (`data/raw/vehicles.csv.dvc`). There is no DVC remote, so the Kaggle download is the source.
3. Run the pipeline from the repo root:
   ```bash
   python -m src.train          # six experiments on a 100,000 row sample, logged to MLflow
   python -m src.compare_runs   # compare the runs, sorted by test MAE
   python -m src.evaluate       # reload the best run and confirm its metrics
   python -m src.train_final    # train on all rows, save models/final_model and error stats
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

`train_final` rewrites `models/final_model/` and `models/error_stats.json`. Those files ship with the repo, so commit them again after retraining.

## Project layout

```
config.yaml            cleaning cutoffs and training settings
streamlit_app.py       the chat page
src/preprocess.py      cleaning rules and the sklearn preprocessing pipeline
src/train.py           experiments with MLflow tracking
src/compare_runs.py    compare logged runs
src/evaluate.py        reload the best run and check its metrics
src/train_final.py     final model on all rows
src/error_stats.py     test-set error ratios behind the price range
src/validate.py        required fields, ranges, allowed values, clarifying questions
src/llm.py             LLM client, field extraction and explanation prompts
src/estimate.py        loads the model, predicts, builds the range and drivers
src/app.py             conversation logic (also a terminal chat)
tests/                 unit tests for each piece
notebooks/             exploratory data analysis
models/                the final trained model and error stats
Dockerfile             app image
```

## Limitations

- **The prices are from 2021.** The listings were collected in 2021, so prices reflect that market, not today's. The app says it gives a typical asking price, not a sale price.
- **Newer cars are underpriced.** There are very few 2020 to 2022 cars in the data, so estimates for them run low. A 2022 Tesla Model 3 with 5,000 miles came out around 14,600 dollars, which is clearly too low.
- **Everyday cars only.** The app is trained on prices from 500 to 100,000 dollars and years 1990 and newer. Exotics, collectibles, cars over 300,000 miles and pre-1990 cars are out of scope.
- **Rare models are less reliable.** If a make and model is rare or unseen, the app flags it and widens the range.
- **Blank details change the answer a lot.** In this data a missing field acts like a signal on its own, so the app asks for year, make, model, mileage, body type, drive, fuel and transmission instead of guessing. Condition is optional, and title status defaults to clean (the app tells you when it assumes that).
- **Condition labels are messy.** Listings use words like "good", "excellent" and "like new" loosely, so condition only helps a little.
- **The app can be picky about names.** If it does not recognize a body type word (for example "roadster"), it asks you for the body type.
- **The model is not smooth.** It is a tree model, so two neighboring years can move the price in unexpected ways. The "What moves the price" section only shows changes that move the price in the expected direction.
- **The free LLM tier has limits.** If a shared key hits the daily limit, the app shows a Python-written summary or an error instead of crashing.

## What I would do next

- Add a live public link and a GitHub Actions workflow that runs the tests.
- Test a smaller `min_frequency` for rare models, and try adding state as a feature.
- Add more body type aliases, such as roadster and cabriolet.
