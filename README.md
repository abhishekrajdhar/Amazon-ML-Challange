🧠 Model Documentation — Price Prediction Challenge
1. Methodology

We developed an end-to-end machine learning pipeline to predict product prices using multimodal features (text + numeric).
The process included:

Batch-wise data loading to handle memory efficiently (batch1.csv … batch13.csv).

Text parsing and cleaning of catalog_content into structured components.

Feature engineering (linguistic, numeric, and derived statistical features).

TF-IDF vectorization for textual data.

LightGBM GPU training for regression on log-transformed prices.

Inference on test batches and generation of submission file.

2. Model Architecture / Algorithms

We used LightGBM (GPU-accelerated) regression model with the following setup:

Objective: regression

Metric: Mean Absolute Error (MAE)

Boosting: Gradient Boosted Decision Trees (GBDT)

Tree learner: GPU-based histogram algorithm (gpu_hist)

Early stopping for overfitting control

Target: log_price = log1p(price)

This choice offered the best trade-off between speed, accuracy, and interpretability.

3. Feature Engineering Techniques

From catalog_content, we extracted and cleaned multiple features:

Feature	Description
title, description	Parsed from concatenated catalog text
IPQ (Item Pack Quantity)	Extracted numeric quantity (e.g., “250ml”, “2kg”)
len_title	Length of product title
num_digits	Count of numeric digits in text
has_brand_keyword	Binary flag if brand-related words appear
num_tokens	Token count after cleaning
log_price	Log-transformed target for stability

Text preprocessing steps:

Lowercasing and Unicode normalization

Removing duplicated whitespace

Preserving digits and measurement units (e.g., “250ml”)

Text vectorization:

TF-IDF (max_features = 10,000) trained on combined title + description.

Numerical features were scaled using StandardScaler before model input.

4. Final Training Setup
Component	Detail
Framework	LightGBM
Training Data	Combined 13 training batches
Validation Split	80/20 train/validation
GPU	Enabled (device='gpu')
Early Stopping Rounds	100
Best Validation MAE	~11.95
SMAPE: 54.88%

Final model and vectorizer were serialized using joblib for reproducibility:

joblib.dump(model, "final_model_lgb_gpu.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

5. Output Format

Predictions were generated for all test batches and saved as:

output/submission.csv


with the following structure:

id	price
test_0001	249.50
test_0002	125.00
6. Additional Notes

Future improvements may include multimodal fusion using image embeddings.

We also ensured reproducibility and consistent preprocessing across batches.

The entire pipeline runs seamlessly on both CPU and GPU environments.
