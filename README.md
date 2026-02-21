# YouTube Video Popularity Prediction and Engagement Analysis

## Project Description

This project develops two machine learning models to predict video engagement metrics and identify key factors influencing YouTube video popularity. The models are trained on data collected from two different sources to compare data collection methodologies and model performance.

### Models

1. **Model 1 (Scraped Data)**: Trained on data collected via web scraping using Selenium - Predicts view count
2. **Model 2 (API Data)**: Trained on data collected via YouTube Data API v3 - Predicts engagement rate

The project demonstrates the differences between scraped and API-sourced data, compares model performance, and provides insights into what makes YouTube videos successful.

## Objectives

- Collect 3000+ YouTube videos from each source (web scraping and API)
- Build comprehensive preprocessing and feature engineering pipelines
- Train Random Forest regression models on both datasets
- Compare model performance and feature importance
- Visualize engagement trends by category, duration, and upload time
- Identify key factors that drive video popularity and engagement

## Data Sources

### 1. Web Scraping (Selenium)

- **Tool**: Selenium WebDriver with Chrome
- **Source**: YouTube search results pages
- **Collected Attributes**:
  - title - Video title
  - channel - Channel name
  - views - View count (parsed from text)
  - upload_date - Upload date (relative, e.g., "2 years ago")
  - duration - Video length (MM:SS format)
  - link - Video URL
- **Target Sample Size**: 3000+ videos

### 2. YouTube Data API v3

- **Tool**: Google API Python Client
- **Source**: YouTube Data API endpoints
- **Collected Attributes**:
  - videoId - Unique video identifier
  - title - Video title
  - description - Video description
  - publish_date - Exact publish timestamp (ISO 8601)
  - duration - Video length (ISO 8601 duration)
  - tags - Video tags/keywords
  - categoryId - YouTube category ID
  - viewCount - Total views
  - likeCount - Total likes
  - commentCount - Total comments
- **Target Sample Size**: 3000+ videos

## Installation and Setup

### Prerequisites

1. Python 3.8 or higher
2. Chrome browser (for web scraping)
3. YouTube API Key (obtain from Google Cloud Console)

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/ompatel/youtube-popularity-analysis.git
cd youtube-popularity-analysis

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:
```
YOUTUBE_API_KEY=your_api_key_here
```

To obtain a YouTube API key:
1. Visit Google Cloud Console (https://console.cloud.google.com/)
2. Create a new project or select existing project
3. Enable YouTube Data API v3
4. Create credentials (API Key)
5. Copy the API key to your .env file

## Project Structure

```
youtube-popularity-analysis/
├── data/
│   ├── raw/                      # Raw collected data
│   │   ├── scraped_videos.csv
│   │   └── api_videos.csv
│   └── processed/                # Preprocessed data
│       ├── scraped_processed.csv
│       └── api_processed.csv
│
├── src/                          # Source code modules
│   ├── scraping.py              # Web scraping functions
│   ├── youtube_api.py           # API data collection
│   ├── preprocess.py            # Data preprocessing
│   ├── features.py              # Feature engineering utilities
│   ├── train.py                 # Model training
│   ├── evaluate.py              # Model evaluation & visualization
│   └── utils.py                 # Helper functions
│
├── notebooks/                    # Jupyter notebooks (execution order)
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling_scraped.ipynb
│   ├── 04_modeling_api.ipynb
│   └── 05_evaluation_visualization.ipynb
│
├── models/                       # Trained models
│   ├── scraped_model.pkl
│   └── api_model.pkl
│
├── reports/
│   ├── figures/                 # Generated visualizations (8 plots)
│   └── FINAL_REPORT.md          # Detailed project report
│
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation
```

## How to Use

### Training Models

#### Option 1: Using Jupyter Notebooks (Recommended)

Run the notebooks in sequential order:

```bash
jupyter notebook
```

Then execute:
1. `01_data_collection.ipynb` - Collect data from both sources
2. `02_preprocessing.ipynb` - Clean and engineer features
3. `03_modeling_scraped.ipynb` - Train Model 1
4. `04_modeling_api.ipynb` - Train Model 2
5. `05_evaluation_visualization.ipynb` - Evaluate and visualize

#### Option 2: Using Python Scripts

```bash
# Collect data
python src/scraping.py          # Web scraping
python src/youtube_api.py       # API collection

# Preprocess data
python src/preprocess.py

# Train models
python src/train.py scraped     # Train Model 1
python src/train.py api         # Train Model 2

# Generate evaluation report
python src/evaluate.py
```

### Model Inference

To use trained models for predictions on new data:

```python
import joblib
import pandas as pd
from textblob import TextBlob

# Load model
model_package = joblib.load('models/scraped_model.pkl')
model = model_package['model']
scaler = model_package['scaler']
features = model_package['features']

# Prepare new data
title = "Top 10 Music Hits of 2024"
new_video = pd.DataFrame({
    'duration_minutes': [5.5],
    'time_since_upload_days': [30],
    'title_len': [len(title.split())],
    'title_char_len': [len(title)],
    'title_upper_ratio': [sum(1 for c in title if c.isupper()) / len(title)],
    'title_sentiment': [TextBlob(title).sentiment.polarity],
    'has_channel': [1],
    'is_music': [1],
    'is_gaming': [0],
    'is_educational': [0]
})

# Scale features and predict
X_scaled = scaler.transform(new_video)
prediction = model.predict(X_scaled)
print(f"Predicted views: {prediction[0]:,.0f}")
```

## Data Collection

### Collection Strategy

Both methods use diverse search queries to ensure variety across video categories:
- Trending videos 2024
- Popular music videos
- Gaming highlights
- Movie trailers
- Technology reviews
- Cooking recipes
- Fitness workouts
- Educational content
- Comedy videos
- Travel vlogs

### Sample Sizes

- **Scraped Data**: 3,066 videos after preprocessing
- **API Data**: 2,456 videos after preprocessing

## Data Preprocessing

### Scraped Data Cleaning

1. **Parse text views**: Convert "8.8B views", "1.5M views" to numeric values
2. **Parse relative dates**: Convert "2 years ago" to actual datetime
3. **Parse duration**: Convert "3:45" to minutes (3.75)
4. **Extract video ID**: Extract from YouTube URL
5. **Remove duplicates**: Based on video links
6. **Handle missing values**: Fill missing durations with median (5.0 minutes)

### API Data Cleaning

1. **Convert ISO 8601 duration**: Parse to minutes
2. **Parse ISO timestamps**: Convert to datetime objects
3. **Handle missing statistics**: Fill with 0 where appropriate
4. **Normalize numeric fields**: Ensure consistent data types
5. **Extract tag information**: Parse JSON tag arrays

### Feature Engineering

#### Scraped Data Features (10 total)

| Feature | Description |
|---------|-------------|
| duration_minutes | Video length in minutes |
| time_since_upload_days | Days since upload |
| title_len | Number of words in title |
| title_char_len | Number of characters in title |
| title_upper_ratio | Ratio of uppercase characters |
| title_sentiment | TextBlob polarity score (-1 to 1) |
| has_channel | Boolean indicating if channel name is available |
| is_music | Boolean flag for music-related content |
| is_gaming | Boolean flag for gaming-related content |
| is_educational | Boolean flag for educational content |

#### API Data Features (13+ total, after one-hot encoding)

| Feature | Description |
|---------|-------------|
| duration_minutes | Video length in minutes |
| time_since_upload_days | Days since upload |
| views_per_day | Average views per day |
| engagement_rate | (likes + comments) / views |
| like_rate | likes / views |
| comment_rate | comments / views |
| title_len | Number of words in title |
| title_char_len | Number of characters in title |
| title_upper_ratio | Ratio of uppercase characters |
| desc_len | Number of words in description |
| desc_char_len | Number of characters in description |
| num_tags | Number of tags |
| categoryId | YouTube category (one-hot encoded) |

### Sample Data After Preprocessing

#### Scraped Data Example:
```
title: "Despacito ft. Daddy Yankee"
views: 8,800,000,000
duration_minutes: 5.0
time_since_upload_days: 2,920
views_per_day: 3,013,698
```

#### API Data Example:
```
videoId: "dQw4w9WgXcQ"
title: "Never Gonna Give You Up"
views: 1,696,931,258
likes: 15,000,000
engagement_rate: 0.0088
duration_minutes: 3.55
num_tags: 12
```

## Model Development and Evaluation

### Train and Test Data Partition

- **Split Ratio**: 80% training, 20% testing
- **Method**: Random split with random_state=42 for reproducibility
- **No stratification**: Regression task

### Model 1: Scraped Data Model

#### Algorithm
- **Model**: Random Forest Regressor
- **Hyperparameters**:
  - n_estimators: 200
  - max_depth: 20
  - min_samples_split: 5
  - min_samples_leaf: 2
  - random_state: 42

#### Input Specifications
- **Target Variable**: views (total view count)
- **Number of Features**: 10
- **Training Samples**: 2,452 videos (80%)
- **Testing Samples**: 614 videos (20%)
- **Note**: `views_per_day` excluded from features — it is derived from the target (views ÷ days since upload), which would cause data leakage.

#### Performance Metrics

**Training Data:**
- RMSE: 224,933,036
- MAE: 52,378,711
- R-squared: 0.7938

**Cross-Validation (5-fold):**
- Mean R²: 0.337 ± 0.085

**Test Data:**
- RMSE: 379,735,456
- MAE: 105,035,828
- R-squared: 0.4122

### Model 2: API Data Model

#### Algorithm
- **Model**: Random Forest Regressor
- **Hyperparameters**: Same as Model 1

#### Input Specifications
- **Target Variable**: engagement_rate = (likes + comments) / views
- **Number of Features**: 18 (after one-hot encoding categoryId)
- **Training Samples**: 1,964 videos (80%)
- **Testing Samples**: 492 videos (20%)

#### Performance Metrics

**Training Data:**
- RMSE: 0.0065
- MAE: 0.0038
- R-squared: 0.8506

**Cross-Validation (5-fold):**
- Mean R²: 0.363 ± 0.042

**Test Data:**
- RMSE: 0.0143
- MAE: 0.0090
- R-squared: 0.3869

## Feature Importance

### Technique

Feature importance is calculated using scikit-learn's built-in `feature_importances_` attribute from the Random Forest model. This method measures the mean decrease in impurity (Gini importance), quantifying how much each feature contributes to reducing prediction error across all decision trees.

### Top Features - Scraped Model

1. time_since_upload_days - 0.2707 (27.07%)
2. is_music - 0.2120 (21.20%)
3. title_upper_ratio - 0.1673 (16.73%)
4. title_char_len - 0.1532 (15.32%)
5. title_sentiment - 0.1146 (11.46%)

### Top Features - API Model

1. time_since_upload_days - 0.2922 (29.22%)
2. views_per_day - 0.1336 (13.36%)
3. title_upper_ratio - 0.0992 (9.92%)
4. desc_len - 0.0822 (8.22%)
5. title_char_len - 0.0726 (7.26%)

## Visualization

The project generates 8 comprehensive visualizations:

### 1. Model Comparison
Side-by-side comparison of R-squared scores, MAE, and feature counts between both models.

**File**: `reports/figures/model_comparison.png`

### 2-3. Actual vs Predicted Plots
Scatter plots showing prediction accuracy with perfect prediction reference line.

**Files**: 
- `reports/figures/scraped_actual_vs_predicted.png`
- `reports/figures/api_actual_vs_predicted.png`

### 4-5. Residual Plots
Analysis of prediction errors to identify systematic bias or patterns in model mistakes.

**Files**:
- `reports/figures/scraped_residuals.png`
- `reports/figures/api_residuals.png`

### 6-7. Feature Importance
Horizontal bar charts showing the top 15 most important features for each model.

**Files**:
- `reports/figures/scraped_feature_importance.png`
- `reports/figures/api_feature_importance.png`

### 8. Engagement Trends (API Data)
Four-panel visualization analyzing engagement patterns:
1. Engagement rate by category
2. Engagement rate by video duration
3. Average views by category
4. Engagement rate by video age

**File**: `reports/figures/engagement_trends.png`

## Key Findings

### Model Performance

**Scraped Model:**
- Moderate predictive performance (R² = 0.41 on test, CV mean = 0.34)
- Predicts views from content signals alone — no data leakage
- Feature importance balanced across 5+ meaningful features

**API Model:**
- Solid performance with full dataset (2,456 videos, R² = 0.39 on test)
- More balanced feature importance across 10+ content and metadata signals
- CV mean R² = 0.36 confirms stable generalization

### Important Factors for Video Success

1. **Views per day**: Most important predictor of total success
2. **Time since upload**: Older videos accumulate more total views
3. **Title characteristics**: Length and formatting impact discoverability
4. **Content category**: Music and entertainment show higher engagement
5. **Video duration**: Default 5-minute duration typical for popular content

## Challenges Encountered

### 1. Web Scraping

- **Dynamic content loading**: Required Selenium with scroll delays
- **Rate limiting**: Implemented delays between requests  
- **Inconsistent data**: Not all videos display same fields (especially duration)
- **Solution**: Robust error handling, retry logic, default value imputation

### 2. Data Quality

- **Missing values**: Especially duration field in scraped data
- **Outliers**: Viral videos skew distributions significantly
- **Text parsing**: Converting "8.8B" to numeric values
- **Solution**: Data cleaning, outlier detection, robust parsing functions

### 3. API Limitations

- **Quota restrictions**: 10,000 units per day limiting data collection
- **Pagination**: Maximum 50 results per request
- **Solution**: Efficient batching, multiple search queries, caching

### 4. Model Training

- **Different targets**: Views vs engagement rate comparison
- **Feature alignment**: Different available features per source
- **Small API dataset**: Only 50 videos collected vs target 3000+
- **Solution**: Separate models with source-specific features

### 5. Compatibility

- **Scikit-learn versions**: Older versions lack certain parameters
- **XGBoost OpenMP dependency**: macOS requires libomp installation
- **Solution**: Backward-compatible code, optional dependencies

## Ethical and Legal Considerations

### Legal Compliance

**YouTube Terms of Service**: 
- Using official API complies with YouTube ToS
- Web scraping limited to publicly available data
- No circumvention of access controls
- Respecting rate limits and server resources

**Data Privacy**:
- Only collecting public video metadata
- No personal user information accessed
- No private or unlisted videos included

**Attribution**:
- Proper citation of data sources
- Acknowledgment of YouTube as platform

### Ethical Considerations

**Responsible Data Use**:
- Educational research purpose only
- No manipulation or spam activities
- No privacy violations

**Rate Limiting**:
- Delays implemented between requests
- Respecting server resources
- Not overloading YouTube infrastructure

**Transparency**:
- Open source code
- Documented methodology
- Reproducible results

**Acknowledged Limitations**:
- Web scraping may conflict with future ToS changes
- Data represents snapshot in time
- Results don't represent all YouTube content

## Recommendations for Improvement

### 1. Data Collection
- **Increase sample size**: Collect full 3000+ API videos
- **Historical tracking**: Monitor videos over time for trend analysis
- **Geographic diversity**: Collect from multiple regions
- **Category balance**: Ensure representation across all categories

### 2. Model Performance
- **Hyperparameter tuning**: Use grid search or Bayesian optimization
- **Alternative algorithms**: Try XGBoost, LightGBM, or neural networks
- **Ensemble methods**: Combine multiple model predictions
- **Cross-validation**: K-fold validation for robust metrics

### 3. Feature Engineering
- **Natural language processing**: Sentiment analysis on titles/descriptions
- **Thumbnail analysis**: Extract image features using CNN
- **Channel features**: Incorporate subscriber count, upload frequency
- **Temporal features**: Day of week, seasonality effects
- **Video tags analysis**: Topic modeling on tag clusters

### 4. Evaluation
- **Time-based split**: Test on videos uploaded after training period
- **SHAP values**: More interpretable feature importance
- **Error analysis**: Deep dive into prediction failures
- **Business metrics**: Translate R-squared to actionable insights

### 5. Deployment
- **Web application**: Interactive prediction interface
- **REST API**: Endpoint for real-time predictions
- **Automated pipeline**: Scheduled data collection and retraining
- **Monitoring**: Track model performance drift over time

## Dependencies

See `requirements.txt` for complete list:

```
pandas
numpy
scikit-learn
xgboost
google-api-python-client
selenium
webdriver-manager
isodate
matplotlib
seaborn
joblib
python-dotenv
```

## License

MIT License — free to use, modify, and distribute with attribution.

## Contact

Questions or collaboration opportunities? Reach out via [GitHub](https://github.com/ompatel).

---

**Note**: Performance metrics reported are based on actual model training results. API model performance is limited by small sample size (50 videos vs. target 3000+).
