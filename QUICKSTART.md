# Quick Start Guide

## Setup in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure YouTube API Key

1. Visit Google Cloud Console: https://console.cloud.google.com/
2. Create a new project
3. Enable "YouTube Data API v3"
4. Create credentials - API Key
5. Create a `.env` file in the project root:

```
YOUTUBE_API_KEY=your_api_key_here
```

### Step 3: Run the Complete Pipeline

#### Using Jupyter Notebooks (Recommended)

```bash
# Start Jupyter
jupyter notebook

# Execute notebooks in order:
# 1. notebooks/01_data_collection.ipynb
# 2. notebooks/02_preprocessing.ipynb  
# 3. notebooks/03_modeling_scraped.ipynb
# 4. notebooks/04_modeling_api.ipynb
# 5. notebooks/05_evaluation_visualization.ipynb
```

#### Using Python Scripts

```bash
# Collect data (takes approximately 1 hour)
cd src
python scraping.py          # Collect via web scraping
python youtube_api.py       # Collect via API

# Preprocess data
python preprocess.py

# Train models
python train.py scraped     # Train Model 1
python train.py api         # Train Model 2

# Evaluate and visualize
python evaluate.py
```

## Expected Output

After running the complete pipeline:

### Data Files
```
data/
├── raw/
│   ├── scraped_videos.csv      # 3000+ videos from web scraping
│   └── api_videos.csv          # 3000+ videos from API
└── processed/
    ├── scraped_processed.csv   # Cleaned and engineered features
    └── api_processed.csv       # Cleaned and engineered features
```

### Trained Models
```
models/
├── scraped_model.pkl    # Model 1: Predicts view count
└── api_model.pkl        # Model 2: Predicts engagement rate
```

### Visualizations
```
reports/figures/
├── model_comparison.png
├── scraped_actual_vs_predicted.png
├── api_actual_vs_predicted.png
├── scraped_residuals.png
├── api_residuals.png
├── scraped_feature_importance.png
├── api_feature_importance.png
└── engagement_trends.png
```

## Testing the Pipeline

To test without collecting new data:

```bash
# Use existing data files and run preprocessing/training only
cd src
python preprocess.py
python train.py
python evaluate.py
```

## Troubleshooting

### Web Scraping Issues
**Problem**: Browser doesn't open
**Solution**: Ensure Chrome is installed and set `headless=True` in code

### API Issues
**Problem**: "API key not found"
**Solution**: Verify `.env` file exists with correct format

### Module Import Errors
**Problem**: "ModuleNotFoundError: No module named 'src'"
**Solution**: Run from project root or ensure notebook adds parent directory to path

### XGBoost Error on macOS
**Problem**: "Library not loaded: libomp.dylib"
**Solution**: Install OpenMP: `brew install libomp`

## Time Estimates

- **Data Collection**: 
  - Web scraping: 30-60 minutes for 3000 videos
  - API: 10-15 minutes for 3000 videos

- **Preprocessing**: 2-3 minutes
- **Model Training**: 5-10 minutes total
- **Evaluation**: 5-10 minutes

## Tips

1. **Test with small sample first**: Use 100 videos to verify pipeline works
2. **Monitor API quota**: YouTube API has 10,000 units/day limit
3. **Use headless mode**: Set `headless=True` for background scraping
4. **Save intermediate results**: Don't delete CSV files

## Project Submission Checklist

- [ ] All 3000+ videos collected from each source
- [ ] Both models trained successfully
- [ ] All 8 visualizations generated
- [ ] Final report written
- [ ] Code pushed to GitHub
- [ ] Repository set to public
- [ ] README.md complete

