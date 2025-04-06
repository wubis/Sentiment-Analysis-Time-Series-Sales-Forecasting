# Sentiment-Analysis-Time-Series-Sales-Forecasting
Enhancing sales forecasting for Levi's jeans by integrating customer sentiments with time-series forecasting models

(In progress) This is the WIP of a deep learning time-series forecasting model augmented with sentiment analysis custom fine-tuned BERT model. The goal is to be able to predict the sales of Levi Jeans and their different models, and to do so, different trends must be predicted. While an LSTM is powerful, it can be further augmented by sentiment from reviews and comments from different websites. Hence, a base BERT model is fine-tuned on customer review comments and ratings to better understand where trends are going.

This concept can prove to be extremely useful for optimizing supply chains, preventing overstocking and understocking.

## Project Structure:
- Scrape data from e-commerce sites
- Use fuzzy match logic to match products from different webistes
- Train sentiment analysis model (BERT) on these reviews to extract sentiment scores
- Apply sentiment scores to time-series forecasting approaches (LSTM, SARIMA)

## Data Gathering
- Data of different jean products are webscraped from websites such as Levi's and Macy's, and then filtered to be matched with each other
- Example data shape: product_type, fit, fit_number, style, gender, size group
- 62 total different products selected
- 93,000+ reviews crawled

## Sentiment Analysis
Review Preprocessing:
- Translate non-english reviews, normalize emojis and symbols (e.g. heart = good, thumb down = bad)
- Tokenize data to feed into BERT

Training Methods:
- Compute loss with MSELoss to see how model is doing vs labeled dataset
- Adjust neural layer node weights with AdamW optimizer (stochastic gradient descent where weight decay is decoupled from gradient update)
- Decay learning rate for fine tuning with LinearLR
- Training: 41548, Validation: 10387, Test: 12984

Model Performance:
- MAE: 0.1328
- MSE: 0.0787
- RMSE: 0.2805
- R^2 Score: 0.8286
