# Student Dropout Prediction (DSI235 Capstone)

Predicts whether a student will drop out, stay enrolled, or graduate.

- *Data:* Kaggle "Predict Students' Dropout and Academic Success" (4,424 students), downloaded via Kaggle; original source is the UCI Machine Learning Repository
- *Model:* Random Forest (CV macro F1 0.72, dropout recall 0.74)
- *App:* Streamlit risk predictor

## How to run
1. pip install -r requirements.txt
2. python main.py  (trains and saves the model files)
3. streamlit run app.py

## Files
- main.py: data prep, feature engineering, modelling, saving
- app.py: Streamlit app
- *.pkl: saved model and preprocessing files
## Data source
Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021).
Predict Students' Dropout and Academic Success [Dataset].
UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/697
Licensed under CC BY 4.0. No changes were made to the original data.
