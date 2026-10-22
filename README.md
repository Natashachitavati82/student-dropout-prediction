# Student Dropout Prediction (DSI235 Capstone)

Predicts whether a student will drop out, stay enrolled, or graduate.

- *Data:* Kaggle "Predict Students' Dropout and Academic Success" (4,424 students)
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
