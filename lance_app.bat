@echo off
echo Installation des dependances...
py -m pip install streamlit pandas numpy scikit-learn

echo Lancement de l'application...
py -m streamlit run car_price_app.py

pause