# Brain-Disorder-prediction-using-MRI-data

s1: run python part1_train_model.py
then 
Run the Back-End (Flask Server):
Open a terminal.

Go to the directory where your app.py, parkinsons_model.joblib, and data_scaler.joblib files are.

Run the command: python app.py

Keep this terminal open! It's now running your server on http://127.0.0.1:5000.
Run the Front-End (Web Page):
   Open a new, separate terminal.
   Go to the folder where you saved index.html, style.css, and script.js.
   Run this command to start a simple local web server: python -m http.server 8000
   Keep this terminal open too!
Use Your App!
   Open your web browser (like Chrome or Firefox).
   Go to this address: http://localhost:8000
    You should see your form. Fill it out (you can use sample data from your parkinsons.csv) and click "Predict"!
