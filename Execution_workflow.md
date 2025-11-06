S1: Train the Models In your first terminal, navigate to your project directory and run all three training scripts.\
&nbsp;&nbsp;&nbsp;&nbsp;`python train_model_alzheimers.py`, `python train_model_braintumor.py`, `python train_model_parkinsons.py`
        
S2: Run the Backend API Open a new terminal (do not close the first one yet, just in case). \
&nbsp;&nbsp;&nbsp;&nbsp;Go to the same project directory (of `app.py`) and run the Flask application using the command `python app.py` on that new terminal. Keep this terminal running.\
&nbsp;&nbsp;&nbsp;&nbsp;It's now running your server on `http://127.0.0.1:5000` and `http://10.116.255.100:5000` open both and keep them open.
    
S3: Run the Frontend Server In your VSCode file explorer:\
    &nbsp;&nbsp;&nbsp;&nbsp;Right-click on your `frontend` folder.\
    &nbsp;&nbsp;&nbsp;&nbsp;Select `Open in Integrated Terminal`.\
    &nbsp;&nbsp;&nbsp;&nbsp;In the new terminal that opens (which is already in the frontend directory), run the following command `python -m http.server 8000`. Keep this terminal running.
    
S4: Use the Application\
    &nbsp;&nbsp;&nbsp;&nbsp;Open your web browser (like Chrome or Firefox).\
    &nbsp;&nbsp;&nbsp;&nbsp;Go to this address: `http://localhost:8000`



NOTE: If you make any changes to the program code then go to the web browser where the localhost is running then enter `ctrl + shift + R` this will refresh the page thus show the new changes
