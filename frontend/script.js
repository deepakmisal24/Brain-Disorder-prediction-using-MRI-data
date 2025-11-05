// Wait for the HTML document to finish loading
document.addEventListener('DOMContentLoaded', () => {

    // Get references to our form and result elements
    const predictionForm = document.getElementById('prediction-form');
    const resultText = document.getElementById('result-text');
    const resultContainer = document.getElementById('result-container');

    // Add a 'submit' event listener to the form
    predictionForm.addEventListener('submit', (event) => {
        // Prevent the form from actually submitting (which reloads the page)
        event.preventDefault();

        // Show a loading message
        resultText.textContent = 'Calculating...';
        resultContainer.className = ''; // Reset styling

        // 1. Collect all 22 values from the form
        // We use 'valueAsNumber' to get numbers, not strings
        const formData = {
            "MDVP:Fo(Hz)": document.getElementById('mdvp_fo').valueAsNumber,
            "MDVP:Fhi(Hz)": document.getElementById('mdvp_fhi').valueAsNumber,
            "MDVP:Flo(Hz)": document.getElementById('mdvp_flo').valueAsNumber,
            "MDVP:Jitter(%)": document.getElementById('mdvp_jitter_percent').valueAsNumber,
            "MDVP:Jitter(Abs)": document.getElementById('mdvp_jitter_abs').valueAsNumber,
            "MDVP:RAP": document.getElementById('mdvp_rap').valueAsNumber,
            "MDVP:PPQ": document.getElementById('mdvp_ppq').valueAsNumber,
            "Jitter:DDP": document.getElementById('jitter_ddp').valueAsNumber,
            "MDVP:Shimmer": document.getElementById('mdvp_shimmer').valueAsNumber,
            "MDVP:Shimmer(dB)": document.getElementById('mdvp_shimmer_db').valueAsNumber,
            "Shimmer:APQ3": document.getElementById('shimmer_apq3').valueAsNumber,
            "Shimmer:APQ5": document.getElementById('shimmer_apq5').valueAsNumber,
            "MDVP:APQ": document.getElementById('mdvp_apq').valueAsNumber,
            "Shimmer:DDA": document.getElementById('shimmer_dda').valueAsNumber,
            "NHR": document.getElementById('nhr').valueAsNumber,
            "HNR": document.getElementById('hnr').valueAsNumber,
            "RPDE": document.getElementById('rpde').valueAsNumber,
            "DFA": document.getElementById('dfa').valueAsNumber,
            "spread1": document.getElementById('spread1').valueAsNumber,
            "spread2": document.getElementById('spread2').valueAsNumber,
            "D2": document.getElementById('d2').valueAsNumber,
            "PPE": document.getElementById('ppe').valueAsNumber
        };

        // 2. Send the data to the Flask API
        // This is the URL of the server we built in Part 2
        const apiURL = 'http://127.0.0.1:5000/predict';

        fetch(apiURL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData) // Convert the JS object to a JSON string
        })
        .then(response => {
            // Check if the request was successful
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            return response.json(); // Parse the JSON response
        })
        .then(data => {
            // 3. Display the prediction
            // 'data' is the JSON object our API sent back
            // e.g., {'prediction': 1, 'confidence': 0.85}
            
            let confidencePercent = (data.confidence * 100).toFixed(2);
            
            if (data.prediction === 1) {
                resultText.textContent = `Result: Positive for Parkinson's (Confidence: ${confidencePercent}%)`;
                resultContainer.className = 'result-positive';
            } else {
                resultText.textContent = `Result: Negative for Parkinson's (Confidence: ${confidencePercent}%)`;
                resultContainer.className = 'result-negative';
            }
        })
        .catch(error => {
            // 4. Handle any errors
            console.error('Error:', error);
            resultText.textContent = 'Prediction failed. Is the server running?';
            resultContainer.className = 'result-error';
        });
    });
});