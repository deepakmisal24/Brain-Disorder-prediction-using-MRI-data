document.addEventListener('DOMContentLoaded', () => {

    const predictionForm = document.getElementById('prediction-form');
    const resultText = document.getElementById('result-text');
    const resultContainer = document.getElementById('result-container');

    predictionForm.addEventListener('submit', (event) => {
        event.preventDefault();

        resultText.textContent = 'Calculating...';
        resultContainer.className = '';
        resultContainer.style.display = 'block';

        // 1. Collect all 8 values from the form
        // We handle missing 'SES' and 'MMSE' by sending null
        const formData = {
            "M/F": Number(document.getElementById('m_f').value),
            "Age": document.getElementById('age').valueAsNumber,
            "EDUC": document.getElementById('educ').valueAsNumber,
            "SES": document.getElementById('ses').valueAsNumber || null, // Send null if empty
            "MMSE": document.getElementById('mmse').valueAsNumber || null, // Send null if empty
            "eTIV": document.getElementById('etiv').valueAsNumber,
            "nWBV": document.getElementById('nwbv').valueAsNumber,
            "ASF": document.getElementById('asf').valueAsNumber
        };

        // 2. Send the data to the Flask API
        const apiURL = 'http://localhost:5000/predict_alzheimers';

        fetch(apiURL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        })
        .then(response => {
            if (!response.ok) {
                // Try to get error message from server
                return response.json().then(err => {
                    throw new Error(err.error || `Server error: ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            // 3. Display the prediction
            let confidencePercent = (data.confidence * 100).toFixed(2);
            resultText.textContent = `Result: ${data.label} (Confidence: ${confidencePercent}%)`;
            
            if (data.prediction === 1) { // 1 = Demented
                resultContainer.className = 'result-positive';
            } else { // 0 = Nondemented
                resultContainer.className = 'result-negative';
            }
        })
        .catch(error => {
            // 4. Handle any errors
            console.error('Error:', error);
            resultText.textContent = `Prediction failed. ${error.message}`;
            resultContainer.className = 'result-error';
        });
    });
});