document.addEventListener('DOMContentLoaded', () => {

    const predictionForm = document.getElementById('prediction-form');
    const resultText = document.getElementById('result-text');
    const resultContainer = document.getElementById('result-container');

    predictionForm.addEventListener('submit', (event) => {
        event.preventDefault();

        resultText.textContent = 'Calculating...';
        resultContainer.className = '';
        resultContainer.style.display = 'block';

        // 1. Collect all 13 values (This part is unchanged)
        const formData = {
            "Mean": document.getElementById('mean').valueAsNumber,
            "Variance": document.getElementById('variance').valueAsNumber,
            "Standard Deviation": document.getElementById('std_dev').valueAsNumber,
            "Entropy": document.getElementById('entropy').valueAsNumber,
            "Skewness": document.getElementById('skewness').valueAsNumber,
            "Kurtosis": document.getElementById('kurtosis').valueAsNumber,
            "Contrast": document.getElementById('contrast').valueAsNumber,
            "Energy": document.getElementById('energy').valueAsNumber,
            "ASM": document.getElementById('asm').valueAsNumber,
            "Homogeneity": document.getElementById('homogeneity').valueAsNumber,
            "Dissimilarity": document.getElementById('dissimilarity').valueAsNumber,
            "Correlation": document.getElementById('correlation').valueAsNumber,
            "Coarseness": document.getElementById('coarseness').valueAsNumber
        };

        // 2. Send the data to the Flask API (Unchanged)
        const apiURL = 'http://localhost:5000/predict_braintumor';

        fetch(apiURL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // 3. Display the prediction (*** UPDATED ***)
            
            // Format confidence to a percentage (e.g., 0.95 -> 95.00%)
            let confidencePercent = (data.confidence * 100).toFixed(2);
            
            // Display label AND confidence
            resultText.textContent = `Result: ${data.label} (Confidence: ${confidencePercent}%)`;
            
            if (data.prediction === 1) {
                resultContainer.className = 'result-positive';
            } else {
                resultContainer.className = 'result-negative';
            }
        })
        .catch(error => {
            // 4. Handle any errors (Unchanged)
            console.error('Error:', error);
            resultText.textContent = 'Prediction failed. Is the server running?';
            resultContainer.className = 'result-error';
        });
    });
});