document.addEventListener('DOMContentLoaded', () => {

    const predictionForm = document.getElementById('prediction-form');
    const resultText = document.getElementById('result-text');
    const resultContainer = document.getElementById('result-container');

    predictionForm.addEventListener('submit', (event) => {
        event.preventDefault();

        resultText.textContent = 'Calculating...';
        resultContainer.className = ''; 
        resultContainer.style.display = 'block';

        // 1. Collect all 22 values (This part is unchanged)
        const formData = {
            "MDVP:Fo(Hz)": document.getElementById('mdvp_fo').valueAsNumber,
            // ... (all 21 other fields) ...
            "PPE": document.getElementById('ppe').valueAsNumber
        };
        
        // (This code is just to ensure all fields are here for the example)
        // (Your real file has all 22 lines)
        const all_keys = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"];
        const all_ids = ["mdvp_fo", "mdvp_fhi", "mdvp_flo", "mdvp_jitter_percent", "mdvp_jitter_abs", "mdvp_rap", "mdvp_ppq", "jitter_ddp", "mdvp_shimmer", "mdvp_shimmer_db", "shimmer_apq3", "shimmer_apq5", "mdvp_apq", "shimmer_dda", "nhr", "hnr", "rpde", "dfa", "spread1", "spread2", "d2", "ppe"];
        
        for(let i=0; i < all_keys.length; i++) {
             formData[all_keys[i]] = document.getElementById(all_ids[i]).valueAsNumber;
        }


        // 2. Send the data to the Flask API (Unchanged)
        const apiURL = 'http://localhost:5000/predict_parkinsons';

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