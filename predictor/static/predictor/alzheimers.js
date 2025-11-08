document.addEventListener('DOMContentLoaded', function () {
    const showImageFormBtn = document.getElementById('show-image-form');
    const showNumericalFormBtn = document.getElementById('show-numerical-form');
    const imageForm = document.getElementById('image-form');
    const numericalForm = document.getElementById('numerical-form');
    const mriImageInput = document.getElementById('mri-image');
    const imagePreview = document.getElementById('image-preview');
    const resultContainer = document.getElementById('prediction-result');
    const resultText = document.getElementById('result-text');
    const confidenceScoreDiv = document.getElementById('confidence-score');
    const confidenceText = document.getElementById('confidence-text');

    // Function to switch between forms
    function switchForm(activeBtn, inactiveBtn, activeForm, inactiveForm) {
        activeBtn.classList.add('active');
        inactiveBtn.classList.remove('active');
        activeForm.classList.add('active-form');
        inactiveForm.classList.remove('active-form');
        resultContainer.style.display = 'none'; // Hide result when switching
    }

    showImageFormBtn.addEventListener('click', () => {
        switchForm(showImageFormBtn, showNumericalFormBtn, imageForm, numericalForm);
    });

    showNumericalFormBtn.addEventListener('click', () => {
        switchForm(showNumericalFormBtn, showImageFormBtn, numericalForm, imageForm);
    });

    // Event listener for image preview
    mriImageInput.addEventListener('change', function () {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            imagePreview.innerHTML = ''; // Clear previous preview
            reader.onload = function (e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                imagePreview.appendChild(img);
            };
            reader.readAsDataURL(file);
        } else {
            imagePreview.innerHTML = '<span class="image-preview-text">Image Preview</span>';
        }
    });

    // --- AJAX Form Submission ---
    async function handleFormSubmit(event) {
        event.preventDefault(); // Prevent default page reload
        const form = event.target;
        const formData = new FormData(form);
        
        // Show a loading state
        resultContainer.style.display = 'block';
        resultText.textContent = 'Analyzing...';
        confidenceScoreDiv.style.display = 'none';

        try {
            const response = await fetch(form.action, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': formData.get('csrfmiddlewaretoken'),
                },
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'An unknown error occurred.');
            }

            const data = await response.json();

            // Display the result
            resultText.textContent = data.prediction;
            if (data.confidence) {
                confidenceText.textContent = data.confidence;
                confidenceScoreDiv.style.display = 'block';
            }

        } catch (error) {
            resultText.textContent = `Error: ${error.message}`;
            confidenceScoreDiv.style.display = 'none';
        }
    }

    imageForm.addEventListener('submit', handleFormSubmit);
    numericalForm.addEventListener('submit', handleFormSubmit);
});
