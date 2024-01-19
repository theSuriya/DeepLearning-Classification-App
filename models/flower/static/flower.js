// flower.js
document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");

    form.addEventListener("submit", function (event) {
        event.preventDefault();

        const formData = new FormData(form);
        fetch("/predict_flower", {
            method: "POST",
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            displayPredictionResult(data);
        })
        .catch(error => console.error("Error:", error));
    });

    // Image preview on file input change
    fileInput.addEventListener("change", function () {
        const file = fileInput.files[0];
        if (file) {
            displayImagePreview(file);
        }
    });
});

function displayPredictionResult(data) {
    const resultDiv = document.getElementById("predictionResult");
    resultDiv.innerHTML = `
        <p>Predicted: ${data.class}</p>
        <p>Confidence: ${data.confidence}%</p>
    `;
}

function displayImagePreview(file) {
    const body = document.body;

    const reader = new FileReader();
    reader.onload = function (e) {
        // Set the background image of the body to the dropped image
        body.style.backgroundImage = `url(${e.target.result})`;
        body.style.backgroundSize = "cover";
        body.style.backgroundPosition = "center";
        body.style.backgroundRepeat = "no-repeat";
    };

    reader.readAsDataURL(file);
}