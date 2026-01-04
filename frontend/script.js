// CrediCheck AI - Backend Integration
// Backend integration for Day 2

const API_BASE_URL = 'http://127.0.0.1:5000';

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    const analyzeButton = document.getElementById('analyze-button');
    const textInput = document.getElementById('text-input');
    const imageInput = document.getElementById('image-input');
    
    if (analyzeButton) {
        analyzeButton.addEventListener('click', handleAnalyze);
    }
    
    // Handle image upload feedback
    if (imageInput) {
        imageInput.addEventListener('change', handleImageChange);
    }
});

function handleImageChange(event) {
    const imageInput = event.target;
    const imageStatus = document.getElementById('image-status');
    const imagePreview = document.getElementById('image-preview');
    const uploadArea = imageInput.closest('.flex.flex-col').querySelector('.rounded-lg.border-2');
    
    if (imageInput.files && imageInput.files[0]) {
        const file = imageInput.files[0];
        const fileName = file.name;
        
        // Show filename
        if (imageStatus) {
            imageStatus.innerHTML = `<p class="text-sm text-green-600 dark:text-green-400 font-medium">Image uploaded: ${fileName}</p>`;
        }
        
        // Show small preview
        if (imagePreview) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview" class="max-w-32 max-h-32 mx-auto rounded border border-border-light dark:border-border-dark">`;
            };
            reader.readAsDataURL(file);
        }
        
        // Hide placeholder text
        if (uploadArea) {
            const placeholderText = uploadArea.querySelector('p.text-base');
            const subText = uploadArea.querySelector('p.text-sm');
            if (placeholderText) placeholderText.style.display = 'none';
            if (subText) subText.style.display = 'none';
        }
    } else {
        // Clear status and preview
        if (imageStatus) {
            imageStatus.innerHTML = '';
        }
        if (imagePreview) {
            imagePreview.innerHTML = '';
        }
        
        // Restore placeholder text
        if (uploadArea) {
            const placeholderText = uploadArea.querySelector('p.text-base');
            const subText = uploadArea.querySelector('p.text-sm');
            if (placeholderText) placeholderText.style.display = 'block';
            if (subText) subText.style.display = 'block';
        }
    }
}

function handleAnalyze() {
    const textInput = document.getElementById('text-input');
    const imageInput = document.getElementById('image-input');
    const analyzeButton = document.getElementById('analyze-button');
    
    // Get input values
    const text = textInput ? textInput.value.trim() : '';
    const imageFile = imageInput ? imageInput.files[0] : null;
    
    // Validate input
    if (!text) {
        showError('Please enter text to analyze.');
        return;
    }
    
    // Disable button and show loading
    if (analyzeButton) {
        analyzeButton.disabled = true;
        analyzeButton.innerHTML = '<span class="loading"></span> Analyzing...';
    }
    
    // Prepare payload
    const payload = {
        text: text,
        headline: text // Use text as headline if not provided separately
    };
    
    // Add image path if file is selected
    if (imageFile) {
        payload.image_path = imageFile.name; // Use filename for now
    }
    
    // Send request to backend
    fetch(`${API_BASE_URL}/analyze-full`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || 'Backend request failed');
            });
        }
        return response.json();
    })
    .then(data => {
        displayResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
        showError(error.message || 'Failed to connect to backend. Make sure the server is running.');
    })
    .finally(() => {
        // Re-enable button
        if (analyzeButton) {
            analyzeButton.disabled = false;
            analyzeButton.innerHTML = '<span class="material-symbols-outlined">query_stats</span><span class="truncate">Analyze</span>';
        }
    });
}

function displayResults(data) {
    // Display text analysis (from aggregated result)
    const textAnalysisDiv = document.getElementById('text-analysis-result');
    if (textAnalysisDiv) {
        let html = '';
        if (data.final_label) {
            html += `<p class="font-semibold">Classification: ${data.final_label}</p>`;
        }
        if (data.final_trust_score !== undefined) {
            html += `<p>Trust Score: ${data.final_trust_score}%</p>`;
        }
        if (data.cache_hit !== undefined) {
            html += `<p class="text-sm mt-2">Cache: ${data.cache_hit ? 'Hit' : 'Miss'}</p>`;
        }
        textAnalysisDiv.innerHTML = html || '<p>No text analysis data available.</p>';
    }
    
    // Display image context (if available in response)
    const imageContextDiv = document.getElementById('image-context-result');
    if (imageContextDiv) {
        let html = '';
        // Note: /analyze-full doesn't return image_context directly, but we can show if image was analyzed
        if (data.reason && data.reason.includes('image')) {
            html += `<p>Image analysis performed</p>`;
        } else {
            html += `<p class="text-sm">No image provided</p>`;
        }
        imageContextDiv.innerHTML = html || '<p>No image analysis performed.</p>';
    }
    
    // Display source verification
    const sourceVerificationDiv = document.getElementById('source-verification-result');
    if (sourceVerificationDiv) {
        let html = '';
        if (data.reason) {
            html += `<p class="text-sm">${data.reason}</p>`;
        }
        if (data.response_time_ms !== undefined) {
            html += `<p class="text-xs mt-2 text-gray-500">Response: ${data.response_time_ms}ms</p>`;
        }
        sourceVerificationDiv.innerHTML = html || '<p>No verification data available.</p>';
    }
    
    // Display final verdict
    const finalVerdictDiv = document.getElementById('final-verdict-result');
    if (finalVerdictDiv) {
        let html = '';
        if (data.final_label) {
            html += `<p class="text-2xl font-bold mb-2">${data.final_label}</p>`;
        }
        if (data.final_trust_score !== undefined) {
            const scoreColor = data.final_trust_score >= 70 ? 'text-green-600 dark:text-green-400' : 
                             data.final_trust_score >= 50 ? 'text-yellow-600 dark:text-yellow-400' : 
                             'text-red-600 dark:text-red-400';
            html += `<p class="text-xl font-semibold ${scoreColor}">Trust Score: ${data.final_trust_score}%</p>`;
        }
        if (data.reason) {
            html += `<p class="text-base mt-2">${data.reason}</p>`;
        }
        finalVerdictDiv.innerHTML = html || '<p>No verdict data available.</p>';
    }
}

function showError(message) {
    // Remove existing error messages
    const existingError = document.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    // Create error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    
    // Insert before results section
    const resultsSection = document.querySelector('.space-y-8');
    if (resultsSection) {
        resultsSection.parentNode.insertBefore(errorDiv, resultsSection);
    } else {
        // Fallback: insert at top of main content
        const main = document.querySelector('main');
        if (main) {
            main.insertBefore(errorDiv, main.firstChild);
        }
    }
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

