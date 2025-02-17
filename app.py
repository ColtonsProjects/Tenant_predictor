from flask import Flask, jsonify, render_template_string, request
from model import predict_tenant, train_model
import pandas as pd
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Updated HTML template with modern styling and file upload
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Tenant Predictor API</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; }
        .container { max-width: 800px; margin-top: 2rem; }
        .card { margin-bottom: 2rem; box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075); }
        .result { display: none; }
        .prediction-pass { color: #198754; font-weight: bold; }
        .prediction-fail { color: #dc3545; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        {% if error %}
            <div class="alert alert-danger">
                <h4>{{ error }}</h4>
                <p>{{ message }}</p>
            </div>
        {% else %}
            <h1 class="mb-4">Tenant Predictor API</h1>
            
            <!-- Training Section -->
            <div class="card mb-4">
                <div class="card-header">
                    <h5 class="mb-0">Train Model</h5>
                </div>
                <div class="card-body">
                    <form action="/train" method="post" enctype="multipart/form-data">
                        <div class="mb-3">
                            <label for="file" class="form-label">Upload Training CSV File (optional)</label>
                            <input type="file" class="form-control" id="file" name="file" accept=".csv">
                            <div class="form-text">If no file is provided, default dataset will be used.</div>
                        </div>
                        <button type="submit" class="btn btn-primary">Train Model</button>
                    </form>
                </div>
            </div>

            <!-- Prediction Form -->
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0">Test Prediction</h5>
                </div>
                <div class="card-body">
                    <ul class="nav nav-tabs mb-3" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="form-tab" data-bs-toggle="tab" data-bs-target="#form-panel" type="button" role="tab">Form Input</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="csv-tab" data-bs-toggle="tab" data-bs-target="#csv-panel" type="button" role="tab">CSV Upload</button>
                        </li>
                    </ul>
                    
                    <div class="tab-content">
                        <!-- Form Input Panel -->
                        <div class="tab-pane fade show active" id="form-panel" role="tabpanel">
                            <form id="predictionForm">
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="MonthlyIncome" class="form-label">Monthly Income ($)</label>
                                        <input type="number" class="form-control" id="MonthlyIncome" name="MonthlyIncome" value="5000">
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="FICOScore" class="form-label">FICO Score</label>
                                        <input type="number" class="form-control" id="FICOScore" name="FICOScore" value="700">
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="RentToIncomeRatio" class="form-label">Rent to Income Ratio (%)</label>
                                        <input type="number" step="0.1" class="form-control" id="RentToIncomeRatio" name="RentToIncomeRatio" value="30">
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="AssetMonthlyValue" class="form-label">Asset Monthly Value ($)</label>
                                        <input type="number" class="form-control" id="AssetMonthlyValue" name="AssetMonthlyValue" value="15000">
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="HasCriminalRecord" class="form-label">Has Criminal Record</label>
                                        <select class="form-select" id="HasCriminalRecord" name="HasCriminalRecord">
                                            <option value="0">No</option>
                                            <option value="1">Yes</option>
                                        </select>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="HasEvictionHistory" class="form-label">Has Eviction History</label>
                                        <select class="form-select" id="HasEvictionHistory" name="HasEvictionHistory">
                                            <option value="0">No</option>
                                            <option value="1">Yes</option>
                                        </select>
                                    </div>
                                </div>
                                <button type="submit" class="btn btn-primary">Predict</button>
                            </form>
                        </div>
                        
                        <!-- CSV Upload Panel -->
                        <div class="tab-pane fade" id="csv-panel" role="tabpanel">
                            <form id="csvPredictionForm">
                                <div class="mb-3">
                                    <label for="csvFile" class="form-label">Upload CSV File for Batch Prediction</label>
                                    <input type="file" class="form-control" id="csvFile" name="csvFile" accept=".csv">
                                    <div class="form-text">CSV should contain columns: MonthlyIncome, FICOScore, RentToIncomeRatio, HasCriminalRecord, HasEvictionHistory, AssetMonthlyValue</div>
                                </div>
                                <button type="submit" class="btn btn-primary">Predict from CSV</button>
                            </form>
                        </div>
                    </div>
                    
                    <div id="result" class="result mt-4">
                        <h5>Prediction Result:</h5>
                        <div class="alert" role="alert">
                            <div id="single-result">
                                Status: <span id="prediction"></span><br>
                                Confidence: <span id="confidence"></span>
                            </div>
                            <div id="batch-result" style="display: none;">
                                <div class="table-responsive">
                                    <table class="table table-striped">
                                        <thead>
                                            <tr>
                                                <th>Row</th>
                                                <th>Prediction</th>
                                                <th>Confidence</th>
                                            </tr>
                                        </thead>
                                        <tbody id="batch-results-body"></tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                // Existing form submission handler
                document.getElementById('predictionForm').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const formData = {
                        MonthlyIncome: Number(document.getElementById('MonthlyIncome').value),
                        FICOScore: Number(document.getElementById('FICOScore').value),
                        RentToIncomeRatio: Number(document.getElementById('RentToIncomeRatio').value) / 100,
                        HasCriminalRecord: Number(document.getElementById('HasCriminalRecord').value),
                        HasEvictionHistory: Number(document.getElementById('HasEvictionHistory').value),
                        AssetMonthlyValue: Number(document.getElementById('AssetMonthlyValue').value)
                    };
                    
                    try {
                        const response = await fetch('/test', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(formData)
                        });
                        const result = await response.json();
                        
                        if (result.error) {
                            throw new Error(result.error);
                        }
                        
                        document.getElementById('single-result').style.display = 'block';
                        document.getElementById('batch-result').style.display = 'none';
                        const resultDiv = document.getElementById('result');
                        const predictionSpan = document.getElementById('prediction');
                        const confidenceSpan = document.getElementById('confidence');
                        const alertDiv = resultDiv.querySelector('.alert');
                        
                        predictionSpan.textContent = result.prediction ? 'ACCEPTED' : 'DECLINED';
                        predictionSpan.className = result.prediction ? 'prediction-pass' : 'prediction-fail';
                        alertDiv.className = 'alert ' + (result.prediction ? 'alert-success' : 'alert-danger');
                        confidenceSpan.textContent = (result.confidence * 100).toFixed(2) + '%';
                        resultDiv.style.display = 'block';
                    } catch (error) {
                        alert('Error making prediction: ' + error.message);
                    }
                });

                // New CSV form submission handler
                document.getElementById('csvPredictionForm').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const formData = new FormData();
                    const fileInput = document.getElementById('csvFile');
                    formData.append('file', fileInput.files[0]);
                    
                    try {
                        const response = await fetch('/test-batch', {
                            method: 'POST',
                            body: formData
                        });
                        const results = await response.json();
                        
                        if (results.error) {
                            throw new Error(results.error);
                        }
                        
                        document.getElementById('single-result').style.display = 'none';
                        document.getElementById('batch-result').style.display = 'block';
                        const resultDiv = document.getElementById('result');
                        const tbody = document.getElementById('batch-results-body');
                        
                        // Clear previous results
                        tbody.innerHTML = '';
                        
                        // Add new results
                        results.predictions.forEach((result, index) => {
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td>${index + 1}</td>
                                <td class="${result.prediction ? 'prediction-pass' : 'prediction-fail'}">
                                    ${result.prediction ? 'ACCEPTED' : 'DECLINED'}
                                </td>
                                <td>${(result.confidence * 100).toFixed(2)}%</td>
                            `;
                            tbody.appendChild(row);
                        });
                        
                        resultDiv.style.display = 'block';
                    } catch (error) {
                        alert('Error processing CSV: ' + error.message);
                    }
                });
            </script>
        {% endif %}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""



@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)





@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'GET':
        return render_template_string(HTML_TEMPLATE)
        
    try:
        data = request.get_json()
        result = predict_tenant([
            data['MonthlyIncome'],
            data['FICOScore'],
            data['RentToIncomeRatio'],
            data['HasCriminalRecord'],
            data['HasEvictionHistory'],
            data['AssetMonthlyValue']
        ])
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500






@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required_fields = [
            'MonthlyIncome',
            'FICOScore',
            'RentToIncomeRatio',
            'HasCriminalRecord',
            'HasEvictionHistory',
            'AssetMonthlyValue'
        ]
        
        # Validate input
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Prepare input data
        tenant_data = [
            data['MonthlyIncome'],
            data['FICOScore'],
            data['RentToIncomeRatio'],
            data['HasCriminalRecord'],
            data['HasEvictionHistory'],
            data['AssetMonthlyValue']
        ]
        
        # Get prediction
        result = predict_tenant(tenant_data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500








@app.route('/train', methods=['GET', 'POST'])
def train():
    try:
        if request.method == 'POST' and 'file' in request.files:
            file = request.files['file']
            if file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                data_path = filepath
            else:
                data_path = 'data/Credit_Income_Check.csv'
        else:
            data_path = 'data/Credit_Income_Check.csv'
            
        # Parse data from the file
        from src.parse import parse_data
        training_data = parse_data(data_path)
        
        if not training_data:
            return render_template_string(
                HTML_TEMPLATE,
                error="Training Error",
                message="No data could be parsed from file."
            )
            
        # Train model with parsed data
        accuracy = train_model(training_data)
        success_message = f"Model trained successfully! Accuracy: {accuracy:.2%} using {len(training_data)} samples."
        print(success_message)
        
        return render_template_string(
            HTML_TEMPLATE,
            error=None,
            message=success_message
        )
        
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            error="Training Error",
            message=str(e)
        )







@app.errorhandler(404)
def page_not_found(e):
    return render_template_string(
        HTML_TEMPLATE,
        error="404 - Page Not Found",
        message="The requested URL was not found on the server."
    ), 404






if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
