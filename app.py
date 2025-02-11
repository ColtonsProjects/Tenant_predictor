from flask import Flask, jsonify, render_template_string, request
from model import predict_tenant, train_model

app = Flask(__name__)

# HTML template with basic styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Tenant Predictor API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .error { color: red; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        input { padding: 5px; width: 200px; }
        button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        .result { margin-top: 20px; padding: 20px; background-color: #f5f5f5; }
        .pass { color: green; }
        .fail { color: red; }
    </style>
</head>
<body>
    <div class="container">
        {% if error %}
            <h1 class="error">{{ error }}</h1>
            <p>{{ message }}</p>
        {% else %}
            <h1>Tenant Predictor API Server</h1>
            <p>The server is running on port 5000.</p>
            
            <h2>Test Prediction</h2>
            <form id="predictionForm">
                <div class="form-group">
                    <label for="MonthlyIncome">Monthly Income ($):</label>
                    <input type="number" id="MonthlyIncome" name="MonthlyIncome" value="5000">
                </div>
                <div class="form-group">
                    <label for="FICOScore">FICO Score:</label>
                    <input type="number" id="FICOScore" name="FICOScore" value="700">
                </div>
                <div class="form-group">
                    <label for="RentToIncomeRatio">Rent to Income Ratio (%):</label>
                    <input type="number" step="0.1" id="RentToIncomeRatio" name="RentToIncomeRatio" value="30">
                </div>
                <div class="form-group">
                    <label for="HasCriminalRecord">Has Criminal Record:</label>
                    <select id="HasCriminalRecord" name="HasCriminalRecord">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="HasEvictionHistory">Has Eviction History:</label>
                    <select id="HasEvictionHistory" name="HasEvictionHistory">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="AssetMonthlyValue">Asset Monthly Value ($):</label>
                    <input type="number" id="AssetMonthlyValue" name="AssetMonthlyValue" value="15000">
                </div>
                <button type="submit">Predict</button>
            </form>
            
            <div id="result" class="result" style="display: none;">
                <h3>Prediction Result:</h3>
                <p>Status: <span id="prediction"></span></p>
                <p>Confidence: <span id="confidence"></span></p>
            </div>
            
            <script>
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
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify(formData)
                        });
                        const result = await response.json();
                        
                        const resultDiv = document.getElementById('result');
                        const predictionSpan = document.getElementById('prediction');
                        const confidenceSpan = document.getElementById('confidence');
                        
                        predictionSpan.textContent = result.prediction ? 'ACCEPTED' : 'DECLINED';
                        predictionSpan.className = result.prediction ? 'pass' : 'fail';
                        confidenceSpan.textContent = (result.confidence * 100).toFixed(2) + '%';
                        resultDiv.style.display = 'block';
                    } catch (error) {
                        alert('Error making prediction: ' + error.message);
                    }
                });
            </script>
        {% endif %}
    </div>
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








@app.route('/train', methods=['GET'])
def train():
    try:
        # Use absolute path to data file
        data_path = 'data/Credit_Income_Check.csv'
            
        # Parse first 20 rows from the data
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
        print(f"Model trained successfully! Accuracy: {accuracy:.2%} using {len(training_data)} samples.")
        
        return render_template_string(
            HTML_TEMPLATE,
            error=None,
            message=f"Model trained successfully! Accuracy: {accuracy:.2%} using {len(training_data)} samples."
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
