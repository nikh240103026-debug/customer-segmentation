from flask import Flask, request, render_template_string
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open('customer_segmentation_model.pkl', 'rb') as f:
    model = pickle.load(f)

cluster_names = {
    0: "💰 Budget Customer — Low income, Low spending",
    1: "👑 Premium Customer — High income, High spending",
    2: "🎯 Impulsive Buyer — Low income, High spending",
    3: "🧠 Careful Spender — High income, Low spending",
    4: "⚖️ Middle Customer — Average income, Average spending"
}

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Customer Segmentation</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; background-color: #f4f4f4; }
        h1 { text-align: center; color: #333; }
        form { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        label { font-weight: bold; display: block; margin-top: 15px; }
        input { width: 100%; padding: 8px; margin-top: 5px; border: 1px solid #ddd; border-radius: 5px; }
        button { width: 100%; padding: 12px; margin-top: 20px; background-color: #333; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }
        .result { text-align: center; font-size: 20px; margin-top: 20px; padding: 15px; background: white; border-radius: 10px; font-weight: bold; color: #333; }
    </style>
</head>
<body>
    <h1>🛍️ Customer Segmentation</h1>
    {% if prediction %}
    <div class="result">{{ prediction }}</div>
    {% endif %}
    <form action="/predict" method="post">
        <label>Annual Income (k$):</label>
        <input type="number" name="income" placeholder="e.g. 60" required>
        <label>Spending Score (1-100):</label>
        <input type="number" name="spending" placeholder="e.g. 50" required>
        <button type="submit">Find My Segment</button>
    </form>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    
    features = np.array([[
        float(data['income']),
        float(data['spending'])
    ]])
    
    cluster = model.predict(features)[0]
    result = cluster_names.get(cluster, "Unknown Segment")
    
    return render_template_string(HTML, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)