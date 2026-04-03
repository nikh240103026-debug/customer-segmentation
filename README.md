# Customer Segmentation

A Machine Learning app that segments mall customers into 5 distinct groups based on income and spending behavior.

## Customer Segments
- Premium Customers — High income, High spending
- Careful Spenders — High income, Low spending
- Impulsive Buyers — Low income, High spending
- Budget Customers — Low income, Low spending
- Middle Customers — Average income, Average spending

## Technologies Used
- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn (K-Means Clustering)
- Flask

## Model Evaluation
- Algorithm: K-Means Clustering (Unsupervised Learning)
- Optimal Clusters: 5 (Elbow Method)
- Silhouette Score: 0.5539

## How to Run
1. Install requirements: pip install flask scikit-learn pandas numpy
2. Run: python app.py
3. Open: http://127.0.0.1:5000
