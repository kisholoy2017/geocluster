**🧪 Geo Clustering App**
A Streamlit-powered tool for clustering and visualizing business performance across U.S. states using key financial metrics.

📦 What It Does
This app allows users to upload a dataset containing state-level business data and performs KMeans clustering based on:

Revenue

Orders

AOV (Average Order Value)

It visualizes the results with interactive charts and provides a downloadable summary of clustered states.

🔍 How It Works
📁 Upload Data
Upload a CSV file that includes state_name, Revenue, Orders, and AOV.

📏 Standardization
Standardizes numeric features to improve clustering quality.

🔢 Cluster Evaluation
Automatically tests cluster sizes from k=2 to k=10 using:

Silhouette Score (higher is better)

Davies-Bouldin Index (lower is better)

🎯 Final Clustering (k=4)
Uses k=4 as the final number of clusters and applies KMeans to segment the data.

📊 Metrics & Visualizations

Final scores are displayed for performance evaluation.

2D Plot: Revenue vs. AOV (by cluster)

3D Plot: Revenue, Orders, AOV (by cluster)

📋 Cluster Table
Displays a clean table showing which states belong to each cluster.
✅ Includes download button for exporting results.

🛠️ Installation
Clone the repo:

bash
Copy
Edit
git clone https://github.com/your-username/geo-clustering-app.git
cd geo-clustering-app
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
streamlit run app.py
🚀 Deploy on Streamlit Cloud
Just push this repo to GitHub and connect it to Streamlit Community Cloud. Done!

📝 Sample Input Format
Your CSV should include the following columns:

python-repl
Copy
Edit
state_name, Revenue, Orders, AOV
Texas, 100000, 200, 500
California, 150000, 300, 500
...
