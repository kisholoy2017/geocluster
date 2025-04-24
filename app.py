import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Business Clustering App", layout="wide")

st.title("🧪 Geo Clustering App")
st.markdown("Upload a dataset with `Revenue`, `Orders`, and `AOV` columns. This app performs KMeans clustering, evaluates metrics, and plots the clusters in 2D and 3D.")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head())

    try:
        # Step 1: Feature prep & scaling
        F = df[['Revenue', 'Orders', 'AOV']].values
        scaler = StandardScaler()
        F_scaled = scaler.fit_transform(F)

        # Step 2: Metrics for multiple k
        st.subheader("📈 Cluster Validation Across Different k-values")

        silhouette_scores = []
        dbi_scores = []
        k_values = list(range(2, 11))

        for k in k_values:
            kmeans_k = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
            labels_k = kmeans_k.fit_predict(F_scaled)
            silhouette_scores.append(silhouette_score(F_scaled, labels_k))
            dbi_scores.append(davies_bouldin_score(F_scaled, labels_k))

        # Plot metrics
        fig_metrics, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(k_values, silhouette_scores, marker='o')
        axes[0].axvline(x=4, color='red', linestyle='--', label='Chosen k=4')
        axes[0].set_title("Silhouette Score vs. k")
        axes[0].set_xlabel("Number of Clusters (k)")
        axes[0].set_ylabel("Silhouette Score")
        axes[0].legend()
        axes[0].grid()

        axes[1].plot(k_values, dbi_scores, marker='o', color='orange')
        axes[1].axvline(x=4, color='red', linestyle='--', label='Chosen k=4')
        axes[1].set_title("Davies-Bouldin Index vs. k")
        axes[1].set_xlabel("Number of Clusters (k)")
        axes[1].set_ylabel("Davies-Bouldin Index")
        axes[1].legend()
        axes[1].grid()

        st.pyplot(fig_metrics)

        # Step 3: KMeans Clustering with 4 clusters
        kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=42)
        df['Cluster'] = kmeans.fit_predict(F_scaled)

        # Step 4: State-level averages
        df_avg = df.groupby("state_name")[["Revenue", "AOV"]].mean().reset_index()
        df_avg = df_avg.merge(df[["state_name", "Cluster"]].drop_duplicates(), on="state_name", how="left")

        # Step 5: Final Metrics
        silhouette = silhouette_score(F_scaled, df['Cluster'])
        dbi = davies_bouldin_score(F_scaled, df['Cluster'])

        st.subheader("📊 Final Metrics (k = 4)")
        st.write(f"**Silhouette Score:** {silhouette:.4f}")
        st.write(f"**Davies-Bouldin Index:** {dbi:.4f}")

        # Step 6: 2D Cluster Plot (Updated Size)
        st.subheader("📉 2D Cluster Plot (Revenue vs AOV)")
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        colors = ['blue', 'green', 'purple', 'orange']
        markers = ['o', 's', 'D', '^']

        fig2d, ax2d = plt.subplots(figsize=(12, 5))  # Updated plot size
        for i in range(4):
            cluster_data = df_avg[df_avg["Cluster"] == i]
            ax2d.scatter(cluster_data["Revenue"], cluster_data["AOV"], s=100, color=colors[i], marker=markers[i], label=f'Cluster {i+1}', alpha=0.7)

        ax2d.scatter(centroids[:, 0], centroids[:, 2], s=200, c='red', marker='X', label='Centroids')
        ax2d.set_xlabel("Average Revenue ($)")
        ax2d.set_ylabel("Average AOV")
        ax2d.set_title("K-Means Clustering — 2D View")
        ax2d.legend()
        ax2d.grid()
        st.pyplot(fig2d)

        # Step 7: 3D Cluster Plot (Updated Size)
        st.subheader("📦 3D Cluster Plot (Revenue, Orders, AOV)")
        fig3d = plt.figure(figsize=(12, 5))  # Updated plot size
        ax3d = fig3d.add_subplot(111, projection='3d')

        for i in range(4):
            cluster_data = df[df['Cluster'] == i]
            ax3d.scatter(cluster_data['Revenue'], cluster_data['Orders'], cluster_data['AOV'],
                         c=colors[i], label=f'Cluster {i+1}', s=80, alpha=0.6)

        ax3d.set_xlabel('Revenue')
        ax3d.set_ylabel('Orders')
        ax3d.set_zlabel('AOV')
        ax3d.zaxis.labelpad = 15
        ax3d.set_title("3D Cluster View (Revenue, Orders, AOV)")
        ax3d.legend()
        st.pyplot(fig3d)

        # Step 8: Cluster Table
        st.subheader("📋 Clustered States")
        clustered_states = df_avg.groupby("Cluster")["state_name"].apply(list).reset_index()
        clustered_states_dict = {
            f"Cluster {i+1}": clustered_states[clustered_states["Cluster"] == i]["state_name"].values.tolist()[0]
            for i in range(4)
        }

        clustered_states_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in clustered_states_dict.items()]))
        clustered_states_df = clustered_states_df.fillna("")
        st.dataframe(clustered_states_df)

        # Download button
        csv = clustered_states_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cluster Table as CSV", data=csv, file_name='clustered_states.csv', mime='text/csv')

    except Exception as e:
        st.error(f"Something went wrong: {e}")
