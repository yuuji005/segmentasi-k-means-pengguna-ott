import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from flask import Flask, render_template, request, jsonify
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Path Dataset
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ott_subscriber_master_raw.csv")
PLOT_DIR = os.path.join(BASE_DIR, "static", "plots")

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Global model objects for prediction simulator
model_store = {
    'kmeans': None,
    'scaler': None,
    'features': ["AGE", "TOTAL_TIME_SPENT", "NUMBER_OF_CONTENT_VIEWED"]
}

def train_model():
    df = pd.read_csv(DATA_PATH)
    df.rename(columns={
        'age': 'AGE',
        'gender': 'GENDER',
        'avg_session_duration_mins': 'TOTAL_TIME_SPENT',
        'titles_watched_per_month': 'NUMBER_OF_CONTENT_VIEWED'
    }, inplace=True)
    
    # Preprocessing
    X = df[model_store['features']].fillna(df[model_store['features']].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Store for global access
    model_store['kmeans'] = kmeans
    model_store['scaler'] = scaler
    
    return df, X_scaled

@app.route('/')
def index():
    df, X_scaled = train_model()
    kmeans = model_store['kmeans']
    scaler = model_store['scaler']
    
    # 3. Mencari K Optimal (Metode Elbow) - Dipersingkat untuk performa
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
    
    # Generate Elbow Plot in memory
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertia, marker='o', linestyle='--', color='#38bdf8')
    plt.title('Metode Elbow untuk Menentukan K Optimal', color='white', pad=20)
    plt.xlabel('Jumlah Klaster (k)', color='white')
    plt.ylabel('Inertia / WCSS', color='white')
    plt.gca().set_facecolor('none')
    plt.gcf().set_facecolor('none')
    plt.tick_params(colors='white')
    
    elbow_buffer = io.BytesIO()
    plt.savefig(elbow_buffer, format='png', transparent=True)
    elbow_buffer.seek(0)
    elbow_base64 = base64.b64encode(elbow_buffer.read()).decode('utf-8')
    plt.close()

    # 5. Visualisasi Hasil Clustering
    threshold = df['TOTAL_TIME_SPENT'].quantile(0.99)
    df_plot = df[df['TOTAL_TIME_SPENT'] <= threshold].copy()
    df_sample = df_plot.sample(n=min(250, len(df_plot)), random_state=42)

    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background') 
    
    sns.scatterplot(data=df_sample, x='TOTAL_TIME_SPENT', y='NUMBER_OF_CONTENT_VIEWED', 
                    hue='Cluster', palette='viridis', s=50, alpha=0.8, edgecolor='grey')
    
    plt.title('Hasil Clustering Pelanggan', fontsize=14, pad=15)
    plt.xlabel('Durasi Menonton (Menit)', fontsize=11)
    plt.ylabel('Jumlah Judul Ditonton', fontsize=11)
    plt.legend(title='Cluster', loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().set_facecolor('none')
    plt.gcf().set_facecolor('none')
    plt.tight_layout()
    
    cluster_buffer = io.BytesIO()
    plt.savefig(cluster_buffer, format='png', transparent=True)
    cluster_buffer.seek(0)
    cluster_base64 = base64.b64encode(cluster_buffer.read()).decode('utf-8')
    plt.close()

    # --- CLUSTER PROFILING ---
    profiles = []
    cluster_means = df.groupby('Cluster')[model_store['features']].mean().sort_values('TOTAL_TIME_SPENT')
    names = ["The Lite User", "The Balanced Viewer", "The Binge Watcher"]
    
    for i, (idx, row) in enumerate(cluster_means.iterrows()):
        profiles.append({
            'id': int(idx),
            'name': names[i],
            'avg_age': round(row['AGE'], 1),
            'avg_time': round(row['TOTAL_TIME_SPENT'], 1),
            'avg_content': round(row['NUMBER_OF_CONTENT_VIEWED'], 1),
            'count': int(df['Cluster'].value_counts()[idx])
        })

    table_data = df[['GENDER', 'AGE', 'TOTAL_TIME_SPENT', 'NUMBER_OF_CONTENT_VIEWED', 'Cluster']].head(15)

    stats = {
        'total_subscribers': len(df),
        'avg_age': round(df['AGE'].mean(), 1),
        'avg_time': round(df['TOTAL_TIME_SPENT'].mean(), 1),
        'avg_content': round(df['NUMBER_OF_CONTENT_VIEWED'].mean(), 1)
    }
    
    return render_template('index.html', 
                           tables=table_data.to_dict(orient='records'),
                           stats=stats,
                           profiles=profiles,
                           elbow_url=f"data:image/png;base64,{elbow_base64}",
                           cluster_url=f"data:image/png;base64,{cluster_base64}")

@app.route('/predict', methods=['POST'])
def predict():
    if model_store['kmeans'] is None:
        train_model() # Ensure model is loaded
        
    data = request.json
    try:
        user_input = np.array([[
            float(data['age']),
            float(data['time']),
            float(data['content'])
        ]])
        
        # Scale & Predict
        scaled_input = model_store['scaler'].transform(user_input)
        cluster_id = int(model_store['kmeans'].predict(scaled_input)[0])
        
        # Determine Name based on current model state
        # We re-calculate the means to get the sorted mapping
        # (Alternatively, we could have stored 'names_map' in model_store)
        df, _ = train_model() # Get current cluster states
        cluster_means = df.groupby('Cluster')['TOTAL_TIME_SPENT'].mean().sort_values()
        names = ["The Lite User", "The Balanced Viewer", "The Binge Watcher"]
        names_map = {idx: names[i] for i, idx in enumerate(cluster_means.index)}
        
        return jsonify({
            'success': True,
            'cluster': cluster_id,
            'name': names_map.get(cluster_id, "Unknown Segment")
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)