import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    df = pd.read_csv('weatherAUS.csv')
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
    return df

# Fungsi untuk preprocessing data
def preprocess_data(df):
    # Drop kolom yang tidak diperlukan
    df = df.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis=1, errors='ignore')

    
    # Encoding variabel kategorikal
    cat_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Imputasi nilai yang hilang
    num_cols = df.select_dtypes(include=['float64']).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    cat_cols = df.select_dtypes(include=['int64']).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

# Fungsi untuk melatih model
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Fungsi untuk menampilkan metrik evaluasi
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

# Main function
def main():
    st.title("Prediksi Hujan di Australia")
    st.markdown("""
    Dashboard ini menggunakan model Logistic Regression untuk memprediksi apakah akan hujan besok di Australia berdasarkan data cuaca hari ini.
    Dataset berasal dari [Kaggle - Rain in Australia](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package).
    """)
    
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.header("Navigasi")
    page = st.sidebar.radio("Pilih halaman:", ["Data Overview", "EDA", "Model Training", "Prediksi"])
    
    if page == "Data Overview":
        st.header("Overview Dataset")
        
        st.subheader("5 Baris Pertama Data")
        st.write(df.head())
        
        st.subheader("Statistik Deskriptif")
        st.write(df.describe())
        
        st.subheader("Informasi Dataset")
        st.text("""
        Jumlah baris: {}
        Jumlah kolom: {}
        """.format(df.shape[0], df.shape[1]))
        
        st.subheader("Missing Values")
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        missing_data = pd.DataFrame({'Missing Values': missing_values,
                                   'Percentage': missing_percentage})
        st.write(missing_data[missing_data['Missing Values'] > 0].sort_values('Missing Values', ascending=False))
        
    elif page == "EDA":
        st.header("Exploratory Data Analysis")
        
        st.subheader("Distribusi Target Variable (RainTomorrow)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='RainTomorrow', data=df, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Distribusi Variabel Numerik")
        num_cols = df.select_dtypes(include=['float64']).columns
        selected_num_col = st.selectbox("Pilih variabel numerik:", num_cols)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[selected_num_col], kde=True, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Korelasi antar Variabel")
        numeric_df = df.select_dtypes(include=[float, int])  # Select only numeric columns
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
    elif page == "Model Training":
        st.header("Pelatihan Model")
        
        # Preprocess data
        df_processed = preprocess_data(df.copy())
        
        # Split data
        X = df_processed.drop('RainTomorrow', axis=1)
        y = df_processed['RainTomorrow']
        
        # Fitur seleksi
        selected_features = st.multiselect(
            "Pilih fitur untuk model:",
            X.columns,
            default=list(X.columns)
        )
        
        X = X[selected_features]
        
        # Split data
        test_size = st.slider("Ukuran data test:", 0.1, 0.5, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if st.button("Train Model"):
            model = train_model(X_train_scaled, y_train)
            st.success("Model berhasil dilatih!")
            
            st.subheader("Evaluasi Model")
            evaluate_model(model, X_test_scaled, y_test)
            
            # Simpan model untuk prediksi
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['selected_features'] = selected_features
    
    elif page == "Prediksi":
        st.header("Prediksi Hujan Besok")
        
        if 'model' not in st.session_state:
            st.warning("Silakan latih model terlebih dahulu di halaman Model Training.")
            return
            
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        selected_features = st.session_state['selected_features']
        
        st.subheader("Input Data untuk Prediksi")
        
        # Buat input form
        input_data = {}
        cols = st.columns(2)
        
        for i, feature in enumerate(selected_features):
            with cols[i % 2]:
                if df[feature].dtype == 'float64':
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    default_val = float(df[feature].median())
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=(max_val-min_val)/100
                    )
                else:
                    unique_vals = df[feature].unique()
                    default_idx = 0
                    input_data[feature] = st.selectbox(
                        f"{feature}",
                        unique_vals,
                        index=default_idx
                    )
        
        if st.button("Prediksi"):
            # Buat DataFrame dari input
            input_df = pd.DataFrame([input_data])
            
            # Preprocess input
            input_df = preprocess_data(input_df)
            input_df = input_df[selected_features]
            input_scaled = scaler.transform(input_df)
            
            # Prediksi
            prediction = model.predict(input_scaled)
            proba = model.predict_proba(input_scaled)
            
            st.subheader("Hasil Prediksi")
            if prediction[0] == 1:
                st.error(f"Prediksi: Akan Hujan Besok (Probabilitas: {proba[0][1]*100:.2f}%)")
            else:
                st.success(f"Prediksi: Tidak Akan Hujan Besok (Probabilitas: {proba[0][0]*100:.2f}%)")

if __name__ == "__main__":
    main()
