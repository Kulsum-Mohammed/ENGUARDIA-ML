import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# Custom theme
def set_custom_theme():
    st.markdown("""
    <style>
    :root {
        --primary: #8956F1;
        --secondary: #CB80FF;
        --dark: #241B35;
        --darker: #020102;
    }
    .stApp { background-color: var(--darker); color: white; }
    .sidebar .sidebar-content { background-color: var(--dark) !important; }
    .stButton>button { border: 1px solid var(--primary); background-color: var(--dark) !important; color: white !important; }
    .stButton>button:hover { border: 1px solid var(--secondary); background-color: var(--primary) !important; }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('intrusion_detection_model.pkl')
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Prediction function
def predict_attack(input_data, pkg):
    try:
        processed = pd.DataFrame(0, index=[0], columns=pkg['feature_names'])
        
        for col in pkg['num_cols']:
            processed[col] = input_data.get(col, pkg['X_train_median'].get(col, 0))

        for col in pkg['cat_cols']:
            val = input_data.get(col)
            if val not in pkg['valid_categories'][col]:
                raise ValueError(f"Invalid {col}='{val}'. Must be one of: {pkg['valid_categories'][col]}")
            encoded_col = f"{col}_{val}"
            if encoded_col in processed.columns:
                processed[encoded_col] = 1

        processed[pkg['num_cols']] = pkg['scaler'].transform(processed[pkg['num_cols']])
        pred = pkg['model'].predict(processed)[0]
        proba = pkg['model'].predict_proba(processed)[0]

        return {
            'attack_type': pkg['attack_mapping'][pred],
            'confidence': float(np.max(proba)),
            'probabilities': {pkg['attack_mapping'][i]: float(p) for i, p in enumerate(proba)},
            'status': 'success'
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# Main app
def main():
    set_custom_theme()
    st.set_page_config(page_title="EnGuardia NIDS", layout="wide")
    
    pkg = load_model()
    if pkg is None:
        return

    # Initialize session state
    if 'attack_history' not in st.session_state:
        st.session_state.attack_history = pd.DataFrame(columns=[
            'timestamp', 'attack_type', 'confidence', 'src_bytes', 'dst_bytes'
        ])

    # Sidebar
    st.sidebar.title("EnGuardia NIDS")
    tab = st.sidebar.radio("Menu", ["Real-Time Monitor", "Manual Inspection"])

    # Main content
    if tab == "Real-Time Monitor":
        st.header("🔄 Real-Time Network Traffic Monitor")
        
        # Simulate live data
        live_data = pd.DataFrame({
            'timestamp': [datetime.now().strftime("%H:%M:%S") for _ in range(20)],
            'source_ip': [f"192.168.1.{np.random.randint(1,50)}" for _ in range(20)],
            'bytes': np.random.randint(100, 5000, 20),
            'threat': np.random.choice(['Normal', 'DoS', 'Probe', 'R2L'], 20, p=[0.85, 0.05, 0.07, 0.03])
        })
        
        # Show alerts
        with st.expander("🚨 Active Alerts", expanded=True):
            alerts = live_data[live_data['threat'] != 'Normal'].head(3)
            if not alerts.empty:
                for _, alert in alerts.iterrows():
                    st.warning(f"{alert['threat']} detected from {alert['source_ip']} at {alert['timestamp']}")
            else:
                st.success("No active threats detected")

        # Show attack history
        st.subheader("Attack History")
        if not st.session_state.attack_history.empty:
            st.dataframe(st.session_state.attack_history)
        else:
            st.info("No attacks logged yet")

    elif tab == "Manual Inspection":
        st.header("🔍 Manual Packet Inspection")
        
        with st.form("packet_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                protocol = st.selectbox("Protocol", pkg['valid_categories']['protocol_type'])
                service = st.selectbox("Service", pkg['valid_categories']['service'])
                flag = st.selectbox("Flag", pkg['valid_categories']['flag'])
                src_bytes = st.number_input("Source Bytes", value=100)
                dst_bytes = st.number_input("Destination Bytes", value=50)
            
            with col2:
                logged_in = st.selectbox("Logged In", [0, 1])
                count = st.number_input("Count", value=10)
                same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 0.8)
            
            if st.form_submit_button("Analyze"):
                input_data = {
                    'protocol_type': protocol,
                    'service': service,
                    'flag': flag,
                    'src_bytes': src_bytes,
                    'dst_bytes': dst_bytes,
                    'logged_in': logged_in,
                    'count': count,
                    'same_srv_rate': same_srv_rate,
                    # Default values for other required features
                    'duration': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
                    'hot': 0, 'num_failed_logins': 0, 'num_compromised': 0,
                    'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
                    'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
                    'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
                    'serror_rate': 0.0, 'srv_serror_rate': 0.0, 'rerror_rate': 0.0,
                    'srv_rerror_rate': 0.0, 'srv_diff_host_rate': 0.1,
                    'dst_host_count': 100, 'dst_host_srv_count': 50,
                    'dst_host_same_srv_rate': 0.9, 'dst_host_diff_srv_rate': 0.1,
                    'dst_host_same_src_port_rate': 0.8, 'dst_host_srv_diff_host_rate': 0.2,
                    'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0,
                    'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0,
                    'last_flag': 0
                }
                
                with st.spinner("Analyzing..."):
                    result = predict_attack(input_data, pkg)
                
                if result['status'] == 'success':
                    st.success(f"Predicted: {result['attack_type'].upper()} (Confidence: {result['confidence']*100:.1f}%)")
                    
                    # Show probabilities
                    prob_df = pd.DataFrame({
                        'Attack': list(result['probabilities'].keys()),
                        'Probability': [p*100 for p in result['probabilities'].values()]
                    })
                    fig = px.bar(prob_df, x='Attack', y='Probability')
                    st.plotly_chart(fig)
                    
                    # Log attack
                    new_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'attack_type': result['attack_type'],
                        'confidence': result['confidence'],
                        'src_bytes': src_bytes,
                        'dst_bytes': dst_bytes
                    }
                    st.session_state.attack_history = pd.concat([
                        st.session_state.attack_history,
                        pd.DataFrame([new_entry])
                    ], ignore_index=True)
                else:
                    st.error(f"Error: {result['message']}")

if __name__ == "__main__":
    main()
