import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Data loading and preprocessing functions
def load_data():
    columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
               "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
               "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
               "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
               "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
               "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
               "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
               "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
               "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
               "attack", "last_flag"]
    
    train = pd.read_csv('data/Train.txt', sep=',', names=columns)
    test = pd.read_csv('data/Test.txt', sep=',', names=columns)
    return train, test

def preprocess_data(train, test):
    # Convert categorical columns
    cat_cols = ['protocol_type', 'service', 'flag', 'attack']
    for col in cat_cols:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)

    # Map attack classes
    def map_attack_class(df):
        df.loc[df.attack == 'normal', 'attack_class'] = 0
        dos_attacks = ['back', 'land', 'pod', 'neptune', 'smurf', 'teardrop',
                      'apache2', 'udpstorm', 'processtable', 'worm', 'mailbomb']
        df.loc[df.attack.isin(dos_attacks), 'attack_class'] = 1
        probe_attacks = ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']
        df.loc[df.attack.isin(probe_attacks), 'attack_class'] = 2
        r2l_attacks = ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop',
                      'warezmaster', 'warezclient', 'spy', 'xlock', 'xsnoop',
                      'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named']
        df.loc[df.attack.isin(r2l_attacks), 'attack_class'] = 3
        u2r_attacks = ['buffer_overflow', 'loadmodule', 'rootkit', 'perl',
                      'sqlattack', 'xterm', 'ps']
        df.loc[df.attack.isin(u2r_attacks), 'attack_class'] = 4
        return df

    train = map_attack_class(train)
    test = map_attack_class(test)
    
    return train, test

def feature_engineering(train_df, test_df):
    cat_cols = ['protocol_type', 'service', 'flag']
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(train_df[cat_cols])

    def encode_data(df):
        encoded = ohe.transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(), index=df.index)
        return pd.concat([df.drop(cat_cols + ['attack'], axis=1), encoded_df], axis=1)

    train_processed = encode_data(train_df)
    test_processed = encode_data(test_df)

    X_train = train_processed.drop('attack_class', axis=1)
    y_train = train_processed['attack_class']
    X_test = test_processed.drop('attack_class', axis=1)
    y_test = test_processed['attack_class']

    num_cols = X_train.select_dtypes(include=['int64','float64']).columns
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, y_train, X_test, y_test, ohe, scaler

def train_and_save_model():
    # Load and preprocess data
    train, test = load_data()
    train, test = preprocess_data(train, test)
    X_train, y_train, X_test, y_test, ohe, scaler = feature_engineering(train, test)
    
    # Train model
    model = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.2, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model package
    deployment_pkg = {
        'model': model,
        'scaler': scaler,
        'encoder': ohe,
        'feature_names': X_train.columns.tolist(),
        'num_cols': [col for col in X_train.columns if X_train[col].dtype in ['int64','float64']],
        'cat_cols': ['protocol_type', 'service', 'flag'],
        'valid_categories': {
            'flag': ohe.categories_[2].tolist(),
            'service': ohe.categories_[1].tolist(),
            'protocol_type': ohe.categories_[0].tolist()
        },
        'X_train_median': X_train.select_dtypes(include=['int64','float64']).median().to_dict(),
        'attack_mapping': {
            0: 'normal',
            1: 'dos',
            2: 'probe',
            3: 'r2l',
            4: 'u2r'
        }
    }
    
    joblib.dump(deployment_pkg, 'intrusion_detection_model.pkl')
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
