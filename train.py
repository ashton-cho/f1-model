import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import f1_score
import pickle
from tqdm import tqdm
import warnings
import argparse
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# two numeric feature sets: base + optional telemetry
BASE_NUMERIC_COLS = [
    'LapNumber', 'SectorTime', 'TyreLife',
    'AirTemp', 'Humidity', 'Pressure', 'TrackTemp',
    'WindDirection', 'WindSpeed'
]
TELEMETRY_NUMERIC_COLS = [
    'LapNumber', 'SectorTime', 'TyreLife', 'Speed_P10',
    'Throttle_Median', 'Throttle_ZeroPct', 
    'Gear_Range', 'DRS_ActivePct', 'TrackStatus_Mean',
    'AirTemp', 'Humidity', 'Pressure', 'TrackTemp',
    'WindDirection', 'WindSpeed'
]

WINDOW_COLS = ['Stint', 'LapNumber', 'SectorNumber']

BASE_CATEGORICAL_COLS = ['EventName', 'Team', 'Compound', 'Stint']
TELEMETRY_CATEGORICAL_COLS = BASE_CATEGORICAL_COLS + ['TrackStatus_Mode']

EMBEDDING_DIMS_DICT = {
    'EventName': 8,
    'Team': 10,
    'Compound': 5,
    'Stint': 4,
    'TrackStatus_Mode': 4
}

PARAM_GRID = {
    "window_size":       [15],
    "num_lstm_layers":   [1],
    "lstm_units":        [64, 96, 128, 256],
    "dropout_rate":      [0.2, 0.3, 0.4, 0.5],
    "recurrent_dropout": [0.2, 0.3],
    "batch_size":        [256, 512],
    "learning_rate":     [1e-3, 1e-4],
    "epochs":            [20]
}


# --- Preprocessing functions ---

def load_and_clean_data(
    filepath,
    features_to_keep,
    numeric_cols,
    window_cols=WINDOW_COLS,
):
    """
    Load and clean raw data from a CSV file.

    Drops unwanted tyre compounds, converts time columns.
    Create backup columns for raw group/sort values (e.g., StintRaw)
    to avoid confusion after label encoding or scaling.

    Returns the cleaned DataFrame with new columns added.
    """
    df = pd.read_pickle(filepath)
    df = df[features_to_keep].copy()
    # df = df[~df['Compound'].isin(['INTERMEDIATE', 'WET', 'UNKNOWN'])]
    # if 'TrackStatus_Mode' in df.columns:
    #     df = df[df['TrackStatus_Mode'] < 4]  # remove sc/vsc rows
    df.loc[:, 'SectorTime'] = pd.to_timedelta(df['SectorTime']).dt.total_seconds()
    df.loc[:, numeric_cols] = df[numeric_cols].astype(float)
    for col in window_cols:
        if col in features_to_keep:
            df[f"{col}Raw"] = df[col]
    df = df.dropna()
    return df

def create_label(df, laps_col='LapsTilPit', n_laps=5):
    """
    Create a binary label column indicating if a pit stop will 
    occur within the next n_laps laps.

    Returns the modified DataFrame and the name of the label column.
    """
    label_name = f'PitNext{n_laps}Laps'
    df[label_name] = (df[laps_col] <= n_laps).astype(int)
    return df, label_name

def split_by_year(df, year_col='Year'):
    """
    Split the DataFrame into train, validation, and test sets based on year.
    Train: <=2023, Val: 2024, Test: 2025
    """
    train_df = df[df[year_col] <= 2023].copy()
    val_df = df[df[year_col] == 2024].copy()
    test_df = df[df[year_col] == 2025].copy()
    return train_df, val_df, test_df

def standardize_per_event(train_df, val_df, test_df, numeric_cols, event_col='EventName'):
    """
    Standardize numeric features per event for each split.
    Uses a global scaler as fallback for unseen events.

    Returns standardized splits and the dictionary of event scalers.
    """
    scaler_dict = {}
    global_scaler = StandardScaler().fit(train_df[numeric_cols])

    # Train: fit scaler per event
    for event in train_df[event_col].unique():
        mask = train_df[event_col] == event
        scaler = StandardScaler()
        train_df.loc[mask, numeric_cols] = scaler.fit_transform(train_df.loc[mask, numeric_cols])
        scaler_dict[event] = scaler

    # Val/Test: use event scaler if available, otherwise global
    for split_df in [val_df, test_df]:
        for event in split_df[event_col].unique():
            mask = split_df[event_col] == event
            if event in scaler_dict:
                split_df.loc[mask, numeric_cols] = scaler_dict[event].transform(split_df.loc[mask, numeric_cols])
            else:
                split_df.loc[mask, numeric_cols] = global_scaler.transform(split_df.loc[mask, numeric_cols])

    scaler_dict['__global__'] = global_scaler
    return train_df, val_df, test_df, scaler_dict

def label_encode_categoricals(train_df, val_df, test_df, categorical_cols):
    """
    Label-encode categorical features, handling unknown/unseen values with -1.

    Returns the splits with encoded columns, a dictionary of encoders,
    and a list of n_classes for each categorical variable.
    """
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))
        encoders[col] = le

        # Transform all splits
        train_df[col] = le.transform(train_df[col].astype(str))
        # For val and test, handle unseen labels
        for split_df in [val_df, test_df]:
            split_df[col] = split_df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Save mapping for use at inference
    n_classes = [train_df[col].nunique() for col in categorical_cols]
    return train_df, val_df, test_df, encoders, n_classes

def run_preprocessing_pipeline(
    filepath,
    features_to_keep,
    numeric_cols,
    categorical_cols,
    laps_label_window=5
):
    """
    Complete preprocessing pipeline for raw CSV input to train/val/test splits.

    Returns processed splits, label column name, scaler dictionary,
    n_classes list, and encoders dictionary.
    """
    df = load_and_clean_data(
        filepath, features_to_keep, numeric_cols=numeric_cols
    )
    df, label_col = create_label(df, n_laps=laps_label_window)
    train_df, val_df, test_df = split_by_year(df)
    train_df, val_df, test_df, scaler_dict = standardize_per_event(train_df, val_df, test_df, numeric_cols)
    train_df, val_df, test_df, encoders, n_classes = label_encode_categoricals(
        train_df, val_df, test_df, categorical_cols
    )
    return train_df, val_df, test_df, label_col, scaler_dict, n_classes, encoders


# --- Sequence creation ---

def make_sector_to_lap_sequences(df, feature_cols, label_col, window_size):
    """
    Create sliding windows of per-sector data for sequence modeling.
    Only keep windows where lap numbers are consecutive and all laps have all three sectors.
    Groups by driver/stint, sorts by lap/sector, and makes overlapping windows.

    Returns X (windows) and y (labels).
    """
    X_seqs, y_seqs = [], []
    group_cols = ['Year', 'EventName', 'Driver', 'StintRaw']
    for _, group in df.groupby(group_cols):
        group = group.sort_values(['LapNumberRaw', 'SectorNumberRaw'])
        feats = group[feature_cols].values
        lap_labels = group[label_col].values
        lap_nums = group['LapNumberRaw'].values
        sector_nums = group['SectorNumberRaw'].values
        lap_sector_pairs = list(zip(lap_nums, sector_nums))

        for i in range(window_size, len(group)):
            win_pairs = lap_sector_pairs[i-window_size:i]
            win_laps = [pair[0] for pair in win_pairs]
            win_sectors = [pair[1] for pair in win_pairs]

            # 1. Laps must be consecutive
            unique_laps = sorted(set(win_laps))
            if len(unique_laps) != (max(win_laps) - min(win_laps) + 1):
                continue  # skip if not consecutive
            if not all(b - a == 1 for a, b in zip(unique_laps[:-1], unique_laps[1:])):
                continue

            # 2. Each lap in window has all three sectors
            lap_to_sectors = {lap: [] for lap in unique_laps}
            for lap, sector in zip(win_laps, win_sectors):
                lap_to_sectors[lap].append(sector)
            if not all(sorted(lap_to_sectors[lap]) == [1, 2, 3] for lap in unique_laps):
                continue

            X_seqs.append(feats[i-window_size:i])
            y_seqs.append(lap_labels[i])  # label for the window's end

    return np.array(X_seqs), np.array(y_seqs)

def split_features(X, cat_idx, num_idx):
    """
    Split feature array X into categorical (as list) and numeric (as array) parts.

    Returns (X_cat_list, X_num).
    """
    X_cat = [X[..., idx].astype(np.int32) for idx in cat_idx]
    X_num = X[..., num_idx].astype(np.float32)
    return X_cat, X_num


# --- Model building and training ---

class F1EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, patience=3, verbose=1):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.patience = patience
        self.best_score = -np.inf
        self.wait = 0
        self.best_weights = None
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        # Predict probabilities, then binarize
        y_pred_prob = self.model.predict(self.X_val, verbose=0)
        y_pred_bin = (y_pred_prob > 0.5).astype(int).flatten()
        # Calculate binary F1
        f1 = f1_score(self.y_val, y_pred_bin)
        if self.verbose:
            print(f"\nEpoch {epoch+1}: val_f1 = {f1:.4f}")
        # Check for improvement
        if f1 > self.best_score:
            self.best_score = f1
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}, best val_f1: {self.best_score:.4f}")
                self.model.stop_training = True
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)

class MacroF1EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, patience=3, verbose=1):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.patience = patience
        self.best_score = -np.inf
        self.wait = 0
        self.best_weights = None
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        # Predict probabilities, then binarize
        y_pred_prob = self.model.predict(self.X_val, verbose=0)
        y_pred_bin = (y_pred_prob > 0.5).astype(int).flatten()
        # Calculate macro F1
        macro_f1 = f1_score(self.y_val, y_pred_bin, average='macro')
        if self.verbose:
            print(f"\nEpoch {epoch+1}: val_macro_f1 = {macro_f1:.4f}")
        # Check for improvement
        if macro_f1 > self.best_score:
            self.best_score = macro_f1
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}, best val_macro_f1: {self.best_score:.4f}")
                self.model.stop_training = True
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)

def build_model(params, embedding_dims_list, n_classes, window_size, num_numeric, categorical_cols):
    """
    Build an LSTM sequence model with per-feature embedding layers for categoricals.

    Returns a compiled Keras model.
    """
    # Inputs
    num_input  = tf.keras.layers.Input((window_size, num_numeric), name='num_input')
    cat_inputs = [tf.keras.layers.Input((window_size,), name=col) for col in categorical_cols]

    # Embeddings
    cat_embeds = []
    for inp, dim, emb_dim, col in zip(cat_inputs, n_classes, embedding_dims_list, categorical_cols):
        cat_embeds.append(
            tf.keras.layers.Embedding(input_dim=dim, output_dim=emb_dim, mask_zero=False, name=f"{col}_emb")(inp)
        )

    x = tf.keras.layers.Concatenate(axis=-1)([num_input] + cat_embeds)

    # LSTM stack
    for i in range(params['num_lstm_layers']):
        return_seq = (i < params['num_lstm_layers'] - 1)

        # Choose LSTM, GRU, SimpleRNN
        x = tf.keras.layers.LSTM(params['lstm_units'],
                                 return_sequences=return_seq,
                                 recurrent_dropout=params['recurrent_dropout'])(x)
        x = tf.keras.layers.Dropout(params['dropout_rate'])(x)

    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=[num_input] + cat_inputs, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(params['learning_rate']),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

def train_model(model, X_train_num, X_train_cat, y_train, X_val_num, X_val_cat, y_val, batch_size, epochs):
    """
    Train the LSTM model with early stopping on validation loss.
    Handles class imbalance using computed class weights.

    Returns the Keras training history object.
    """
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',      # Monitor validation AUC
        patience=3,
        verbose=1,
        mode='max',             # 'max' because higher AUC is better
        restore_best_weights=True
    )

    history = model.fit(
        [X_train_num] + X_train_cat, y_train,
        validation_data=([X_val_num] + X_val_cat, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[callback],
        verbose=0,
        class_weight=class_weight_dict
    )
    return history

def predict(model, X_num, X_cat):
    """
    Predict binary probabilities for a dataset.

    Returns prediction probabilities and binarized predictions.
    """
    y_pred_prob = model.predict([X_num] + X_cat, verbose=0)
    y_pred_bin = (y_pred_prob > 0.5).astype(int).flatten()
    return y_pred_prob, y_pred_bin

def run_experiment(
    train_df, val_df, test_df,
    feature_cols, label_col,
    categorical_cols, numeric_cols,
    embedding_dims_dict, n_classes, window_size, params
):
    """
    Run a full experiment: windowing, feature split, model build, train, and evaluation.

    Returns the trained model and all binarized predictions.
    """
    X_train, y_train = make_sector_to_lap_sequences(train_df, feature_cols, label_col, window_size)
    X_val, y_val = make_sector_to_lap_sequences(val_df, feature_cols, label_col, window_size)
    X_test, y_test = make_sector_to_lap_sequences(test_df, feature_cols, label_col, window_size)

    cat_idx = [feature_cols.index(col) for col in categorical_cols]
    num_idx = [feature_cols.index(col) for col in numeric_cols]

    X_train_cat, X_train_num = split_features(X_train, cat_idx, num_idx)
    X_val_cat, X_val_num = split_features(X_val, cat_idx, num_idx)
    X_test_cat, X_test_num = split_features(X_test, cat_idx, num_idx)

    embedding_dims_list = [embedding_dims_dict[col] for col in categorical_cols]
    num_numeric = X_train_num.shape[-1]
    model = build_model(params, embedding_dims_list, n_classes, window_size, num_numeric, categorical_cols)

    train_model(
        model,
        X_train_num, X_train_cat, y_train,
        X_val_num, X_val_cat, y_val,
        batch_size=params.get('batch_size'), epochs=params.get('epochs')
    )

    y_test_pred_prob, y_test_pred_bin = predict(model, X_test_num, X_test_cat)

    return model, y_test_pred_bin, y_test

def run_once(hp, train_df, val_df, test_df, label_col, categorical_cols, n_classes, numeric_cols, features_to_keep):
    # Suppress ALL output in this worker process to keep tqdm clean
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    ws = hp["window_size"]
    params = {
        "num_lstm_layers":  hp["num_lstm_layers"],
        "lstm_units":       hp["lstm_units"],
        "dropout_rate":     hp["dropout_rate"],
        "recurrent_dropout":hp["recurrent_dropout"],
        "learning_rate":    hp["learning_rate"],
        "batch_size":       hp["batch_size"],
        "epochs":           hp["epochs"],
    }
    # Use validation set for early stopping, report val macro F1 for tuning
    model, y_val_pred, y_val_true = run_experiment(
        train_df, val_df, test_df,
        features_to_keep, label_col,
        categorical_cols, numeric_cols,
        EMBEDDING_DIMS_DICT, n_classes,
        window_size=ws,
        params=params
    )
    f1 = f1_score(y_val_true, y_val_pred, average='macro')
    return {**hp, "macro_f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Train pit-stop prediction model")
    parser.add_argument(
        "-t", "--telemetry",
        action="store_true",
        help="include in-sector telemetry features"
    )
    parser.add_argument(
        "-a", "--all", 
        action="store_true", 
        help="run both base and telemetry models back-to-back")
    args = parser.parse_args()
    if args.all:
        python = sys.executable
        runs = [
            [python, sys.argv[0]],
            [python, sys.argv[0], "--telemetry"]
        ]
        for cmd in runs:
            print(f">>> {' '.join(cmd)}\n")
            subprocess.run(cmd, check=True)
        return
    numeric_cols = TELEMETRY_NUMERIC_COLS if args.telemetry else BASE_NUMERIC_COLS
    categorical_cols = TELEMETRY_CATEGORICAL_COLS if args.telemetry else BASE_CATEGORICAL_COLS
    features_to_keep = numeric_cols + categorical_cols + ['Year', 'Driver', 'SectorNumber', 'LapsTilPit']
    print("=== Starting train.py ===\n")
    print(f"Using {'telemetry+' if args.telemetry else 'base'} numeric features: {numeric_cols}")
    print(f"Using {'telemetry+' if args.telemetry else 'base'} categorical features: {categorical_cols}")
    model_suffix = "telemetry" if args.telemetry else "base"
    model_filename = f"models/model_{model_suffix}.pkl"

    # Preprocess data
    train_df, val_df, test_df, label_col, scaler_dict, n_classes, encoders = run_preprocessing_pipeline(
        filepath='data/f1_sector_data.pkl',
        features_to_keep=features_to_keep,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        laps_label_window=5
    )
    print("\nFinished preprocessing: train_df shape =", train_df.shape)
    print("                        val_df shape    =", val_df.shape)
    print("                        test_df shape   =", test_df.shape)
    print("                        label           =", label_col)

    # Save scaler and encoder
    scaler_path = f"preprocess/scaler_dict_{model_suffix}.pkl"
    print(f"\nSaving scaler to {scaler_path}.")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_dict, f)
    encoder_path = f"preprocess/encoder_{model_suffix}.pkl"
    print(f"Saving encoder to {encoder_path}.")
    with open(encoder_path, "wb") as f:
        pickle.dump(encoders, f)

    # Grid search
    keys, values = zip(*PARAM_GRID.items())
    grid = [dict(zip(keys, vs)) for vs in product(*values)]
    print(f"\nStarting grid search over {len(grid)} hyperparameter combinations.\n")
    results = []
    with ProcessPoolExecutor(max_workers=4) as exe:
        futures = [
            exe.submit(run_once, hp, train_df, val_df, test_df, label_col, categorical_cols, n_classes, numeric_cols, features_to_keep)
            for hp in grid
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Grid search"):
            results.append(fut.result())
    results.sort(key=lambda x: x["macro_f1"], reverse=True)
    best_hp = results[0]
    print("\nBest hyperparameters (by val macro F1):", best_hp)

    # Retrain best model on train+val, test on test
    print("\nRetraining best model on train+val, evaluating on test.")
    window_size = best_hp["window_size"]
    params = {
        "num_lstm_layers":  best_hp["num_lstm_layers"],
        "lstm_units":       best_hp["lstm_units"],
        "dropout_rate":     best_hp["dropout_rate"],
        "recurrent_dropout":best_hp["recurrent_dropout"],
        "learning_rate":    best_hp["learning_rate"],
        "batch_size":       best_hp["batch_size"],
        "epochs":           best_hp["epochs"],
    }
    # Concatenate train and val for final training
    trainval_df = pd.concat([train_df, val_df], axis=0)
    model, y_test_pred, y_test_true = run_experiment(
        trainval_df, val_df, test_df,
        feature_cols=features_to_keep,
        label_col=label_col,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        embedding_dims_dict=EMBEDDING_DIMS_DICT,
        n_classes=n_classes,
        window_size=window_size,
        params=params
    )
    test_macro_f1 = f1_score(y_test_true, y_test_pred, average='macro')
    print(f"\nTest macro F1: {test_macro_f1:.4f}")

    # Save model
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel training complete and saved to {model_filename}.\n\n")

if __name__ == "__main__":
    main()

