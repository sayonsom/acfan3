import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Layer, Lambda, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CustomFeatureLayer(Layer):
    """Custom layer for feature engineering"""
    def __init__(self, **kwargs):
        super(CustomFeatureLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Square features
        squares = tf.square(inputs)
        
        # Simple pairwise feature (multiply adjacent features)
        # Reshape to ensure proper broadcasting
        shifted = inputs[:, 1:]
        original = inputs[:, :-1]
        pairwise = original * shifted
        
        # Concatenate original features with engineered features
        return concatenate([inputs, squares, pairwise])
    
    def compute_output_shape(self, input_shape):
        # Original features + squared features + pairwise features
        return (input_shape[0], input_shape[1] * 3 - 1)

def create_model(input_shape, reg_factor=0.001):
    inputs = Input(shape=input_shape)
    
    # Feature engineering layer
    x = CustomFeatureLayer()(inputs)
    
    # First block with larger capacity
    x = Dense(256, activation='relu', kernel_regularizer=l2(reg_factor))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second block
    x = Dense(128, activation='relu', kernel_regularizer=l2(reg_factor))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Third block
    x = Dense(64, activation='relu', kernel_regularizer=l2(reg_factor))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Output layers
    setpoint_output = Dense(
        1, 
        name='setpoint',
        kernel_initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05)
    )(x)
    
    velocity_output = Dense(
        1, 
        name='velocity',
        kernel_initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05)
    )(x)
    
    return Model(inputs=inputs, outputs=[setpoint_output, velocity_output])

def save_training_plots(history, fold=None, save_dir='plots'):
    """Save detailed training history plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = ['loss', 'setpoint_loss', 'velocity_loss',
               'setpoint_mae', 'velocity_mae']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Plot training metric
        if metric in history.history:
            plt.plot(history.history[metric], label=f'Training {metric}')
        
        # Plot validation metric
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=f'Validation {metric}')
        
        plt.title(f'{metric.replace("_", " ").title()} Over Time')
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(True)
        
        suffix = f'_fold_{fold}' if fold is not None else ''
        plt.savefig(os.path.join(save_dir, f'{metric}{suffix}.png'))
        plt.close()

def prepare_data():
    """Load and prepare data with validation"""
    logging.info("Loading data...")
    try:
        df = pd.read_csv('synthetic_ac_fan_data_combined.csv')  # Update with your CSV filename
        logging.info(f"Data loaded successfully! Shape: {df.shape}")
        
        # Select input features
        input_features = [
            'Total_Occupants', 'Actual_Indoor_Temp', 'Outdoor_Temperature',
            'AC_Capacity', 'Humidity', 'Air_Velocity', 'PMV_Initial',
            'PMV_Target', 'Metabolic_Rate', 'Clothing_Insulation', 'Age'
        ]
        
        # Basic data validation
        assert not df[input_features].isnull().any().any(), "Dataset contains null values"
        
        return df, input_features
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def train_model(X_train, X_test, y_train, y_test, input_shape, fold=None):
    """Train model with given data split"""
    # Create and compile model
    model = create_model(input_shape)
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )

    model.compile(
        optimizer=optimizer,
        loss={
            'setpoint': 'mse',
            'velocity': 'mse'
        },
        metrics={
            'setpoint': ['mae', 'mse'],
            'velocity': ['mae', 'mse']
        }
    )
    
    # Prepare callbacks
    fold_suffix = f'_fold_{fold}' if fold is not None else ''
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            f'models/best_model{fold_suffix}.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=f'logs/fit/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )
    ]
    
    # Train model
    history = model.fit(
        X_train,
        {
            'setpoint': y_train[:, 0],
            'velocity': y_train[:, 1]
        },
        validation_data=(
            X_test,
            {
                'setpoint': y_test[:, 0],
                'velocity': y_test[:, 1]
            }
        ),
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('scalers', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    try:
        # Load and prepare data
        df, input_features = prepare_data()
        
        # Prepare features and targets
        X = df[input_features].values
        y = np.column_stack((
            df['AC_Setpoint(MRT)'].values,
            df['Desired_AirVelocity'].values
        ))
        
        # Initialize scalers
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        # Perform k-fold cross-validation
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logging.info(f"\nTraining Fold {fold + 1}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale data
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_val_scaled = scaler_X.transform(X_val)
            y_train_scaled = scaler_y.fit_transform(y_train)
            y_val_scaled = scaler_y.transform(y_val)
            
            # Train model
            model, history = train_model(
                X_train_scaled, 
                X_val_scaled, 
                y_train_scaled, 
                y_val_scaled,
                input_shape=(len(input_features),),
                fold=fold
            )
            
            # Save training plots
            save_training_plots(history, fold)
            
            # Evaluate model
            scores = model.evaluate(
                X_val_scaled,
                {
                    'setpoint': y_val_scaled[:, 0],
                    'velocity': y_val_scaled[:, 1]
                },
                verbose=0
            )
            fold_scores.append(scores)
            
            logging.info(f"Fold {fold + 1} Scores:")
            for metric_name, score in zip(model.metrics_names, scores):
                logging.info(f"{metric_name}: {score:.4f}")
        
        # Save final scalers (from last fold)
        joblib.dump(scaler_X, 'scalers/scaler_X.joblib')
        joblib.dump(scaler_y, 'scalers/scaler_y.joblib')
        
        # Print average scores across folds
        logging.info("\nAverage scores across folds:")
        mean_scores = np.mean(fold_scores, axis=0)
        for metric_name, score in zip(model.metrics_names, mean_scores):
            logging.info(f"Mean {metric_name}: {score:.4f}")
        
        # Train final model on all data
        logging.info("\nTraining final model on all data...")
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        final_model, final_history = train_model(
            X_scaled, 
            X_scaled,  # Using same data for validation
            y_scaled, 
            y_scaled,
            input_shape=(len(input_features),)
        )
        
        # Save final model
        final_model.save('models/final_model.keras')
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()