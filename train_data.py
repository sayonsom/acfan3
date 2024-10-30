import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Dropout, 
    Layer, Lambda, concatenate
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, 
    ReduceLROnPlateau, TensorBoard
)
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
        # Create interaction features
        interaction = inputs[:, 0] * inputs[:, 1]
        interaction = tf.expand_dims(interaction, -1)
        
        # Create squared features
        squares = tf.square(inputs)
        
        # Concatenate all features
        return concatenate([inputs, interaction, squares])
    

def create_model(input_shape=(2,), reg_factor=0.005):  # Reduced regularization
    inputs = Input(shape=input_shape)
    
    # Feature engineering layer
    x = CustomFeatureLayer()(inputs)
    
    # Increased capacity in first layers
    x = Dense(256, activation='relu', kernel_regularizer=l2(reg_factor))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)  # Reduced dropout
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(reg_factor))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l2(reg_factor))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Output layers with smaller initializer range
    temp_output = Dense(
        1, 
        name='temperature',
        kernel_initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05)
    )(x)
    
    vel_output = Dense(
        1, 
        name='velocity',
        kernel_initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05)
    )(x)
    
    return Model(inputs=inputs, outputs=[temp_output, vel_output])




def save_training_plots(history, fold=None, save_dir='plots'):
    """
    Save detailed training history plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = ['loss', 'temperature_loss', 'velocity_loss',
               'temperature_mae', 'velocity_mae']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Plot training metric
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
        
        # Save with fold number if doing cross-validation
        suffix = f'_fold_{fold}' if fold is not None else ''
        plt.savefig(os.path.join(save_dir, f'{metric}{suffix}.png'))
        plt.close()

def prepare_data():
    """
    Load and prepare data with validation
    """
    logging.info("Loading data...")
    try:
        df = pd.read_csv('synthetic_ac_fan_data.csv')
        logging.info(f"Data loaded successfully! Shape: {df.shape}")
        
        # Basic data validation
        assert not df.isnull().any().any(), "Dataset contains null values"
        assert df['Outdoor_Temperature'].between(0, 50).all(), "Invalid temperature range"
        assert df['PMV'].between(-3, 3).all(), "Invalid PMV range"
        
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def train_model(X_train, X_test, y_train, y_test, fold=None):
    """
    Train model with given data split
    """
    # Create and compile model
    model = create_model()
    
    optimizer = tf.keras.optimizers.Adam(
    learning_rate=5e-4,  # Lower initial learning rate
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07)

    model.compile(
        optimizer=optimizer,
        loss={
            'temperature': 'mse',
            'velocity': 'mse'
        },
        metrics={
            'temperature': ['mae', 'mse'],
            'velocity': ['mae', 'mse']
        }
    )

   
    
    # Prepare callbacks
    fold_suffix = f'_fold_{fold}' if fold is not None else ''
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=25,  # Increased patience
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
            factor=0.1,  # More aggressive LR reduction
            patience=15,
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
            'temperature': y_train[:, 0],
            'velocity': y_train[:, 1]
        },
        validation_data=(
            X_test,
            {
                'temperature': y_test[:, 0],
                'velocity': y_test[:, 1]
            }
        ),
        epochs=100,
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
        df = prepare_data()
        
        # Prepare features and targets
        X = df[['Outdoor_Temperature', 'PMV']].values
        y = np.column_stack((
            df['Air_Temperature'].values,
            df['Air_Velocity'].values
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
                fold
            )
            
            # Save training plots
            save_training_plots(history, fold)
            
            # Evaluate model
            scores = model.evaluate(
                X_val_scaled,
                {
                    'temperature': y_val_scaled[:, 0],
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
            y_scaled
        )
        
        # Save final model
        final_model.save('models/final_model.keras')
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()