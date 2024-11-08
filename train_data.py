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
np.random.seed(2024)
tf.random.set_seed(2024)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CustomFeatureLayer(Layer):
    """Custom layer for feature engineering with proper dimension handling"""
    def __init__(self, **kwargs):
        super(CustomFeatureLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Ensure inputs are rank 2
        if tf.rank(inputs) != 2:
            raise ValueError(f"Expected rank 2 input, got rank {tf.rank(inputs)}")
            
        # Square features
        squares = tf.square(inputs)
        
        # Create pairwise interactions more safely
        input_shape = tf.shape(inputs)
        if input_shape[1] > 1:  # Only create pairwise if we have at least 2 features
            # Take first n-1 features and multiply with next feature
            left = inputs[:, :-1]
            right = inputs[:, 1:]
            pairwise = left * right
            # Concatenate all features
            return tf.concat([inputs, squares, pairwise], axis=1)
        else:
            # If we only have one feature, just return original and squared
            return tf.concat([inputs, squares], axis=1)

def create_model(input_shape, reg_factor=0.0005):  # Reduced regularization
    inputs = Input(shape=input_shape)
    
    # Feature engineering layer
    x = CustomFeatureLayer()(inputs)
    
    # Deeper architecture with residual connections
    x1 = Dense(512, activation='relu', kernel_regularizer=l2(reg_factor))(x)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.4)(x1)
    
    x2 = Dense(256, activation='relu', kernel_regularizer=l2(reg_factor))(x1)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    
    # Residual connection
    x2 = concatenate([x2, x1])
    
    x3 = Dense(128, activation='relu', kernel_regularizer=l2(reg_factor))(x2)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.2)(x3)
    
    # Final dense layers for each output
    setpoint_hidden = Dense(64, activation='relu', name='setpoint_hidden')(x3)
    velocity_hidden = Dense(64, activation='relu', name='velocity_hidden')(x3)
    
    # Output layers with appropriate activation functions
    setpoint_output = Dense(1, name='setpoint')(setpoint_hidden)
    velocity_output = Dense(1, name='velocity', activation='sigmoid')(velocity_hidden)
    
    return Model(inputs=inputs, outputs=[setpoint_output, velocity_output])

def custom_loss():
    mse = tf.keras.losses.MeanSquaredError()
    
    def combined_loss(y_true, y_pred):
        # MSE loss
        mse_loss = mse(y_true, y_pred)
        
        # Add penalty for large changes in predictions
        smoothness_loss = tf.reduce_mean(tf.square(y_pred[:, 1:] - y_pred[:, :-1]))
        
        return mse_loss + 0.1 * smoothness_loss
    
    return combined_loss

def create_model(input_shape, reg_factor=0.0005):
    """Create model with improved error handling"""
    inputs = Input(shape=input_shape)
    
    # Feature engineering layer with error handling
    try:
        x = CustomFeatureLayer()(inputs)
    except Exception as e:
        logging.error(f"Error in CustomFeatureLayer: {e}")
        # Fallback to just using original inputs if feature engineering fails
        x = inputs
    
    # First block
    x = Dense(256, activation='relu', kernel_regularizer=l2(reg_factor))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second block with residual connection
    x2 = Dense(128, activation='relu', kernel_regularizer=l2(reg_factor))(x)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.2)(x2)
    
    # Third block
    x3 = Dense(64, activation='relu', kernel_regularizer=l2(reg_factor))(x2)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.1)(x3)
    
    # Separate branches for setpoint and velocity
    setpoint_hidden = Dense(32, activation='relu', kernel_regularizer=l2(reg_factor))(x3)
    velocity_hidden = Dense(32, activation='relu', kernel_regularizer=l2(reg_factor))(x3)
    
    # Output layers
    setpoint_output = Dense(1, name='setpoint')(setpoint_hidden)
    velocity_output = Dense(1, name='velocity')(velocity_hidden)
    
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

def train_model(X_train, X_test, y_train, y_test, input_shape, fold=None):
    """Train model with improved error handling and simpler loss functions"""
    model = create_model(input_shape)
    
    # Simple Adam optimizer with fixed learning rate initially
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss={
            'setpoint': 'mse',
            'velocity': 'mse'
        },
        metrics={
            'setpoint': ['mae'],
            'velocity': ['mae']
        }
    )
    
    # Simplified callbacks
    fold_suffix = f'_fold_{fold}' if fold is not None else ''
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
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
        )
    ]
    
    # Train with smaller batch size initially
    try:
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
            epochs=15,
            batch_size=32,  # Reduced batch size
            callbacks=callbacks,
            verbose=1
        )
        return model, history
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise
def prepare_data():
    """Load and prepare data with minimal but important features"""
    logging.info("Loading data...")
    try:
        df = pd.read_csv('synthetic_ac_fan_data.csv')
        logging.info(f"Data loaded successfully! Shape: {df.shape}")
        
        # Reduced set of most important features
        input_features = [
            'Actual_Indoor_Temp',
            'Outdoor_Temperature',
            'Room_Volume',
            'AC_Capacity',
            'Humidity',
            'PMV_Target',
            'Hour'
        ]
        
        # Basic data validation
        assert not df[input_features].isnull().any().any(), "Dataset contains null values"
        
        return df, input_features
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


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