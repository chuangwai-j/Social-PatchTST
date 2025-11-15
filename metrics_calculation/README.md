# Metrics Calculation Directory

This directory contains training metrics and calculation results for the Social-PatchTST model.

## 📁 File Structure

- `final_training_metrics.json` - Complete training metrics after training completion
- `training_metrics_epoch_{N}.json` - Intermediate metrics saved every 5 epochs
- `baseline_training_metrics.json` - Baseline model metrics (if --baseline mode is used)
- `social_patchtst_training_metrics.json` - Social-PatchTST model metrics

## 📊 Metrics Included

### Training Losses
- `train_total_loss` - Overall training loss
- `train_position_loss` - Position prediction loss
- `train_altitude_loss` - Altitude prediction loss
- `train_velocity_loss` - Velocity prediction loss
- `train_mindist_loss` - Minimum distance prediction loss

### Validation Losses
- `val_total_loss` - Overall validation loss
- `val_position_loss` - Position validation loss
- `val_altitude_loss` - Altitude validation loss
- `val_velocity_loss` - Velocity validation loss
- `val_mindist_loss` - Minimum distance validation loss

### Performance Metrics
- `val_position_rmse` - Position Root Mean Square Error
- `val_altitude_rmse` - Altitude Root Mean Square Error
- `val_velocity_rmse` - Velocity Root Mean Square Error
- `val_position_mae` - Position Mean Absolute Error
- `val_altitude_mae` - Altitude Mean Absolute Error
- `val_velocity_mae` - Velocity Mean Absolute Error
- `val_far` - False Alarm Rate (social model specific)

### Training Information
- `learning_rates` - Learning rate history
- `epoch_times` - Training time per epoch

## 🔍 Usage

These metrics files are used for:
1. **Performance Analysis** - Compare model performance across epochs
2. **Paper Writing** - Extract results for academic publications
3. **Model Comparison** - Baseline vs Social-PatchTST performance
4. **Hyperparameter Tuning** - Analyze training dynamics

## 📈 Example Usage

```python
import json

# Load final metrics
with open('metrics_calculation/final_training_metrics.json', 'r') as f:
    metrics = json.load(f)

# Access best epoch metrics
best_epoch = metrics['experiment_info']['best_epoch']
best_metrics = metrics['best_epoch_metrics']

print(f"Best RMSE: {best_metrics['val_position_rmse']:.6f}")
print(f"Best FAR: {best_metrics['val_far']:.6f}")
```

## 🧪 Comparison Experiments

For comparing Baseline vs Social-PatchTST:
- Baseline metrics are saved with `_baseline` suffix
- Social-PatchTST metrics are saved with `_social` suffix
- Use the same evaluation metrics for fair comparison