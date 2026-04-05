import os
import tensorflow as tf

print("="*60)
print("TESTING MODEL LOADING")
print("="*60)

# Current directory
print(f"Current directory: {os.getcwd()}")

# List all files
print(f"\nFiles in directory:")
for f in os.listdir('.'):
    if f.endswith('.h5'):
        size = os.path.getsize(f) / (1024*1024)
        print(f"  - {f} ({size:.2f} MB)")

# Try loading each model
print("\n" + "="*60)
print("LOADING MODELS")
print("="*60)

models = [
    'best_pneumonia_model.h5',
    'best_pathmnist_model.h5',
    'best_retinamnist_model.h5',
    'best_brain_tumor_fixed.h5'
]

for model_file in models:
    print(f"\nAttempting to load: {model_file}")
    if os.path.exists(model_file):
        try:
            model = tf.keras.models.load_model(model_file)
            print(f"✅ SUCCESS: {model_file} loaded!")
            print(f"   Model type: {type(model)}")
        except Exception as e:
            print(f"❌ ERROR loading {model_file}:")
            print(f"   {str(e)}")
    else:
        print(f"❌ FILE NOT FOUND: {model_file}")

print("\n" + "="*60)
