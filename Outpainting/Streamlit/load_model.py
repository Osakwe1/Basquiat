from tensorflow.keras import models

def load_model(model_path_gen):
    """
    load the latest saved model
    """
    model_generator = models.load_model(model_path_gen)
    print("\n✅ generator loaded from ComputeEngine")

    return model_generator
