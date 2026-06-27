"""Image classification utility using pretrained models."""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

_logger = logging.getLogger(__name__)


class ImageClassifier:
    """Lightweight image classifier using TensorFlow/Keras pretrained models."""

    def __init__(self, model_name: str = "mobilenetv2"):
        """Initialize classifier with specified model.
        
        Args:
            model_name: Model to use ('mobilenetv2' or 'resnet50')
        """
        self.model_name = model_name.lower()
        self.model = None
        # Set display name based on model
        if self.model_name == "mobilenetv2":
            self.model_name_display = "MobileNetV2"
        elif self.model_name == "resnet50":
            self.model_name_display = "ResNet50"
        else:
            self.model_name_display = model_name
        self._load_model()

    def _load_model(self) -> None:
        """Load the specified pretrained model."""
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import MobileNetV2, ResNet50, preprocess_input
            from tensorflow.keras.preprocessing import image as keras_image
            
            self.tf = tf
            self.preprocess_input = preprocess_input
            self.keras_image = keras_image
            
            if self.model_name == "mobilenetv2":
                self.model = MobileNetV2(weights="imagenet")
            elif self.model_name == "resnet50":
                self.model = ResNet50(weights="imagenet")
            else:
                self.model = MobileNetV2(weights="imagenet")
                
            _logger.info(f"Loaded {self.model_name_display} model")
        except ImportError:
            _logger.warning("TensorFlow not available, using PIL-only classification")
            self.model = None
            self.tf = None

    def predict(self, image: Image.Image) -> Tuple[str, float, Dict[str, Any]]:
        """Classify an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (predicted_class, confidence, details_dict)
        """
        if self.model is None:
            return self._fallback_classify(image)
        
        try:
            # Prepare image
            img_resized = image.convert("RGB").resize((224, 224))
            img_array = self.keras_image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = self.preprocess_input(img_array)
            
            # Predict
            predictions = self.model.predict(img_array, verbose=0)
            
            # Decode predictions
            from tensorflow.keras.applications import imagenet_utils
            decoded = imagenet_utils.decode_predictions(predictions, top=5)[0]
            
            top_pred = decoded[0]
            class_name = top_pred[1].replace("_", " ").title()
            confidence = float(top_pred[2]) * 100
            
            # Top 5 alternatives
            top_5 = [
                {
                    "class": pred[1].replace("_", " ").title(),
                    "confidence": float(pred[2]) * 100,
                }
                for pred in decoded
            ]
            
            return class_name, confidence, {
                "top_5": top_5,
                "model": self.model_name_display,
                "input_size": "224x224",
            }
        except Exception as e:
            _logger.error(f"Classification error: {e}")
            return self._fallback_classify(image)

    def _fallback_classify(self, image: Image.Image) -> Tuple[str, float, Dict[str, Any]]:
        """Fallback simple classification based on image properties."""
        try:
            # Simple heuristic based on image properties
            img_array = np.array(image.convert("RGB"))
            
            # Calculate basic statistics
            mean_r = np.mean(img_array[:, :, 0])
            mean_g = np.mean(img_array[:, :, 1])
            mean_b = np.mean(img_array[:, :, 2])
            
            brightness = (mean_r + mean_g + mean_b) / 3
            
            # Classify based on color dominance
            if mean_r > mean_g and mean_r > mean_b:
                predicted_class = "Red Image"
            elif mean_g > mean_r and mean_g > mean_b:
                predicted_class = "Green Image"
            elif mean_b > mean_r and mean_b > mean_g:
                predicted_class = "Blue Image"
            elif brightness > 200:
                predicted_class = "Bright Image"
            elif brightness < 100:
                predicted_class = "Dark Image"
            else:
                predicted_class = "Regular Image"
            
            confidence = 65.0  # Default confidence for heuristic
            
            return predicted_class, confidence, {
                "top_5": [
                    {"class": predicted_class, "confidence": confidence},
                    {"class": "Image (Generic)", "confidence": 25.0},
                ],
                "model": "Heuristic (TensorFlow unavailable)",
                "input_size": f"{image.width}x{image.height}",
            }
        except Exception as e:
            _logger.error(f"Fallback classification error: {e}")
            return "Unknown", 0.0, {"error": str(e)}


def load_image_from_bytes(image_bytes: bytes) -> Optional[Image.Image]:
    """Load PIL Image from bytes.
    
    Args:
        image_bytes: Image file bytes
        
    Returns:
        PIL Image or None if loading fails
    """
    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        _logger.error(f"Failed to load image: {e}")
        return None


def validate_image_format(file_extension: str) -> bool:
    """Check if file extension is a supported image format.
    
    Args:
        file_extension: File extension (e.g., 'jpg', 'png')
        
    Returns:
        True if supported format
    """
    supported = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}
    return file_extension.lower() in supported
