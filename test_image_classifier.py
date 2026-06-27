"""Test script for image classification module."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
from utils.image_classifier import ImageClassifier, load_image_from_bytes, validate_image_format


def create_test_image(width: int = 224, height: int = 224) -> Image.Image:
    """Create a simple test image."""
    # Create an RGB image with random colors
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(data, mode="RGB")


def test_image_format_validation():
    """Test image format validation."""
    print("Testing image format validation...")
    assert validate_image_format("jpg") == True
    assert validate_image_format("JPG") == True
    assert validate_image_format("png") == True
    assert validate_image_format("PNG") == True
    assert validate_image_format("gif") == True
    assert validate_image_format("bmp") == True
    assert validate_image_format("webp") == True
    assert validate_image_format("txt") == False
    assert validate_image_format("pdf") == False
    print("✓ Image format validation passed")


def test_classifier_initialization():
    """Test classifier initialization."""
    print("\nTesting classifier initialization...")
    
    # Test MobileNetV2
    print("  Loading MobileNetV2...")
    classifier_v2 = ImageClassifier("mobilenetv2")
    assert classifier_v2.model_name == "mobilenetv2"
    assert classifier_v2.model_name_display == "MobileNetV2"
    print("  ✓ MobileNetV2 initialized")
    
    # Test ResNet50
    print("  Loading ResNet50...")
    classifier_r50 = ImageClassifier("resnet50")
    assert classifier_r50.model_name == "resnet50"
    assert classifier_r50.model_name_display == "ResNet50"
    print("  ✓ ResNet50 initialized")


def test_prediction():
    """Test image classification prediction."""
    print("\nTesting image classification...")
    
    # Create test image
    test_image = create_test_image()
    print(f"  Created test image: {test_image.size}")
    
    # Try with MobileNetV2 first
    print("  Predicting with MobileNetV2...")
    classifier = ImageClassifier("mobilenetv2")
    predicted_class, confidence, details = classifier.predict(test_image)
    
    print(f"  Predicted class: {predicted_class}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"  Model: {details.get('model')}")
    
    assert isinstance(predicted_class, str)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 100
    assert isinstance(details, dict)
    print("  ✓ Prediction successful")


def test_image_loading():
    """Test image loading from bytes."""
    print("\nTesting image loading from bytes...")
    
    # Create a test image and save as bytes
    test_image = create_test_image()
    image_bytes = b""
    import io
    buffer = io.BytesIO()
    test_image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    
    print(f"  Image bytes size: {len(image_bytes)} bytes")
    
    # Load from bytes
    loaded_image = load_image_from_bytes(image_bytes)
    assert loaded_image is not None
    assert loaded_image.size == test_image.size
    print(f"  ✓ Image loaded: {loaded_image.size}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Image Classification Module Tests")
    print("=" * 60)
    
    try:
        test_image_format_validation()
        test_classifier_initialization()
        test_image_loading()
        test_prediction()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
