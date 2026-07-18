"""
Image Loader Utility
Handles multiple image formats including PNG, JPG, TIF, TIFF, and CR3.
Converts all formats to RGB numpy arrays compatible with the pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple
import warnings


class ImageLoader:
    """Unified image loading interface for diverse formats."""
    
    # Supported formats
    STANDARD_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    RAW_FORMATS = {'.cr3', '.cr2', '.nef', '.arw', '.dng'}
    ALL_FORMATS = STANDARD_FORMATS | RAW_FORMATS
    
    @staticmethod
    def get_supported_extensions():
        """Return tuple of supported file extensions."""
        return tuple(ImageLoader.ALL_FORMATS)
    
    @staticmethod
    def load_image_cv2(path: str) -> Optional[np.ndarray]:
        """
        Load image using OpenCV.
        
        Args:
            path: Path to image file
            
        Returns:
            BGR numpy array or None if failed
        """
        try:
            img = cv2.imread(str(path))
            return img
        except Exception as e:
            warnings.warn(f"CV2 failed to load {path}: {e}")
            return None
    
    @staticmethod
    def load_image_pil(path: str, target_format: str = "RGB") -> Optional[np.ndarray]:
        """
        Load image using PIL.
        
        Args:
            path: Path to image file
            target_format: Target color space ("RGB" or "RGBA")
            
        Returns:
            RGB/RGBA numpy array or None if failed
        """
        try:
            img = Image.open(path)
            if target_format == "RGB":
                img = img.convert("RGB")
            elif target_format == "RGBA":
                img = img.convert("RGBA")
            return np.array(img)
        except Exception as e:
            warnings.warn(f"PIL failed to load {path}: {e}")
            return None
    
    @staticmethod
    def load_cr3(path: str) -> Optional[np.ndarray]:
        """
        Load CR3 (Canon RAW) image and convert to RGB.
        
        Args:
            path: Path to CR3 file
            
        Returns:
            RGB numpy array (uint8) or None if failed
        """
        try:
            import rawpy
            
            with rawpy.imread(str(path)) as raw:
                # Try different rawpy API versions
                # Newer versions don't have bayer_dither_fs parameter
                try:
                    # Try new API first (works with recent rawpy)
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        output_bps=8,
                    )
                except TypeError:
                    # Fall back to old API with bayer_dither_fs
                    try:
                        rgb = raw.postprocess(
                            bayer_dither_fs=None,
                            half_size=False,
                            output_bps=8,
                        )
                    except TypeError:
                        # Fallback to minimal postprocess
                        rgb = raw.postprocess(output_bps=8)
            return rgb
        except ImportError:
            error_msg = (
                f"\n{'='*80}\n"
                f"⚠️  RAWPY NOT INSTALLED - Cannot load CR3 files!\n"
                f"{'='*80}\n"
                f"File: {path}\n"
                f"\nFIX THIS ISSUE:\n"
                f"\n1. Create a virtual environment (recommended):\n"
                f"   On Linux/macOS:  bash setup_env.sh\n"
                f"   On Windows:      setup_env.bat\n"
                f"\n2. Or install rawpy directly:\n"
                f"   pip install --break-system-packages rawpy\n"
                f"\n3. Then verify:\n"
                f"   python -c \"import rawpy; print('✅ rawpy installed')\"\n"
                f"\nDoc: See ENVIRONMENT_SETUP_GUIDE.md for full instructions\n"
                f"{'='*80}\n"
            )
            warnings.warn(error_msg)
            return None
        except Exception as e:
            error_msg = (
                f"\n{'='*80}\n"
                f"❌ ERROR loading CR3 file: {path}\n"
                f"{'='*80}\n"
                f"Error: {e}\n"
                f"\nPossible causes:\n"
                f"  1. File is corrupt or not a valid CR3 image\n"
                f"  2. rawpy is not properly installed\n"
                f"  3. Insufficient disk space or permissions\n"
                f"\nDebugging steps:\n"
                f"  python -c \"import rawpy; print('rawpy OK')\"\n"
                f"  file {path}  # Check file type\n"
                f"{'='*80}\n"
            )
            warnings.warn(error_msg)
            return None
    
    @staticmethod
    def load_raw_image(path: str) -> Optional[np.ndarray]:
        """
        Load RAW formats (CR3, CR2, NEF, ARW, DNG).
        
        Args:
            path: Path to RAW file
            
        Returns:
            RGB numpy array or None if failed
        """
        ext = Path(path).suffix.lower()
        
        if ext == '.cr3':
            return ImageLoader.load_cr3(path)
        else:
            # For other RAW formats, try rawpy
            try:
                import rawpy
                with rawpy.imread(str(path)) as raw:
                    # Try different rawpy API versions
                    try:
                        # Try new API first (works with recent rawpy)
                        rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
                    except TypeError:
                        # Fall back to old API with bayer_dither_fs
                        try:
                            rgb = raw.postprocess(bayer_dither_fs=None, half_size=False, output_bps=8)
                        except TypeError:
                            # Fallback to minimal postprocess
                            rgb = raw.postprocess(output_bps=8)
                return rgb
            except ImportError:
                error_msg = (
                    f"\n{'='*70}\n"
                    f"⚠️  RAWPY NOT INSTALLED - Cannot load {ext.upper()} files!\n"
                    f"{'='*70}\n"
                    f"File: {path}\n"
                    f"\nTo process RAW files, install rawpy:\n"
                    f"  pip install rawpy\n"
                    f"\nOn Linux, you may also need system dependencies:\n"
                    f"  sudo apt-get install libcaca-dev\n"
                    f"{'='*70}\n"
                )
                warnings.warn(error_msg)
                return None
            except Exception as e:
                error_msg = (
                    f"\n{'='*70}\n"
                    f"❌ ERROR loading {ext.upper()} file: {path}\n"
                    f"{'='*70}\n"
                    f"Error: {e}\n"
                    f"\nMake sure:\n"
                    f"  1. rawpy is installed: pip install rawpy\n"
                    f"  2. File is a valid {ext.upper()} image\n"
                    f"  3. File is not corrupted\n"
                    f"{'='*70}\n"
                )
                warnings.warn(error_msg)
                return None
    
    @staticmethod
    def load_image(path: str, return_format: str = "rgb") -> Optional[np.ndarray]:
        """
        Universal image loader. Automatically detects format and loads appropriately.
        
        Args:
            path: Path to image file
            return_format: "rgb" for RGB (default), "bgr" for BGR, "rgba" for RGBA
            
        Returns:
            numpy array in requested format or None if failed
            
        Raises:
            ValueError: If return_format is invalid
        """
        if return_format not in ["rgb", "bgr", "rgba"]:
            raise ValueError(f"return_format must be 'rgb', 'bgr', or 'rgba', got {return_format}")
        
        path = Path(path)
        ext = path.suffix.lower()
        
        if not path.exists():
            warnings.warn(f"File does not exist: {path}")
            return None
        
        img = None
        
        # Try loading RAW formats first
        if ext in ImageLoader.RAW_FORMATS:
            img = ImageLoader.load_raw_image(str(path))
        
        # Fall back to PIL for standard formats
        if img is None and ext in ImageLoader.STANDARD_FORMATS:
            if return_format == "bgr":
                img = ImageLoader.load_image_cv2(str(path))
            else:
                img = ImageLoader.load_image_pil(str(path), target_format="RGB")
        
        # Final fallback: try both methods
        if img is None:
            img = ImageLoader.load_image_pil(str(path), target_format="RGB")
        if img is None:
            img = ImageLoader.load_image_cv2(str(path))
        
        # Convert to requested format
        if img is not None and len(img.shape) == 3:
            if return_format == "rgb":
                # Ensure RGB
                if img.shape[2] == 4:  # RGBA to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                elif img.shape[2] == 3:
                    # Check if it's BGR (from cv2) or RGB
                    # We'll rely on PIL being called for RGB formats
                    pass
            elif return_format == "bgr":
                # Ensure BGR
                if img.shape[2] == 4:  # RGBA to BGR
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif img.shape[2] == 3:
                    # Assume we need BGR for cv2
                    pass
        
        return img
    
    @staticmethod
    def load_pil_image(path: str) -> Optional[Image.Image]:
        """
        Load image as PIL Image object.
        
        Args:
            path: Path to image file
            
        Returns:
            PIL Image or None if failed
        """
        try:
            return Image.open(path)
        except Exception as e:
            warnings.warn(f"Failed to load image as PIL: {path}: {e}")
            return None


def convert_output_filename(filename: str, target_format: str = "png") -> str:
    """
    Convert filename to a writable format.
    
    RAW formats (CR3, CR2, NEF, etc.) can't be written by OpenCV,
    so convert them to standard formats for output.
    
    Args:
        filename: Original filename (e.g., 'photo.CR3', 'photo.jpg')
        target_format: Target format ('png', 'jpg', 'tif')
        
    Returns:
        Converted filename (e.g., 'photo.png')
    """
    path = Path(filename)
    ext = path.suffix.lower()
    
    # RAW formats that need conversion
    raw_formats = {'.cr3', '.cr2', '.nef', '.arw', '.dng'}
    
    # If it's a RAW format or if target format differs, convert
    if ext in raw_formats:
        return str(path.with_suffix(f".{target_format}"))
    
    # If extension is not a standard writable format, convert it
    writable_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    if ext not in writable_formats:
        return str(path.with_suffix(f".{target_format}"))
    
    # Otherwise keep original
    return filename


def get_image_files(directory: str, supported_only: bool = True) -> list:
    """
    Get all supported image files from a directory.
    
    Args:
        directory: Path to directory
        supported_only: If True, only return supported formats
        
    Returns:
        List of Path objects
    """
    dir_path = Path(directory)
    
    if not dir_path.is_dir():
        return []
    
    if supported_only:
        extensions = ImageLoader.get_supported_extensions()
        files = []
        seen = set()
        for ext in extensions:
            for f in list(dir_path.glob(f"*{ext}")) + list(dir_path.glob(f"*{ext.upper()}")):
                # Déduplique via le chemin résolu (crucial sur Windows : case-insensitive)
                key = f.resolve()
                if key not in seen:
                    seen.add(key)
                    files.append(f)
        return sorted(files)
    else:
        return sorted(dir_path.glob("*"))


def check_format_support() -> dict:
    """
    Check which image formats are supported with current dependencies.
    
    Returns:
        Dictionary with format status information
    """
    status = {
        "standard_formats": {
            "JPG/JPEG": True,
            "PNG": True,
            "TIF/TIFF": True,
            "BMP": True,
        },
        "raw_formats": {}
    }
    
    try:
        import rawpy
        status["raw_formats"]["CR3"] = True
        status["raw_formats"]["CR2"] = True
        status["raw_formats"]["NEF"] = True
        status["raw_formats"]["ARW"] = True
        status["raw_formats"]["DNG"] = True
        status["rawpy_installed"] = True
    except ImportError:
        status["raw_formats"]["CR3"] = False
        status["raw_formats"]["CR2"] = False
        status["raw_formats"]["NEF"] = False
        status["raw_formats"]["ARW"] = False
        status["raw_formats"]["DNG"] = False
        status["rawpy_installed"] = False
    
    return status


def print_format_support():
    """Print a nice table of supported image formats."""
    status = check_format_support()
    
    print("\n" + "="*70)
    print("📸 IMAGE FORMAT SUPPORT STATUS")
    print("="*70)
    
    print("\n✅ STANDARD FORMATS (Always Available):")
    for fmt, supported in status["standard_formats"].items():
        symbol = "✅" if supported else "❌"
        print(f"   {symbol} {fmt}")
    
    print("\n🔧 RAW FORMATS (Requires rawpy):")
    for fmt, supported in status["raw_formats"].items():
        symbol = "✅" if supported else "❌"
        print(f"   {symbol} {fmt}")
    
    if not status["rawpy_installed"]:
        print("\n⚠️  TO ENABLE RAW FORMAT SUPPORT:")
        print("   pip install rawpy")
        print("\n   On Linux, also run:")
        print("   sudo apt-get install libcaca-dev")
    else:
        print("\n✨ All formats are supported!")
    
    print("="*70 + "\n")
