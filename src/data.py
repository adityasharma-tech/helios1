import numpy as np
import logging
import re
import xml.etree.ElementTree as ET
import cv2

logger = logging.getLogger(__name__)

class ISROReader:
    def __init__(self, img_path: str, xml_path: str):
        self.img_path = img_path
        self.xml_path = xml_path
        self.offset = 0
        self.height = 0
        self.width = 0
        self.mm = None
        self._parse_xml()

    def _parse_xml(self):
        ns = {'isda': 'https://isda.issdc.gov.in/pds4/isda/v1',
              'pds': 'http://pds.nasa.gov/pds4/pds/v1'}
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            # ISRO Array_2D_Image structure
            arr = root.find('.//pds:Array_2D_Image', ns)
            if arr is None:
                # If namespace fails, fallback to simple string matching
                with open(self.xml_path, 'r') as f:
                    xml_str = f.read()
                
                offset_match = re.search(r'<offset unit="byte">(\d+)</offset>', xml_str)
                if offset_match: self.offset = int(offset_match.group(1))
                lines_match = re.search(r'<axis_name>Line</axis_name>\s*<elements>(\d+)</elements>', xml_str)
                samples_match = re.search(r'<axis_name>Sample</axis_name>\s*<elements>(\d+)</elements>', xml_str)
                if lines_match: self.height = int(lines_match.group(1))
                if samples_match: self.width = int(samples_match.group(1))
            else:
                self.offset = int(arr.find('pds:offset', ns).text)
                axes = arr.findall('pds:Axis_Array', ns)
                for axis in axes:
                    name = axis.find('pds:axis_name', ns).text
                    elems = int(axis.find('pds:elements', ns).text)
                    if name == 'Line': self.height = elems
                    elif name == 'Sample': self.width = elems
        except Exception as e:
            logger.error(f"Failed to parse ISRO XML: {e}")

    def __enter__(self):
        self.mm = np.memmap(
            self.img_path, 
            dtype='<u2', # Little-endian unsigned 16-bit
            mode='r', 
            offset=self.offset, 
            shape=(self.height, self.width)
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mm is not None:
            del self.mm

    def extract_patch(self, cx: int, cy: int, size: int) -> np.ndarray:
        half = size // 2
        y1, y2 = max(0, cy - half), min(self.height, cy + half)
        x1, x2 = max(0, cx - half), min(self.width, cx + half)
        patch = np.array(self.mm[y1:y2, x1:x2])
        return patch

class JAXAReader:
    def __init__(self, img_path: str, lbl_path: str):
        self.img_path = img_path
        self.lbl_path = lbl_path
        self.offset = 0
        self.height = 0
        self.width = 0
        self.mm = None
        self._parse_lbl()

    def _parse_lbl(self):
        try:
            with open(self.lbl_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            lines_match = re.search(r'LINES\s*=\s*(\d+)', content)
            samples_match = re.search(r'LINE_SAMPLES\s*=\s*(\d+)', content)
            
            # Record offset (e.g., ^IMAGE = ("...", 1 <BYTES>))
            offset_match = re.search(r'\^IMAGE\s*=\s*\("[^"]+",\s*(\d+)\s*<BYTES>\)', content)
            if offset_match:
                # PDS offsets are 1-based usually
                self.offset = max(0, int(offset_match.group(1)) - 1)
                
            if lines_match: self.height = int(lines_match.group(1))
            if samples_match: self.width = int(samples_match.group(1))
            
        except Exception as e:
            logger.error(f"Failed to parse JAXA LBL: {e}")

    def __enter__(self):
        self.mm = np.memmap(
            self.img_path, 
            dtype='>u2', # Big-endian unsigned 16-bit
            mode='r', 
            offset=self.offset, 
            shape=(self.height, self.width)
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mm is not None:
            del self.mm

    def read_all(self) -> np.ndarray:
        return np.array(self.mm)
        
    def extract_patch(self, cx: int, cy: int, size: int) -> np.ndarray:
        half = size // 2
        y1, y2 = max(0, cy - half), min(self.height, cy + half)
        x1, x2 = max(0, cx - half), min(self.width, cx + half)
        patch = np.array(self.mm[y1:y2, x1:x2])
        return patch

def normalize_16bit_to_8bit(image_16: np.ndarray) -> np.ndarray:
    """Normalize raw 16-bit data to 8-bit using CLAHE for feature matching."""
    valid_mask = image_16 > 0
    if not valid_mask.any():
        return np.zeros_like(image_16, dtype=np.uint8)
        
    # Clip extreme percentiles
    p_min, p_max = np.percentile(image_16[valid_mask], 1), np.percentile(image_16[valid_mask], 99)
    p_max = max(p_max, p_min + 1)
    
    normalized = np.clip((image_16 - p_min) / (p_max - p_min), 0, 1)
    image_8 = (normalized * 255).astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_clahe = clahe.apply(image_8)
    
    return image_clahe
