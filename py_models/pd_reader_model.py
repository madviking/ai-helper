from pydantic import BaseModel
from typing import Optional, Dict, Any

class PDReaderModel(BaseModel):
    content: Optional[str] = None
    num_pages: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    extracted_text: Optional[str] = None
