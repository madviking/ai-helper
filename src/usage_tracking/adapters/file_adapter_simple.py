# file_adapter_simple.py

import os
import json
from datetime import datetime
from typing import Dict, Any, List
from .base import StorageAdapter
from ..models import UsageRecord, ToolUsageRecord, FillPercentageRecord


class FileStorageAdapter(StorageAdapter):
    """Simple file-based storage adapter using JSON files."""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            self.config_path = os.path.join(os.path.dirname(__file__), '../../../logs/usage.json')
        else:
            self.config_path = os.path.join(base_path, 'logs/usage.json')
        
        if not os.path.exists(self.config_path):
            self._create_empty_file()
    
    def _create_empty_file(self):
        """Create empty usage file."""
        empty_data = {
            "usage_records": [],
            "tool_usage_records": [],
            "fill_percentage_records": []
        }
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(empty_data, f, indent=4)
    
    def _load_data(self) -> Dict[str, List]:
        """Load raw data from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                # Ensure required keys exist
                for key in ["usage_records", "tool_usage_records", "fill_percentage_records"]:
                    if key not in data:
                        data[key] = []
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            self._create_empty_file()
            return {"usage_records": [], "tool_usage_records": [], "fill_percentage_records": []}
    
    def _save_data(self, data: Dict[str, List]):
        """Save data to JSON file."""
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=4, default=str)
    
    def add_usage_record(self, record: UsageRecord) -> None:
        """Add or update a usage record."""
        data = self._load_data()
        
        # Find existing record with same key
        for existing in data['usage_records']:
            if (existing.get('day') == record.day and
                existing.get('model') == record.model and
                existing.get('service') == record.service and
                existing.get('pydantic_model_name') == record.pydantic_model_name):
                # Update existing
                existing['input_tokens'] += record.input_tokens
                existing['output_tokens'] += record.output_tokens
                existing['total_tokens'] += record.total_tokens
                existing['requests'] += record.requests
                existing['cost'] += record.cost
                self._save_data(data)
                return
        
        # Add new record
        data['usage_records'].append(record.model_dump())
        self._save_data(data)
    
    def add_tool_usage_record(self, record: ToolUsageRecord) -> None:
        """Add or update a tool usage record."""
        data = self._load_data()
        
        # Find existing record
        for existing in data['tool_usage_records']:
            if (existing.get('day') == record.day and
                existing.get('tool_name') == record.tool_name):
                existing['calls'] += record.calls
                self._save_data(data)
                return
        
        # Add new record
        data['tool_usage_records'].append(record.model_dump())
        self._save_data(data)
    
    def add_fill_percentage_record(self, record: FillPercentageRecord) -> None:
        """Add a fill percentage record."""
        data = self._load_data()
        data['fill_percentage_records'].append(record.model_dump())
        self._save_data(data)
    
    def get_usage_today(self) -> float:
        """Get total cost for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        data = self._load_data()
        return sum(record.get('cost', 0) for record in data['usage_records'] 
                  if record.get('day') == today)
    
    def get_usage_this_month(self) -> float:
        """Get total cost for this month."""
        current_month = datetime.now().strftime("%Y-%m")
        data = self._load_data()
        return sum(record.get('cost', 0) for record in data['usage_records'] 
                  if record.get('month') == current_month)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary - MINIMAL, just return the raw records."""
        data = self._load_data()
        return {
            'usage_today': self.get_usage_today(),
            'usage_this_month': self.get_usage_this_month(),
            'usage_records': data['usage_records'],
            'tool_usage_records': data['tool_usage_records'],
            'fill_percentage_records': data['fill_percentage_records']
        }