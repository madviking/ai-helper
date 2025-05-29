# file_adapter.py

import os
import json
import re
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict
from .base import StorageAdapter
from ..models import UsageRecord, ToolUsageRecord, FillPercentageRecord, FillPercentageStats


class FileStorageAdapter(StorageAdapter):
    """File-based storage adapter using JSON files."""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            self.config_path = os.path.join(os.path.dirname(__file__), '../../../logs/usage.json')
        else:
            self.config_path = os.path.join(base_path, 'logs/usage.json')
        
        if not os.path.exists(self.config_path):
            self._create_empty_file()
        
    def _create_empty_file(self):
        """Create empty usage file with proper structure."""
        empty_data = {
            "usage_records": [],
            "tool_usage_records": [],
            "fill_percentage_records": []
        }
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(empty_data, f, indent=4)
    
    def _load_data(self) -> Dict[str, List]:
        """Load data from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                # Ensure all required keys exist
                if 'usage_records' not in data:
                    data['usage_records'] = []
                if 'tool_usage_records' not in data:
                    data['tool_usage_records'] = []
                if 'fill_percentage_records' not in data:
                    data['fill_percentage_records'] = []
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            self._create_empty_file()
            return {"usage_records": [], "tool_usage_records": [], "fill_percentage_records": []}
    
    def _save_data(self, data: Dict[str, List]):
        """Save data to JSON file."""
        json_str = json.dumps(data, indent=4, default=str)
        
        # Handle scientific notation
        def replace_scientific(match):
            num = float(match.group(0))
            return f"{num:.8f}".rstrip('0').rstrip('.') if '.' in f"{num:.8f}" else f"{num:.8f}"
        
        json_str = re.sub(r'\d+\.?\d*e[+-]?\d+', replace_scientific, json_str, flags=re.IGNORECASE)
        
        with open(self.config_path, 'w') as f:
            f.write(json_str)
    
    def add_usage_record(self, record: UsageRecord) -> None:
        """Add or update a usage record."""
        data = self._load_data()
        
        # Find existing record to update
        existing_record = None
        for i, existing in enumerate(data['usage_records']):
            if (existing.get('day') == record.day and
                existing.get('model') == record.model and
                existing.get('service') == record.service and
                existing.get('pydantic_model_name') == record.pydantic_model_name):
                existing_record = i
                break
        
        record_dict = record.model_dump()
        if existing_record is not None:
            # Update existing record
            existing = data['usage_records'][existing_record]
            existing['input_tokens'] += record.input_tokens
            existing['output_tokens'] += record.output_tokens
            existing['total_tokens'] += record.total_tokens
            existing['requests'] += record.requests
            existing['cost'] += record.cost
        else:
            # Add new record
            data['usage_records'].append(record_dict)
        
        self._save_data(data)
    
    def add_tool_usage_record(self, record: ToolUsageRecord) -> None:
        """Add or update a tool usage record."""
        data = self._load_data()
        
        # Find existing record to update
        existing_record = None
        for i, existing in enumerate(data['tool_usage_records']):
            if (existing.get('day') == record.day and
                existing.get('tool_name') == record.tool_name):
                existing_record = i
                break
        
        record_dict = record.model_dump()
        if existing_record is not None:
            # Update existing record
            data['tool_usage_records'][existing_record]['calls'] += record.calls
        else:
            # Add new record
            data['tool_usage_records'].append(record_dict)
        
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
        """Get comprehensive usage summary for reporting."""
        data = self._load_data()
        summary = {}
        
        # Basic stats
        summary['usage_today'] = self.get_usage_today()
        summary['usage_this_month'] = self.get_usage_this_month()
        
        # Daily usage (already aggregated from storage)
        summary['daily_usage'] = data['usage_records']
        summary['daily_tool_usage'] = data['tool_usage_records']
        
        # Monthly aggregations
        monthly_llm_summary = defaultdict(lambda: {'requests': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'cost': 0.0})
        for record in data['usage_records']:
            month = record.get('month')
            monthly_llm_summary[month]['requests'] += record.get('requests', 0)
            monthly_llm_summary[month]['input_tokens'] += record.get('input_tokens', 0)
            monthly_llm_summary[month]['output_tokens'] += record.get('output_tokens', 0)
            monthly_llm_summary[month]['total_tokens'] += record.get('total_tokens', 0)
            monthly_llm_summary[month]['cost'] += record.get('cost', 0.0)
        summary['monthly_llm_summary'] = dict(monthly_llm_summary)
        
        monthly_tool_summary = defaultdict(lambda: {'total_calls': 0})
        for record in data['tool_usage_records']:
            month = record.get('month')
            monthly_tool_summary[month]['total_calls'] += record.get('calls', 0)
        summary['monthly_tool_summary'] = dict(monthly_tool_summary)
        
        # All-time aggregations
        by_model = defaultdict(lambda: {'requests': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'cost': 0.0})
        by_service = defaultdict(lambda: {'requests': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'cost': 0.0})
        usage_by_pydantic_model = defaultdict(lambda: {'requests': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'cost': 0.0})
        
        for record in data['usage_records']:
            model = record.get('model')
            service = record.get('service')
            pydantic_model = record.get('pydantic_model_name')
            
            for target_dict, key in [(by_model, model), (by_service, service)]:
                target_dict[key]['requests'] += record.get('requests', 0)
                target_dict[key]['input_tokens'] += record.get('input_tokens', 0)
                target_dict[key]['output_tokens'] += record.get('output_tokens', 0)
                target_dict[key]['total_tokens'] += record.get('total_tokens', 0)
                target_dict[key]['cost'] += record.get('cost', 0.0)
            
            if pydantic_model != "N/A":
                usage_by_pydantic_model[pydantic_model]['requests'] += record.get('requests', 0)
                usage_by_pydantic_model[pydantic_model]['input_tokens'] += record.get('input_tokens', 0)
                usage_by_pydantic_model[pydantic_model]['output_tokens'] += record.get('output_tokens', 0)
                usage_by_pydantic_model[pydantic_model]['total_tokens'] += record.get('total_tokens', 0)
                usage_by_pydantic_model[pydantic_model]['cost'] += record.get('cost', 0.0)
        
        summary['by_model'] = dict(by_model)
        summary['by_service'] = dict(by_service)
        summary['usage_by_pydantic_model'] = dict(usage_by_pydantic_model)
        
        # Tool usage aggregation
        by_tool = defaultdict(lambda: {'calls': 0})
        for record in data['tool_usage_records']:
            tool_name = record.get('tool_name')
            by_tool[tool_name]['calls'] += record.get('calls', 0)
        summary['by_tool'] = dict(by_tool)
        
        # Fill percentage stats
        fill_by_pydantic = defaultdict(lambda: FillPercentageStats())
        fill_by_llm = defaultdict(lambda: FillPercentageStats())
        
        for record in data['fill_percentage_records']:
            model_type = record.get('model_type')
            model_name = record.get('model_name')
            fill_pct = record.get('fill_percentage', 0.0)
            
            target_dict = fill_by_pydantic if model_type == 'pydantic' else fill_by_llm
            stats = target_dict[model_name]
            stats.count += 1
            stats.sum_total += fill_pct
            stats.average = stats.sum_total / stats.count
        
        summary['fill_percentage_by_pydantic_model'] = dict(fill_by_pydantic)
        summary['fill_percentage_by_llm_model'] = dict(fill_by_llm)
        
        summary['total_llm_requests'] = sum(stats['requests'] for stats in by_model.values())
        summary['total_tool_calls'] = sum(stats['calls'] for stats in by_tool.values())
        
        return summary