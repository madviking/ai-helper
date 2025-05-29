# tracker.py

from typing import Any, Dict, List, Optional
from datetime import datetime
from .adapters.base import StorageAdapter
from .adapters.file_adapter_simple import FileStorageAdapter
from .adapters.database_adapter import DatabaseStorageAdapter
from .models import UsageRecord, ToolUsageRecord, FillPercentageRecord
from .aggregator import UsageAggregator
from .formatting import format_usage_data, print_usage_report
from py_models.base import LLMReport
from pydantic_ai.usage import Usage


class UsageTracker:
    """
    Simplified usage tracker with pluggable storage adapters.
    Supports both file-based and database storage through adapter pattern.
    """
    
    def __init__(self, storage_adapter: Optional[StorageAdapter] = None, base_path: Optional[str] = None):
        if storage_adapter is None:
            # Default to file storage for backward compatibility
            self.adapter = FileStorageAdapter(base_path)
        else:
            self.adapter = storage_adapter
    
    @classmethod
    def create_file_storage(cls, base_path: Optional[str] = None) -> 'UsageTracker':
        """Create UsageTracker with file storage adapter."""
        return cls(FileStorageAdapter(base_path))
    
    @classmethod
    def create_database_storage(cls, database_url: str = "sqlite:///usage.db") -> 'UsageTracker':
        """Create UsageTracker with database storage adapter."""
        return cls(DatabaseStorageAdapter(database_url))
    
    def add_usage(self, usage_report: LLMReport, model_name: str, service: str,
                  pydantic_model_name: Optional[str] = None,
                  tool_names_called: Optional[List[str]] = None):
        """Add usage data to storage."""
        current_date = datetime.now()
        current_day = current_date.strftime("%Y-%m-%d")
        current_month = current_date.strftime("%Y-%m")
        
        actual_pydantic_model_name = pydantic_model_name or "N/A"
        
        # Extract usage data
        llm_usage_obj = usage_report.usage or Usage()
        input_tokens = llm_usage_obj.request_tokens or 0
        output_tokens = llm_usage_obj.response_tokens or 0
        total_tokens = llm_usage_obj.total_tokens or 0
        requests = llm_usage_obj.requests or 1
        cost = usage_report.cost
        
        # Create and add usage record
        usage_record = UsageRecord(
            month=current_month,
            day=current_day,
            model=model_name,
            service=service,
            pydantic_model_name=actual_pydantic_model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            requests=requests,
            cost=cost
        )
        self.adapter.add_usage_record(usage_record)
        
        # Add fill percentage records if available
        if pydantic_model_name and usage_report.fill_percentage is not None and usage_report.fill_percentage >= 0:
            # Pydantic model fill percentage
            pydantic_fill_record = FillPercentageRecord(
                model_type="pydantic",
                model_name=actual_pydantic_model_name,
                fill_percentage=usage_report.fill_percentage
            )
            self.adapter.add_fill_percentage_record(pydantic_fill_record)
            
            # LLM model fill percentage
            llm_fill_record = FillPercentageRecord(
                model_type="llm",
                model_name=model_name,
                fill_percentage=usage_report.fill_percentage
            )
            self.adapter.add_fill_percentage_record(llm_fill_record)
        
        # Add tool usage records
        if tool_names_called:
            for tool_name in tool_names_called:
                tool_record = ToolUsageRecord(
                    month=current_month,
                    day=current_day,
                    tool_name=tool_name,
                    calls=1
                )
                self.adapter.add_tool_usage_record(tool_record)
    
    def get_usage_today(self) -> float:
        """Get total cost for today."""
        return self.adapter.get_usage_today()
    
    def get_usage_this_month(self) -> float:
        """Get total cost for this month."""
        return self.adapter.get_usage_this_month()
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary for reporting."""
        raw_data = self.adapter.get_usage_summary()
        
        # Add basic stats
        summary = {
            'usage_today': raw_data['usage_today'],
            'usage_this_month': raw_data['usage_this_month']
        }
        
        # Add aggregated data
        aggregated = UsageAggregator.aggregate_usage_summary(
            raw_data['usage_records'],
            raw_data['tool_usage_records'], 
            raw_data['fill_percentage_records']
        )
        summary.update(aggregated)
        
        return summary
    
    def print_usage_report(self):
        """Print formatted usage report."""
        summary = self.get_usage_summary()
        print_usage_report(summary)
    
    # Legacy property for backward compatibility
    @property
    def config(self):
        """Legacy property for backward compatibility."""
        return self.get_usage_summary()