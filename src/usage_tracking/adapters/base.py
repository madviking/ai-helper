# base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ..models import UsageRecord, ToolUsageRecord, FillPercentageRecord


class StorageAdapter(ABC):
    """Abstract base class for usage data storage adapters."""
    
    @abstractmethod
    def add_usage_record(self, record: UsageRecord) -> None:
        """Add or update a usage record."""
        pass
    
    @abstractmethod
    def add_tool_usage_record(self, record: ToolUsageRecord) -> None:
        """Add or update a tool usage record."""
        pass
    
    @abstractmethod
    def add_fill_percentage_record(self, record: FillPercentageRecord) -> None:
        """Add a fill percentage record."""
        pass
    
    @abstractmethod
    def get_usage_today(self) -> float:
        """Get total cost for today."""
        pass
    
    @abstractmethod
    def get_usage_this_month(self) -> float:
        """Get total cost for this month."""
        pass
    
    @abstractmethod
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary for reporting."""
        pass