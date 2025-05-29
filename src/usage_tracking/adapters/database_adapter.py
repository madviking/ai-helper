# database_adapter.py

from datetime import datetime
from typing import Dict, Any, Optional
from collections import defaultdict
from sqlmodel import SQLModel, create_engine, Session, select, func
from .base import StorageAdapter
from ..models import UsageRecord, ToolUsageRecord, FillPercentageRecord, FillPercentageStats


class DatabaseStorageAdapter(StorageAdapter):
    """Database storage adapter using SQLModel."""
    
    def __init__(self, database_url: str = "sqlite:///usage.db"):
        self.engine = create_engine(database_url)
        SQLModel.metadata.create_all(self.engine)
    
    def add_usage_record(self, record: UsageRecord) -> None:
        """Add or update a usage record."""
        with Session(self.engine) as session:
            # Try to find existing record
            statement = select(UsageRecord).where(
                UsageRecord.day == record.day,
                UsageRecord.model == record.model,
                UsageRecord.service == record.service,
                UsageRecord.pydantic_model_name == record.pydantic_model_name
            )
            existing = session.exec(statement).first()
            
            if existing:
                # Update existing record
                existing.input_tokens += record.input_tokens
                existing.output_tokens += record.output_tokens
                existing.total_tokens += record.total_tokens
                existing.requests += record.requests
                existing.cost += record.cost
            else:
                # Add new record
                session.add(record)
            
            session.commit()
    
    def add_tool_usage_record(self, record: ToolUsageRecord) -> None:
        """Add or update a tool usage record."""
        with Session(self.engine) as session:
            # Try to find existing record
            statement = select(ToolUsageRecord).where(
                ToolUsageRecord.day == record.day,
                ToolUsageRecord.tool_name == record.tool_name
            )
            existing = session.exec(statement).first()
            
            if existing:
                # Update existing record
                existing.calls += record.calls
            else:
                # Add new record
                session.add(record)
            
            session.commit()
    
    def add_fill_percentage_record(self, record: FillPercentageRecord) -> None:
        """Add a fill percentage record."""
        with Session(self.engine) as session:
            session.add(record)
            session.commit()
    
    def get_usage_today(self) -> float:
        """Get total cost for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        with Session(self.engine) as session:
            statement = select(func.sum(UsageRecord.cost)).where(UsageRecord.day == today)
            result = session.exec(statement).first()
            return float(result or 0.0)
    
    def get_usage_this_month(self) -> float:
        """Get total cost for this month."""
        current_month = datetime.now().strftime("%Y-%m")
        with Session(self.engine) as session:
            statement = select(func.sum(UsageRecord.cost)).where(UsageRecord.month == current_month)
            result = session.exec(statement).first()
            return float(result or 0.0)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary - return raw records only."""
        with Session(self.engine) as session:
            # Get all records as simple dicts
            usage_records = [record.model_dump() for record in session.exec(select(UsageRecord)).all()]
            tool_records = [record.model_dump() for record in session.exec(select(ToolUsageRecord)).all()] 
            fill_records = [record.model_dump() for record in session.exec(select(FillPercentageRecord)).all()]
            
            return {
                'usage_today': self.get_usage_today(),
                'usage_this_month': self.get_usage_this_month(),
                'usage_records': usage_records,
                'tool_usage_records': tool_records,
                'fill_percentage_records': fill_records
            }