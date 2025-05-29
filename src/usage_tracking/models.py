# models.py

from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class UsageRecord(SQLModel, table=True):
    __tablename__ = "usage_records"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    month: str = Field(index=True)
    day: str = Field(index=True)
    model: str = Field(index=True)
    service: str = Field(index=True)
    pydantic_model_name: str = Field(default="N/A", index=True)
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    requests: int = Field(default=0)
    cost: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ToolUsageRecord(SQLModel, table=True):
    __tablename__ = "tool_usage_records"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    month: str = Field(index=True)
    day: str = Field(index=True)
    tool_name: str = Field(index=True)
    calls: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FillPercentageRecord(SQLModel, table=True):
    __tablename__ = "fill_percentage_records"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    model_type: str = Field(index=True)  # "pydantic" or "llm"
    model_name: str = Field(index=True)
    fill_percentage: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FillPercentageStats(SQLModel):
    average: float = Field(default=0.0)
    count: int = Field(default=0)
    sum_total: float = Field(default=0.0)