# aggregator.py

from typing import Dict, Any, List
from collections import defaultdict
from .models import FillPercentageStats


class UsageAggregator:
    """Pure aggregation logic using only Pydantic models."""
    
    @staticmethod
    def aggregate_usage_summary(usage_records: List[Dict], 
                               tool_records: List[Dict], 
                               fill_records: List[Dict]) -> Dict[str, Any]:
        """Aggregate raw records into summary format."""
        
        # Basic aggregations using simple loops over Pydantic model data
        by_model = defaultdict(lambda: {'requests': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'cost': 0.0})
        by_service = defaultdict(lambda: {'requests': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'cost': 0.0})
        by_pydantic_model = defaultdict(lambda: {'requests': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'cost': 0.0})
        monthly_summary = defaultdict(lambda: {'requests': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'cost': 0.0})
        
        # Aggregate usage records
        for record in usage_records:
            model = record.get('model', '')
            service = record.get('service', '')
            pydantic_model = record.get('pydantic_model_name', 'N/A')
            month = record.get('month', '')
            
            requests = record.get('requests', 0)
            input_tokens = record.get('input_tokens', 0) 
            output_tokens = record.get('output_tokens', 0)
            total_tokens = record.get('total_tokens', 0)
            cost = record.get('cost', 0.0)
            
            # Update all aggregations
            for target_dict, key in [(by_model, model), (by_service, service), (monthly_summary, month)]:
                target_dict[key]['requests'] += requests
                target_dict[key]['input_tokens'] += input_tokens
                target_dict[key]['output_tokens'] += output_tokens
                target_dict[key]['total_tokens'] += total_tokens
                target_dict[key]['cost'] += cost
            
            if pydantic_model != 'N/A':
                by_pydantic_model[pydantic_model]['requests'] += requests
                by_pydantic_model[pydantic_model]['input_tokens'] += input_tokens
                by_pydantic_model[pydantic_model]['output_tokens'] += output_tokens
                by_pydantic_model[pydantic_model]['total_tokens'] += total_tokens
                by_pydantic_model[pydantic_model]['cost'] += cost
        
        # Aggregate tool records
        by_tool = defaultdict(lambda: {'calls': 0})
        monthly_tool_summary = defaultdict(lambda: {'total_calls': 0})
        
        for record in tool_records:
            tool_name = record.get('tool_name', '')
            month = record.get('month', '')
            calls = record.get('calls', 0)
            
            by_tool[tool_name]['calls'] += calls
            monthly_tool_summary[month]['total_calls'] += calls
        
        # Aggregate fill percentage records
        fill_by_pydantic = defaultdict(lambda: FillPercentageStats())
        fill_by_llm = defaultdict(lambda: FillPercentageStats())
        
        for record in fill_records:
            model_type = record.get('model_type', '')
            model_name = record.get('model_name', '')
            fill_pct = record.get('fill_percentage', 0.0)
            
            target_dict = fill_by_pydantic if model_type == 'pydantic' else fill_by_llm
            stats = target_dict[model_name]
            stats.count += 1
            stats.sum_total += fill_pct
            stats.average = stats.sum_total / stats.count
        
        return {
            'daily_usage': usage_records,
            'daily_tool_usage': tool_records,
            'monthly_llm_summary': dict(monthly_summary),
            'monthly_tool_summary': dict(monthly_tool_summary),
            'by_model': dict(by_model),
            'by_service': dict(by_service),
            'usage_by_pydantic_model': dict(by_pydantic_model),
            'by_tool': dict(by_tool),
            'fill_percentage_by_pydantic_model': dict(fill_by_pydantic),
            'fill_percentage_by_llm_model': dict(fill_by_llm),
            'total_llm_requests': sum(stats['requests'] for stats in by_model.values()),
            'total_tool_calls': sum(stats['calls'] for stats in by_tool.values())
        }