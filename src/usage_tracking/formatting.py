# formatting.py

import os
import json
from typing import Dict, Any
from tabulate import tabulate


def format_usage_data(data: Dict[str, Any]) -> str:
    """
    Standalone function to format any usage JSON data into nicely formatted tables.
    Can be used independently of the UsageTracker class.
    """
    output = []

    # Overall Summary section (Today, This Month Costs)
    if 'usage_today' in data or 'usage_this_month' in data:
        output.append("=" * 60)
        output.append("OVERALL USAGE SUMMARY (COSTS)")
        output.append("=" * 60)
        summary_data = []
        if 'usage_today' in data:
            summary_data.append(['Today (LLM Cost)', f"${data['usage_today']:.6f}"])
        if 'usage_this_month' in data:
            summary_data.append(['This Month (LLM Cost)', f"${data['usage_this_month']:.6f}"])
        if summary_data:
            output.append(tabulate(summary_data, headers=['Period', 'Value'], tablefmt='grid'))
            output.append("")

    # Monthly LLM Usage Summary
    if 'monthly_llm_summary' in data and data['monthly_llm_summary']:
        output.append("MONTHLY LLM USAGE SUMMARY")
        output.append("-" * 40)
        monthly_llm_data = []
        sorted_monthly_models = sorted(data['monthly_llm_summary'].items(), 
                                     key=lambda item: item[1].get('total_tokens', 0), reverse=True)
        for month, stats in sorted_monthly_models:
            monthly_llm_data.append([
                month,
                stats.get('requests', 0),
                stats.get('input_tokens', 0),
                stats.get('output_tokens', 0),
                stats.get('total_tokens', 0),
                f"${stats.get('cost', 0):.6f}"
            ])
        headers_monthly_llm = ['Month', 'Requests', 'Input Tokens', 'Output Tokens', 'Total Tokens', 'Cost']
        output.append(tabulate(monthly_llm_data, headers=headers_monthly_llm, tablefmt='grid'))
        output.append("")

    # Monthly Tool Usage Summary
    if 'monthly_tool_summary' in data and data['monthly_tool_summary']:
        output.append("MONTHLY TOOL USAGE SUMMARY")
        output.append("-" * 40)
        monthly_tool_data = []
        sorted_months = sorted(data['monthly_tool_summary'].keys())
        for month in sorted_months:
            stats = data['monthly_tool_summary'][month]
            monthly_tool_data.append([
                month,
                stats.get('total_calls', 0)
            ])
        headers_monthly_tool = ['Month', 'Total Tool Calls']
        output.append(tabulate(monthly_tool_data, headers=headers_monthly_tool, tablefmt='grid'))
        output.append("")

    # Daily LLM usage table
    if 'daily_usage' in data and data['daily_usage']:
        output.append("DAILY LLM USAGE BREAKDOWN")
        output.append("-" * 40)
        daily_data = []
        sorted_daily_usage = sorted(data['daily_usage'], 
                                  key=lambda item: item.get('total_tokens', 0), reverse=True)
        for item in sorted_daily_usage:
            daily_data.append([
                item.get('day', 'N/A'),
                item.get('model', 'N/A'),
                item.get('service', 'N/A'),
                item.get('pydantic_model_name', 'N/A'),
                item.get('requests', 0),
                item.get('input_tokens', 0),
                item.get('output_tokens', 0),
                item.get('total_tokens', 0),
                f"${item.get('cost', 0):.6f}"
            ])
        headers_daily_llm = ['Date', 'LLM Model', 'Service', 'Pydantic Model', 'Requests', 
                           'Input Tokens', 'Output Tokens', 'Total Tokens', 'Cost']
        output.append(tabulate(daily_data, headers=headers_daily_llm, tablefmt='grid'))
        output.append("")

    # Daily tool usage table
    if 'daily_tool_usage' in data and data['daily_tool_usage']:
        output.append("DAILY TOOL USAGE BREAKDOWN")
        output.append("-" * 40)
        tool_daily_data = []
        for item in data['daily_tool_usage']:
            tool_daily_data.append([
                item.get('day', 'N/A'),
                item.get('tool_name', 'N/A'),
                item.get('calls', 0)
            ])
        tool_headers = ['Date', 'Tool Name', 'Calls']
        output.append(tabulate(tool_daily_data, headers=tool_headers, tablefmt='grid'))
        output.append("")

    # Fill percentage stats
    if 'fill_percentage_by_pydantic_model' in data and data['fill_percentage_by_pydantic_model']:
        output.append("FILL PERCENTAGE BY PYDANTIC MODEL")
        output.append("-" * 40)
        pydantic_data = []
        for model_name, stats_obj in data['fill_percentage_by_pydantic_model'].items():
            pydantic_data.append([
                model_name,
                f"{stats_obj.average:.2f}%" if hasattr(stats_obj, 'average') else "0.00%",
                stats_obj.count if hasattr(stats_obj, 'count') else 0
            ])
        output.append(tabulate(pydantic_data, headers=['Pydantic Model', 'Avg Fill %', 'Samples'], tablefmt='grid'))
        output.append("")

    if 'fill_percentage_by_llm_model' in data and data['fill_percentage_by_llm_model']:
        output.append("FILL PERCENTAGE BY LLM MODEL")
        output.append("-" * 40)
        llm_data = []
        for model_name, stats_obj in data['fill_percentage_by_llm_model'].items():
            llm_data.append([
                model_name,
                f"{stats_obj.average:.2f}%" if hasattr(stats_obj, 'average') else "0.00%",
                stats_obj.count if hasattr(stats_obj, 'count') else 0
            ])
        output.append(tabulate(llm_data, headers=['LLM Model', 'Avg Fill %', 'Samples'], tablefmt='grid'))
        output.append("")

    # All time LLM Usage by LLM Model
    if 'by_model' in data:
        output.append("LLM USAGE BY LLM MODEL (ALL TIME)")
        output.append("-" * 40)
        model_data = []
        sorted_models = sorted(data['by_model'].items(), 
                             key=lambda item: item[1].get('total_tokens', 0), reverse=True)
        for model, stats in sorted_models:
            fill_stats = data.get('fill_percentage_by_llm_model', {}).get(model)
            avg_fill_percentage = f"{fill_stats.average:.2f}%" if fill_stats and hasattr(fill_stats, 'average') else "N/A"

            model_data.append([
                model,
                stats.get('requests', 0),
                stats.get('input_tokens', 0),
                stats.get('output_tokens', 0),
                stats.get('total_tokens', 0),
                f"${stats.get('cost', 0):.6f}",
                avg_fill_percentage
            ])
        output.append(
            tabulate(model_data,
                     headers=['LLM Model', 'Requests', 'Input Tokens', 'Output Tokens', 'Total Tokens', 'Cost', 'Avg Fill %'],
                     tablefmt='grid'))
        output.append("")

    # All time LLM Usage by Pydantic Model
    if 'usage_by_pydantic_model' in data and data['usage_by_pydantic_model']:
        output.append("LLM USAGE BY PYDANTIC MODEL (ALL TIME)")
        output.append("-" * 40)
        pydantic_usage_data = []
        for p_model_name, stats in data['usage_by_pydantic_model'].items():
            pydantic_usage_data.append([
                p_model_name,
                stats.get('requests', 0),
                stats.get('input_tokens', 0),
                stats.get('output_tokens', 0),
                stats.get('total_tokens', 0),
                f"${stats.get('cost', 0):.6f}"
            ])
        output.append(
            tabulate(pydantic_usage_data,
                     headers=['Pydantic Model', 'Requests', 'Input Tokens', 'Output Tokens', 'Total Tokens', 'Cost'],
                     tablefmt='grid'))
        output.append("")

    # Service usage summary
    if 'by_service' in data:
        output.append("LLM USAGE BY SERVICE (ALL TIME)")
        output.append("-" * 40)
        service_data = []
        for service, stats in data['by_service'].items():
            service_data.append([
                service,
                stats.get('requests', 0),
                stats.get('input_tokens', 0),
                stats.get('output_tokens', 0),
                stats.get('total_tokens', 0),
                f"${stats.get('cost', 0):.6f}"
            ])
        output.append(tabulate(service_data,
                               headers=['Service', 'Requests', 'Input Tokens', 'Output Tokens', 'Total Tokens', 'Cost'],
                               tablefmt='grid'))
        output.append("")

    # Tool usage summary
    if 'by_tool' in data:
        output.append("TOOL USAGE BY NAME (ALL TIME)")
        output.append("-" * 40)
        tool_summary_data = []
        for tool_name, stats in data['by_tool'].items():
            tool_summary_data.append([
                tool_name,
                stats.get('calls', 0)
            ])
        output.append(tabulate(tool_summary_data, headers=['Tool Name', 'Total Calls'], tablefmt='grid'))
        output.append("")

    return "\n".join(output)


def print_usage_report(data: Dict[str, Any]):
    """Print formatted usage report."""
    print(format_usage_data(data))


def format_usage_from_file(file_path: str) -> str:
    """Load usage data from file and format it."""
    if not os.path.exists(file_path):
        print(f"Warning: Usage file {file_path} not found. Displaying empty report structure.")
        return format_usage_data({})
    
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. File might be corrupted or empty.")
            return format_usage_data({})
    
    return format_usage_data(data)