from src.ai_copilot.llm_client import build_workforce_context


def test_build_workforce_context_includes_metrics():
    workforce_summary = {
        "total_employees": 1470,
        "attrition_rate": 16.04,
    }

    forecast_summary = {
        "forecast_horizon_months": 12,
        "projected_headcount_change": 10,
    }

    context = build_workforce_context(
        workforce_summary,
        forecast_summary,
    )

    assert "CURRENT WORKFORCE METRICS" in context
    assert "Total Employees: 1470" in context
    assert "Attrition Rate: 16.04" in context
    assert "Forecast Horizon Months: 12" in context
    assert "synthetic" in context.lower()


def test_build_workforce_context_handles_empty_values():
    context = build_workforce_context({}, {})

    assert "CURRENT WORKFORCE METRICS" in context
    assert "WORKFORCE FORECAST SUMMARY" in context
    assert "IMPORTANT LIMITATION" in context