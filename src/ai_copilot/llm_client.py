from __future__ import annotations


from typing import Any



def build_workforce_context(
    workforce_summary: dict[str, Any],
    forecast_summary: dict[str, Any],
) -> str:
    workforce_lines = []

    for key, value in workforce_summary.items():
        label = key.replace("_", " ").title()
        workforce_lines.append(f"{label}: {value}")

    forecast_lines = []

    for key, value in forecast_summary.items():
        label = key.replace("_", " ").title()
        forecast_lines.append(f"{label}: {value}")

    context = "CURRENT WORKFORCE METRICS:\n"
    context += "\n".join(workforce_lines)
    context += "\n\nWORKFORCE FORECAST SUMMARY:\n"
    context += "\n".join(forecast_lines)
    context += "\n\nIMPORTANT LIMITATION:\n"
    context += "Forecast outputs are synthetic planning demonstrations based on "
    context += "available workforce history. They should be reviewed alongside "
    context += "operational HR data before workforce decisions."

    return context



def ask_llm(
    question: str,
    workforce_summary: dict[str, Any],
    forecast_summary: dict[str, Any],
    api_key: str | None = None,
    model_name: str = "local-rule-based",
    temperature: float = 0.0,
) -> str:
    question_lower = question.lower()

    attrition = workforce_summary.get("attrition_rate_percent", "Not available")
    employees = workforce_summary.get("total_employees", "Not available")
    active = workforce_summary.get("active_employees", "Not available")
    salary = workforce_summary.get("average_monthly_salary_usd", "Not available")
    tenure = workforce_summary.get("average_tenure_years", "Not available")
    satisfaction = workforce_summary.get("average_job_satisfaction", "Not available")
    overtime = workforce_summary.get("overtime_percentage", "Not available")
    promotions = workforce_summary.get("recent_promotion_rate_percent", "Not available")

    current_headcount = forecast_summary.get("current_headcount", "Not available")
    projected_headcount = forecast_summary.get(
        "projected_headcount_after_12_months",
        "Not available",
    )
    headcount_change = forecast_summary.get(
        "projected_headcount_change",
        "Not available",
    )
    projected_attrition = forecast_summary.get(
        "projected_monthly_attrition_rate_after_12_months",
        "Not available",
    )

    if any(
        word in question_lower
        for word in ["attrition", "turnover", "leave", "leaving", "resign"]
    ):
        lines = []
        lines.append(f"**Current attrition rate:** {attrition}")
        lines.append(
            f"The workforce has {employees} employees, including "
            f"{active} active employees."
        )
        lines.append("")
        lines.append(
            f"**Forecast:** projected monthly attrition rate after 12 months "
            f"is {projected_attrition}."
        )
        lines.append("")
        lines.append("**Suggested HR actions:**")
        lines.append(
            f"- Review overtime patterns; current overtime share is {overtime}."
        )
        lines.append(
            f"- Review engagement; average job satisfaction is {satisfaction}."
        )
        lines.append(
            f"- Review career progression; recent promotion rate is {promotions}."
        )
        lines.append(
            "- Prioritize manager check-ins and retention conversations for "
            "high-risk groups."
        )

        return "\n".join(lines)

    if any(
        word in question_lower
        for word in [
            "headcount",
            "workforce",
            "hiring",
            "hire",
            "staffing",
            "capacity",
            "change over",
        ]
    ):
        lines = []
        lines.append(f"**Current headcount:** {current_headcount}")
        lines.append(
            f"**Projected headcount after 12 months:** {projected_headcount}"
        )
        lines.append(f"**Projected net change:** {headcount_change}")
        lines.append("")
        lines.append(
            f"The projected monthly attrition rate after 12 months is "
            f"{projected_attrition}."
        )
        lines.append("")
        lines.append("**Planning implication:**")
        lines.append(
            "- If the projected change is negative, review hiring plans, "
            "internal mobility, and retention actions."
        )
        lines.append(
            "- If it is positive, validate whether hiring can support workload "
            "and skill needs."
        )
        lines.append(
            "- Use the Forecast tab to inspect headcount, hires, exits, and "
            "attrition-rate trends."
        )

        return "\n".join(lines)

    if any(
        word in question_lower
        for word in ["salary", "compensation", "pay", "payroll", "income"]
    ):
        lines = []
        lines.append(f"**Average monthly salary:** {salary}")
        lines.append(f"**Current workforce size:** {employees} employees")
        lines.append("")
        lines.append("**Compensation review suggestions:**")
        lines.append(
            "- Compare salary distributions by department, job role, and job level."
        )
        lines.append(
            "- Review internal equity alongside market benchmarks."
        )
        lines.append(
            "- Use the salary predictor as decision support, not as an automatic "
            "pay decision."
        )

        return "\n".join(lines)

    if any(
        word in question_lower
        for word in ["risk", "retention", "concern", "priority", "action"]
    ):
        lines = []
        lines.append("**Main workforce signals to review:**")
        lines.append(f"- Attrition rate: {attrition}")
        lines.append(f"- Overtime share: {overtime}")
        lines.append(f"- Average job satisfaction: {satisfaction}")
        lines.append(f"- Recent promotion rate: {promotions}")
        lines.append(f"- Average tenure: {tenure} years")
        lines.append("")
        lines.append("**Recommended focus areas:**")
        lines.append(
            "- Review employees with high model-estimated attrition probability."
        )
        lines.append(
            "- Investigate departments or roles with elevated attrition."
        )
        lines.append(
            "- Examine workload, engagement, promotion gaps, and compensation "
            "alignment."
        )
        lines.append(
            "- Use the People to review and What-if simulator tabs for follow-up."
        )

        return "\n".join(lines)

    if any(
        word in question_lower
        for word in ["forecast", "future", "next 12 months", "projection"]
    ):
        lines = []
        lines.append("**12-month workforce outlook:**")
        lines.append(f"- Current headcount: {current_headcount}")
        lines.append(f"- Projected headcount: {projected_headcount}")
        lines.append(f"- Projected net change: {headcount_change}")
        lines.append(
            f"- Projected monthly attrition rate: {projected_attrition}"
        )
        lines.append("")
        lines.append("**Interpretation:**")
        lines.append(
            "- A negative headcount change suggests potential retention or "
            "hiring pressure."
        )
        lines.append(
            "- A positive change suggests expected workforce growth."
        )
        lines.append(
            "- Review the Forecast tab before making hiring or budget decisions."
        )

        return "\n".join(lines)

    if any(
        word in question_lower
        for word in [
            "satisfaction",
            "engagement",
            "overtime",
            "promotion",
            "tenure",
        ]
    ):
        lines = []
        lines.append("**Current engagement and workload signals:**")
        lines.append(f"- Average job satisfaction: {satisfaction}")
        lines.append(f"- Overtime share: {overtime}")
        lines.append(f"- Recent promotion rate: {promotions}")
        lines.append(f"- Average tenure: {tenure} years")
        lines.append("")
        lines.append("**Suggested review:**")
        lines.append(
            "- Identify teams with high overtime and lower satisfaction."
        )
        lines.append(
            "- Review promotion and development opportunities for long-tenured "
            "employees."
        )
        lines.append(
            "- Use manager check-ins to understand engagement concerns."
        )

        return "\n".join(lines)

    lines = []
    lines.append(
        "I can help with workforce health, attrition, headcount planning, "
        "compensation, engagement, and the 12-month forecast."
    )
    lines.append("")
    lines.append("**Current snapshot:**")
    lines.append(f"- Employees: {employees}")
    lines.append(f"- Attrition rate: {attrition}")
    lines.append(f"- Average monthly salary: {salary}")
    lines.append(f"- Average tenure: {tenure} years")
    lines.append(f"- Overtime share: {overtime}")
    lines.append("")
    lines.append("**Try asking:**")
    lines.append("- What is the current overall attrition rate?")
    lines.append(
        "- How is the workforce expected to change over the next 12 months?"
    )
    lines.append(
        "- What workforce metrics should HR review before planning hiring?"
    )
    lines.append("- Summarize the main workforce risks.")

    return "\n".join(lines)