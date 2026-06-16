def get_resource_recommendation(severity: int) -> dict:
    """
    Returns rule-based resource allocation based on predicted severity bucket.
    Severity: 0 (Low), 1 (Medium), 2 (High)
    """
    if severity == 0:
        return {
            "Level": "Low",
            "Officers": 1,
            "Tow Trucks": 0,
            "Barricades": 0,
            "Action": "Monitor and clear minor obstruction. Signage only."
        }
    elif severity == 1:
        return {
            "Level": "Medium",
            "Officers": 2,
            "Tow Trucks": 1,
            "Barricades": 1,
            "Action": "Deploy 1 tow truck to clear carriageway. Setup 1 temporary diversion point."
        }
    else:
        return {
            "Level": "High",
            "Officers": 4,
            "Tow Trucks": 2,
            "Barricades": 3,
            "Action": "Full response required. Setup diversion signage, alert public channels."
        }
