from typing import TypedDict


class ResourcePlan(TypedDict):
    Level: str
    Officers: int
    Tow_Trucks: int
    Barricades: int
    Action: str


# Lookup table: severity bucket → resource plan.
# Framed as a heuristic starting point — could become a learned policy once
# real dispatch/deployment logs are available (none exist in the current schema).

_PLANS: dict[int, ResourcePlan] = {
    0: ResourcePlan(
        Level="Low",
        Officers=1,
        Tow_Trucks=0,
        Barricades=0,
        Action="Monitor and clear minor obstruction. Signage only.",
    ),
    1: ResourcePlan(
        Level="Medium",
        Officers=2,
        Tow_Trucks=1,
        Barricades=1,
        Action="Deploy 1 tow truck to clear carriageway. Setup 1 temporary diversion point.",
    ),
    2: ResourcePlan(
        Level="High",
        Officers=4,
        Tow_Trucks=2,
        Barricades=3,
        Action="Full response required. Setup diversion signage, alert public channels.",
    ),
}


def get_resource_recommendation(severity: int) -> ResourcePlan:
    """
    Return a rule-based resource allocation plan for a predicted severity bucket.

    Parameters
    ----------
    severity : int
        0 = Low, 1 = Medium, 2 = High.  Values outside this range are clamped
        to the nearest valid bucket rather than silently returning a wrong plan.

    Returns
    -------
    ResourcePlan
        Dict-like object with keys: Level, Officers, Tow_Trucks, Barricades, Action.
    """
    clamped = max(0, min(severity, 2))
    return _plans_compat(_PLANS[clamped])


def _plans_compat(plan: ResourcePlan) -> dict:
    """
    Return a plain dict with the legacy 'Tow Trucks' key (space, not underscore)
    so existing callers in app.py that use rec['Tow Trucks'] keep working.
    """
    d = dict(plan)
    d['Tow Trucks'] = d.pop('Tow_Trucks')
    return d
