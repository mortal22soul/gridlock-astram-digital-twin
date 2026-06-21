import datetime

# Hardcoded list of major public holidays in Karnataka/India for 2026
# For a production system, this would query an API or load a comprehensive calendar.
KARNATAKA_HOLIDAYS_2026 = {
    "2026-01-01": "New Year's Day",
    "2026-01-14": "Makar Sankranti",
    "2026-01-26": "Republic Day",
    "2026-02-17": "Maha Shivaratri",
    "2026-03-03": "Holi",
    "2026-03-20": "Eid al-Fitr",
    "2026-03-22": "Ugadi",
    "2026-04-03": "Good Friday",
    "2026-04-14": "Ambedkar Jayanti",
    "2026-05-01": "May Day",
    "2026-05-27": "Bakrid / Eid al-Adha",
    "2026-08-15": "Independence Day",
    "2026-09-14": "Ganesh Chaturthi",
    "2026-10-02": "Gandhi Jayanti",
    "2026-10-18": "Ayudha Puja",
    "2026-10-19": "Vijayadashami",
    "2026-11-01": "Kannada Rajyotsava",
    "2026-11-08": "Diwali",
    "2026-12-25": "Christmas Day"
}

def get_holiday_context(date: datetime.date = None) -> str | None:
    """Checks if the given date is a known public holiday or major event."""
    if date is None:
        date = datetime.date.today()
    
    date_str = date.strftime("%Y-%m-%d")
    return KARNATAKA_HOLIDAYS_2026.get(date_str)
