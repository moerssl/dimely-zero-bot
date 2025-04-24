from datetime import datetime
import pytz

def inTzTime(hours, minutes, timezone="America/New_York"):
    # Define the target timezone using pytz
    target_tz = pytz.timezone(timezone)
    # Get today's date in the target timezone
    today = datetime.now(target_tz).date()
    # Create a naive datetime object with today's date and provided time
    naive_time = datetime(today.year, today.month, today.day, hours, minutes)
    # Localize the naive datetime to the target timezone (no conversion)
    localized_time = target_tz.localize(naive_time)
    return localized_time

if __name__ == "__main__":
    # Example usage:
    ny_time = inTzTime(9, 30)
    print("Time in New York today:", ny_time)
