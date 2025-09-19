from datetime import datetime, timedelta
import pytz
import re

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

def format_datetime_for_ib_utc(dt: datetime) -> str:
    """
    Takes a datetime (naive or timezone-aware) and converts it to IB-compatible UTC string.
    Format: yyyymmdd-hh:mm:ss
    """
    if dt.tzinfo is None:
        raise ValueError("Datetime object must be timezone-aware")

    dt_utc = dt.astimezone(pytz.utc)
    return dt_utc.strftime('%Y%m%d-%H:%M:%S')

def utcInMinutes(minutes=1, timezone="US/Eastern"):
    ny_timezone = pytz.timezone(timezone)

    # Calculate the time 3 minutes from now in UTC
    utc_now = datetime.now(pytz.utc)
    cancel_time = utc_now + timedelta(minutes=minutes)

    return format_datetime_for_ib_utc(cancel_time)

def get_time_plus_x_minutes_ny(x=5):
    # Define New York timezone
    ny_tz = pytz.timezone('America/New_York')

    # Get current time in New York and add 5 minutes
    now = datetime.now(ny_tz)
    future_time = now + timedelta(minutes=x)

    # Format into the required string: 'yyyymmdd hh:mm:ss TimeZone'
    date_str = future_time.strftime('%Y%m%d %H:%M:%S')

    return f"{date_str} America/New_York"

def is_valid_date_string(date_str):
    utc_format = re.compile(r'^\d{8}-\d{2}:\d{2}:\d{2}$')
    local_tz_format = re.compile(r'^(?:(\d{8}) )?(\d{2}:\d{2}:\d{2})(?: ([\w/_]+))?$')

    try:
        if utc_format.match(date_str):
            # UTC format: yyyymmdd-hh:mm:ss
            dt = datetime.strptime(date_str, '%Y%m%d-%H:%M:%S')
            return True
        match = local_tz_format.match(date_str)
        if match:
            date_part, time_part, tz_part = match.groups()

            # Use current date if date is not specified
            if not date_part:
                date_part = datetime.now().strftime('%Y%m%d')
            full_str = f'{date_part} {time_part}'

            # Parse datetime to check format
            datetime.strptime(full_str, '%Y%m%d %H:%M:%S')

            # Check if time zone is valid
            if tz_part:
                if tz_part not in pytz.all_timezones:
                    return False

            return True
    except Exception:
        return False
    return False

if __name__ == "__main__":
    # Example usage:
    ny_time = inTzTime(9, 30)
    print("Time in New York today:", ny_time)

    plusTime1 = utcInMinutes(1)
    plusTime2 = get_time_plus_x_minutes_ny(1)

    print("Now in 1 minute in New York:"+plusTime1+",", is_valid_date_string(plusTime1))
    print("Now in 1 minute in New York:", plusTime2, is_valid_date_string(plusTime2))
