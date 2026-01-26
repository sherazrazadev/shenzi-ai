import csv
import requests
from io import StringIO
from typing import List, Dict

# Public CSV export URL for the Google Sheet
gsheet_csv_url = "https://docs.google.com/spreadsheets/d/1i0HfQG8BreKKXkjMLRbVytH1mRuq0tHJkLtWTEntcdM/export?format=csv"

def fetch_sheet_data() -> List[Dict[str, str]]:
    """Fetches and parses the Google Sheet as a list of dicts."""
    response = requests.get(gsheet_csv_url)
    response.raise_for_status()
    csvfile = StringIO(response.text)
    reader = csv.DictReader(csvfile)
    return list(reader)

def get_tasks_for_date(date_str: str) -> List[Dict[str, str]]:
    """Returns all tasks for a given date (YYYY-MM-DD)."""
    data = fetch_sheet_data()
    return [row for row in data if row.get("Date") == date_str]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        date = sys.argv[1]
        tasks = get_tasks_for_date(date)
        for task in tasks:
            print(task)
    else:
        print("Usage: python sheet_tasks.py YYYY-MM-DD")
