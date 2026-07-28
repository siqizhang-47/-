"""Fixed schema of the HEEW dataset."""

SHEET_NAME = "Aligned_Data"
EXCLUDED_SHEETS = ["Missing_Report"]

# timestamp is assembled from these columns (no single Date column)
TIME_COLUMNS = ["Year", "Month", "Day", "Hour"]

TARGET_COLUMNS = ["Electricity", "Cooling", "Heat"]
TARGET_NAMES = ["electricity", "cooling", "heat"]

WEATHER_COLUMNS = ["Temperature", "Dew Point", "Humidity"]
WEATHER_NAMES = ["temperature", "dew_point", "humidity"]

CALENDAR_NAMES = [
    "sin_hour",
    "cos_hour",
    "sin_dayofweek",
    "cos_dayofweek",
    "sin_dayofyear",
    "cos_dayofyear",
    "is_weekend",
]

NUM_TARGETS = len(TARGET_NAMES)          # 3
NUM_WEATHER = len(WEATHER_COLUMNS)       # 3
NUM_CALENDAR = len(CALENDAR_NAMES)       # 7
CONDITION_DIM = NUM_WEATHER + NUM_CALENDAR  # 10

EXPECTED_ROWS = 78888
EXPECTED_FREQ_SECONDS = 3600

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10  # val_end = int(N * 0.80)

CONTEXT_LENGTH = 168
PREDICTION_LENGTH = 24
LABEL_LENGTH = 84
HORIZON = 1

FUTURE_WEATHER_MODE = "oracle_observed"
