"""Fixed schema of the low-carbon community dataset (spec section 3)."""

SHEET_NAME = "Merged"
EXCLUDED_SHEETS = ["Anomaly_Log", "Anomaly_Summary"]

DATE_COLUMN = "Date"

TARGET_COLUMNS = [
    "[Electricity] Electricity load (kW)",
    "[Electricity] Cooling load (kW)",
    "[Electricity] Heating load (kW)",
    "[Power] Solar energy generation (kW)",
]
TARGET_NAMES = ["electricity", "cooling", "heating", "pv"]
ZERO_TARGET_INDICES = [1, 2, 3]          # cooling, heating, pv
ZERO_TARGET_NAMES = ["cooling", "heating", "pv"]
# target index -> gate index
TARGET_TO_GATE = {1: 0, 2: 1, 3: 2}

WEATHER_COLUMNS = [
    "[Weather] Horizontal solar irradition (W)",
    "[Weather] Outdoor air temperature (℃)",
    "[Weather] Outdoor air humidity (%)",
    "[Weather] Wind speed (m/s)",
]
WEATHER_NAMES = ["irradiance", "temperature", "humidity", "wind_speed"]

FORBIDDEN_COLUMN_KEYWORDS = ["Gas engine", "Fuel cell", "Wind direction"]

CALENDAR_NAMES = [
    "sin_hour",
    "cos_hour",
    "sin_dayofweek",
    "cos_dayofweek",
    "sin_dayofyear",
    "cos_dayofyear",
    "is_weekend",
]

NUM_TARGETS = len(TARGET_NAMES)          # 4
NUM_WEATHER = len(WEATHER_COLUMNS)       # 4
NUM_CALENDAR = len(CALENDAR_NAMES)       # 7
CONDITION_DIM = NUM_WEATHER + NUM_CALENDAR  # 11

EXPECTED_ROWS = 187752
EXPECTED_FREQ_SECONDS = 3600

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10  # val_end = int(N * 0.80)

CONTEXT_LENGTH = 168
PREDICTION_LENGTH = 24
LABEL_LENGTH = 84
HORIZON = 1

FUTURE_WEATHER_MODE = "oracle_observed"
