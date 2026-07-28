from src.data import low_carbon_schema as S


def test_fixed_orders():
    assert S.TARGET_NAMES == ["electricity", "cooling", "heat"]
    assert S.TARGET_COLUMNS == ["Electricity", "Cooling", "Heat"]
    assert S.WEATHER_COLUMNS == ["Temperature", "Dew Point", "Humidity"]
    assert S.CONDITION_DIM == 10
    assert len(S.CALENDAR_NAMES) == 7


def test_dimensions():
    assert S.NUM_TARGETS == 3
    assert S.NUM_WEATHER == 3
    assert S.CONTEXT_LENGTH == 168
    assert S.PREDICTION_LENGTH == 24
