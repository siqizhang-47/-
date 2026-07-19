from src.data import low_carbon_schema as S


def test_fixed_orders():
    assert S.TARGET_NAMES == ["electricity", "cooling", "heating", "pv"]
    assert S.ZERO_TARGET_INDICES == [1, 2, 3]
    assert S.WEATHER_NAMES == ["irradiance", "temperature", "humidity", "wind_speed"]
    assert S.CONDITION_DIM == 11
    assert len(S.CALENDAR_NAMES) == 7


def test_target_gate_mapping():
    assert S.TARGET_TO_GATE == {1: 0, 2: 1, 3: 2}


def test_forbidden_columns_listed():
    for kw in ["Gas engine", "Fuel cell", "Wind direction"]:
        assert kw in S.FORBIDDEN_COLUMN_KEYWORDS
