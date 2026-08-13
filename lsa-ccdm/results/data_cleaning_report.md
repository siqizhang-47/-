# Data cleaning report

- rows: 78888, span: 2014-01-01 00:00:00 .. 2022-12-31 23:00:00 (continuous hourly)
- DewPoint: 119 zeros found, 88 classified as missing markers -> time-interpolated (boundary values back/forward-filled), 31 kept as genuine zero crossings, 0 NaN remaining
  - gaps > 3h (checked, brief sensor outages, interpolated): 2014-01-01 06:00:00 .. 2014-01-01 10:00:00 (5h); 2014-02-05 20:00:00 .. 2014-02-05 23:00:00 (4h); 2015-08-31 22:00:00 .. 2015-09-01 05:00:00 (8h)
- Humidity: 17 zeros found, 17 classified as missing markers -> time-interpolated (boundary values back/forward-filled), 0 kept as genuine zero crossings, 0 NaN remaining
  - gaps > 3h (checked, brief sensor outages, interpolated): 2015-08-31 22:00:00 .. 2015-09-01 05:00:00 (8h)
- Temperature: 15 zeros found, 15 classified as missing markers -> time-interpolated (boundary values back/forward-filled), 0 kept as genuine zero crossings, 0 NaN remaining
  - gaps > 3h (checked, brief sensor outages, interpolated): 2015-08-31 22:00:00 .. 2015-09-01 05:00:00 (8h)

Column order: date, PV, Electricity, Cooling, Heat, Temperature, DewPoint, Humidity, GHI
