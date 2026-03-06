import pytest
import numpy as np
from astropy.time import Time
from sentinel.core.time_utils import (
    ensure_time, to_jd, to_mjd, utc_to_tt, utc_to_tdb, utc_to_ut1, gmst, elapsed_sec, epoch_range
)

def test_ensure_time_string():
    t = ensure_time("2026-03-05T12:00:00")
    assert isinstance(t, Time)
    assert t.scale == "utc"

def test_ensure_time_passthrough():
    t_in = Time("2026-03-05T12:00:00", scale="utc")
    t_out = ensure_time(t_in)
    assert t_out is t_in

def test_jd_mjd_relationship():
    t = ensure_time("2026-03-05T12:00:00")
    jd = to_jd(t)
    mjd = to_mjd(t)
    assert abs(jd - mjd - 2400000.5) < 1e-10

def test_utc_tdb_roundtrip():
    t_utc = ensure_time("2026-03-05T12:00:00")
    t_tdb = utc_to_tdb(t_utc)
    t_back = Time(t_tdb.jd1, t_tdb.jd2, format="jd", scale="tdb").utc
    assert abs((t_back - t_utc).sec) < 1e-9 # nanosecond

def test_tt_delta():
    t = ensure_time("2026-03-05T12:00:00")
    tt  = utc_to_tt(t)
    # TT - UTC should be leap_seconds + 32.184
    # Compare numerical JD values (not physical instants, which are equal)
    diff = (tt.jd - t.jd) * 86400.0
    # As of 2026, 37 leap seconds: TT-UTC = 37 + 32.184 = 69.184
    assert abs(diff - 69.184) < 0.5  # within 0.5s allows for future leap second

def test_gmst_type_range():
    t = ensure_time("2026-03-05T12:00:00")
    theta = gmst(t)
    assert 0 <= theta < 2 * np.pi

def test_elapsed_sec():
    t1 = "2026-03-05T00:00:00"
    t2 = "2026-03-05T01:00:00"
    assert abs(elapsed_sec(t1, t2) - 3600.0) < 1e-6

def test_epoch_range_endpoints():
    start = "2026-03-05T00:00:00"
    stop = "2026-03-05T01:00:00"
    times = epoch_range(start, stop, step_sec=600)
    assert len(times) == 7  # 0, 600, 1200, ..., 3600
    assert abs((times[-1] - ensure_time(stop)).sec) < 1e-6
