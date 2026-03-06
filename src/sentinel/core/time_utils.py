"""
sentinel/core/time_utils.py

Time System Hnadling
Thin wrapper around astropy.time providing a consistent interface
for all time conversions used.
"""
import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.utils.iers import conf as iers_conf

iers_conf.auto_download = True

def ensure_time(t, scale="utc"):
    if isinstance(t, Time):
        return t
    if isinstance(t, str):
        return Time(t, scale=scale)
    if isinstance(t, (int, float)):
        return Time(t, format="jd", scale=scale)
    return Time(t, scale=scale)

# Return Julian Date in given time scale
def to_jd(t, scale="utc"):
    t = ensure_time(t)
    return getattr(t, scale).jd

# " Modified Julian Date
def to_mjd(t, scale="utc"):
    t = ensure_time(t)
    return getattr(t, scale).mjd

# " TDB (for planetary ephemeris)
def utc_to_tdb(t):
    t = ensure_time(t)
    return t.tdb

# " TT
def utc_to_tt(t):
    t = ensure_time(t)
    return t.tt

# ut1
def utc_to_ut1(t):
    t = ensure_time(t)
    return t.ut1

# gps
def utc_to_gps(t):
    t = ensure_time(t)
    return t.gps

# Greenwich mean Sidereal time (in radians)
def gmst(t):
    t = ensure_time(t)
    return t.sidereal_time("mean", longitude = 0).rad 

# ut1 utc offset
def delta_ut1_utc(t):
    t = ensure_time(t)
    return (t.ut1 - t).sec

# epoch delta/elapsed time
def elapsed_sec(t1, t2):
    t1, t2 = ensure_time(t1), ensure_time(t2)
    return (t2 - t1).sec

# array of 'Time' objects from start to stop
# step_seconds -> float; start,stop -> 'Time'
def epoch_range(start, stop, step_sec):
    start, stop = ensure_time(start), ensure_time(stop)
    total = (stop - start).sec 
    n = round(total / step_sec) + 1
    delta = TimeDelta(np.linspace(0, total, n), format="sec")
    return start + delta