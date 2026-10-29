#!/usr/bin/env python3
"""
fleet_maintenance_stream.py - streams per-vehicle maintenance "daily snapshot" events on a fast clock.

Companion to fleet_maintenance_gen.py (same simulation rules and feature columns) but emitted live:
one simulated day passes every --seconds-per-day wall seconds, and every vehicle emits one event per
simulated day, spread randomly across that interval. It reads only the ERP `vehicles` table and does
not touch fleet_telemetry_stream.py.

Event (one JSON line / Event Hub message):
    source="maintenance", vehicle_id, timestamp (simulated, UTC), snapshot_date,
    vehicle_type, vehicle_model, fuel_type, transmission_type, engine_size_cc, vehicle_age_yrs,
    odometer_km, last_service_date, days_since_service, km_since_service, service_count,
    accident_count, tire_condition, brake_condition, battery_status, fuel_efficiency_kmpl
The label is NOT in the event (a real stream would not have it). Ground truth goes to
<out-dir>/maintenance_truth.csv (vehicle_id, snapshot_date, Need_Maintenance, true_risk) so score_stream.py
can evaluate the predictions afterwards.

Data-quality faults (counted in run_summary.json; the scorer should reject/dedup them):
    malformed JSON / missing key, out-of-range values, exact duplicate resends.
    Duplicates: dedup on (vehicle_id, snapshot_date).

Use a DIFFERENT --seed from the one used to create your training data (default here is 1001, the batch
generator's default is 42). With the same seed and fleet you would replay the training vehicles exactly.

Sinks:
    --sink file       <out-dir>/maintenance_events.ndjson (appended, flushed every second: tail it)
    --sink eventhub   env EH_CONN_MAINT (connection string incl. EntityPath), EH_NAME_MAINT (only if no EntityPath)

    pip install numpy pandas azure-eventhub
    python fleet_maintenance_stream.py --erp-dir ./erp --seconds-per-day 10 --days 60
    python fleet_maintenance_stream.py --erp-dir ./erp --seconds-per-day 10 --days 0 --sink eventhub   # forever
"""
import argparse
import csv
import datetime as dt
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["Actros", "Atego", "FH16", "FM", "Daily", "Transit", "Sprinter", "Cargo"]
FUELS = ["Diesel", "Diesel", "Diesel", "Petrol", "Electric"]
TRANS = ["Manual", "Automatic"]
BASE_KMPL = {"Diesel": 4.5, "Petrol": 5.5, "Electric": 0.0}


# --------------------------------------------------------------------------- ERP loading
def read_table(erp, name, load_date):
    d = erp / name / f"load_date={load_date}"
    for ext in ("parquet", "csv"):
        f = d / f"{name}.{ext}"
        if f.exists():
            return pd.read_parquet(f) if ext == "parquet" else pd.read_csv(f)
    raise FileNotFoundError(f"{name} not found in {d}")


def load_vehicles(erp_dir, limit):
    erp = Path(erp_dir)
    runs = sorted(p.parent.name.split("=", 1)[1] for p in (erp / "_runs").glob("load_date=*/_SUCCESS"))
    if not runs:
        sys.exit(f"No completed ERP days (_SUCCESS) under {erp}. Run fleet_erp_batch.py first.")
    veh = read_table(erp, "vehicles", runs[0])[["vehicle_id", "vehicle_type"]].drop_duplicates("vehicle_id")
    if limit:
        veh = veh.head(limit)
    return veh.reset_index(drop=True)


# --------------------------------------------------------------------------- simulation (same rules as batch gen)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cond3(x, lo=0.45, hi=0.75):
    return np.where(x < lo, "Good", np.where(x < hi, "Average", "Poor"))


class MaintSim:
    def __init__(self, veh, seed, start_date):
        rng = self.rng = np.random.default_rng(seed)
        n = self.n = len(veh)
        self.veh = veh
        is_reefer = veh["vehicle_type"].astype(str).str.startswith("REEFER").to_numpy()
        self.model = rng.choice(MODELS, n)
        self.fuel = rng.choice(FUELS, n)
        self.trans = rng.choice(TRANS, n, p=[0.7, 0.3])
        self.engine_cc = np.where(self.fuel == "Electric", 0, rng.choice([2000, 2500, 3000, 5000, 7000, 10000], n))
        self.age0 = rng.uniform(1, 12, n)
        self.interval = rng.choice([10000, 15000, 20000], n).astype(float)
        self.delay = rng.uniform(0.0, 0.35, n)
        self.tire_rate = rng.lognormal(0, 0.25, n)
        self.brake_rate = rng.lognormal(0, 0.25, n)
        self.batt_rate = rng.lognormal(0, 0.3, n)
        self.daily_km_mu = np.where(is_reefer, rng.uniform(250, 480, n), rng.uniform(180, 420, n))
        self.odo = self.age0 * rng.uniform(40000, 100000, n)
        self.km_since = rng.uniform(0, self.interval * (1 + self.delay), n)
        self.last_service = np.array(
            [start_date - dt.timedelta(days=int(d)) for d in self.km_since / np.maximum(self.daily_km_mu, 1)],
            dtype=object)
        self.service_count = np.floor(self.odo / self.interval).astype(int)
        self.accidents = rng.poisson(0.25 * self.age0 / 3.0)
        self.tire = np.clip(rng.uniform(0, 0.8, n), 0, 1)
        self.brake = np.clip(rng.uniform(0, 0.8, n), 0, 1)
        self.batt = np.clip(1 - self.age0 * 0.04 * self.batt_rate - rng.uniform(0, 0.15, n), 0.2, 1)
        self.need = np.zeros(n, dtype=bool)
        self.day_i = 0

    def step(self, day):
        """Advance one simulated day; return (features DataFrame, need array, risk array)."""
        rng, n = self.rng, self.n
        km = np.clip(rng.normal(self.daily_km_mu, 60), 0, None) * (rng.random(n) > 0.12)
        self.odo += km
        self.km_since += km
        self.tire = np.clip(self.tire + km / (70000 * self.tire_rate), 0, 1.2)
        self.brake = np.clip(self.brake + km / (110000 * self.brake_rate) * (1 + 0.3 * self.accidents / 3), 0, 1.2)
        self.batt = np.clip(self.batt - 0.00012 * self.batt_rate - rng.normal(0, 0.0002, n), 0.2, 1)
        self.accidents = self.accidents + (rng.random(n) < 0.0004).astype(int)

        overdue = self.km_since > self.interval * (1 + self.delay)
        flagged_fix = self.need & (rng.random(n) < 0.10)
        serviced = overdue | flagged_fix
        self.km_since[serviced] = 0.0
        self.service_count[serviced] += 1
        self.brake[serviced] *= 0.35
        self.batt[serviced] = np.minimum(1.0, self.batt[serviced] + 0.03)
        for i in np.nonzero(serviced)[0]:
            self.last_service[i] = day
        worn_t = self.tire >= 1.0
        self.tire[worn_t] = rng.uniform(0.0, 0.1, int(worn_t.sum()))
        worn_b = self.brake >= 1.0
        self.brake[worn_b] = rng.uniform(0.0, 0.1, int(worn_b.sum()))

        days_since = np.array([(day - d).days for d in self.last_service], dtype=int)
        age = self.age0 + (self.day_i + 1) / 365.0
        km_ratio = np.clip(self.km_since / self.interval, 0, 2.0)
        z = (-5.0 + 1.8 * km_ratio + 2.0 * np.clip(self.tire, 0, 1) + 1.8 * np.clip(self.brake, 0, 1)
             + 1.5 * (1 - self.batt) + 0.08 * age + 0.4 * np.minimum(self.accidents, 4)
             + 0.4 * np.minimum(days_since / 180, 2.0) + rng.normal(0, 0.5, n))
        risk = sigmoid(z)
        self.need = rng.random(n) < risk
        wear_pen = 1 - 0.15 * np.clip(self.tire, 0, 1) - 0.10 * np.clip(self.brake, 0, 1) - 0.05 * km_ratio
        kmpl = np.array([BASE_KMPL[f] for f in self.fuel]) * wear_pen * rng.normal(1, 0.03, n)
        kmpl = np.where(self.fuel == "Electric", np.nan, kmpl)
        self.day_i += 1

        df = pd.DataFrame({
            "vehicle_id": self.veh["vehicle_id"],
            "vehicle_type": self.veh["vehicle_type"],
            "vehicle_model": self.model,
            "fuel_type": self.fuel,
            "transmission_type": self.trans,
            "engine_size_cc": self.engine_cc,
            "vehicle_age_yrs": np.round(age, 2),
            "odometer_km": np.round(self.odo, 1),
            "last_service_date": [d.isoformat() for d in self.last_service],
            "days_since_service": days_since,
            "km_since_service": np.round(self.km_since, 1),
            "service_count": self.service_count,
            "accident_count": self.accidents,
            "tire_condition": cond3(self.tire),
            "brake_condition": cond3(self.brake),
            "battery_status": np.where(self.batt > 0.75, "Good", np.where(self.batt > 0.5, "Weak", "Replace")),
            "fuel_efficiency_kmpl": np.round(kmpl, 2),
        })
        return df, self.need.copy(), risk


# --------------------------------------------------------------------------- sinks
class FileSink:
    def __init__(self, out_dir):
        self.f = open(out_dir / "maintenance_events.ndjson", "a", buffering=1 << 20)

    def send(self, bodies):
        if bodies:
            self.f.write("\n".join(bodies) + "\n")

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()


class EventHubSink:
    def __init__(self, out_dir):
        from azure.eventhub import EventHubProducerClient
        conn = os.environ.get("EH_CONN_MAINT")
        if not conn:
            sys.exit("Missing env var EH_CONN_MAINT")
        self.client = EventHubProducerClient.from_connection_string(conn, eventhub_name=os.environ.get("EH_NAME_MAINT"))
        self.dead = out_dir / "failed_batches.ndjson"

    def _send_all(self, bodies):
        from azure.eventhub import EventData
        batch = self.client.create_batch()
        for b in bodies:
            try:
                batch.add(EventData(b))
            except ValueError:
                self.client.send_batch(batch)
                batch = self.client.create_batch()
                batch.add(EventData(b))
        if len(batch) > 0:
            self.client.send_batch(batch)

    def send(self, bodies):
        if not bodies:
            return
        delay = 0.5
        for attempt in range(1, 6):
            try:
                self._send_all(bodies)
                return
            except Exception as e:                      # noqa: BLE001
                print(f"[warn] send attempt {attempt} failed: {e!r}", file=sys.stderr)
                time.sleep(delay)
                delay *= 2
        with open(self.dead, "a") as f:
            for b in bodies:
                f.write(json.dumps({"source": "maintenance", "body": b}) + "\n")

    def flush(self):
        pass

    def close(self):
        self.client.close()


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--erp-dir", default="./erp")
    ap.add_argument("--out-dir", default="./maint_stream_out")
    ap.add_argument("--sink", choices=["file", "eventhub"], default="file")
    ap.add_argument("--vehicles", type=int, default=0, help="limit fleet size (0 = all)")
    ap.add_argument("--seconds-per-day", type=float, default=10.0, help="wall seconds per simulated day")
    ap.add_argument("--days", type=int, default=30, help="simulated days to stream (0 = until stopped)")
    ap.add_argument("--start-date", default=None, help="first simulated day YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--seed", type=int, default=1001, help="use a different seed than your training data")
    ap.add_argument("--malformed-rate", type=float, default=0.002)
    ap.add_argument("--outofrange-rate", type=float, default=0.002)
    ap.add_argument("--duplicate-rate", type=float, default=0.003)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    veh = load_vehicles(args.erp_dir, args.vehicles)
    n = len(veh)
    start = dt.date.fromisoformat(args.start_date) if args.start_date else dt.datetime.now(dt.timezone.utc).date()
    sim = MaintSim(veh, args.seed, start)
    pyr = np.random.default_rng(args.seed + 1)            # separate stream for fault injection / offsets
    sink = EventHubSink(out_dir) if args.sink == "eventhub" else FileSink(out_dir)
    truth_f = open(out_dir / "maintenance_truth.csv", "w", newline="")
    truth = csv.writer(truth_f)
    truth.writerow(["vehicle_id", "snapshot_date", "Need_Maintenance", "true_risk"])

    stop = {"flag": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: stop.update(flag=True))
        except (ValueError, OSError):
            pass

    ticks = max(1, int(round(args.seconds_per_day)))
    print(f"fleet: {n} vehicles, start {start}, {args.seconds_per_day}s per simulated day "
          f"(~{n / max(args.seconds_per_day, 1e-9):.0f} events/s), sink={args.sink}", flush=True)
    stats = {"events": 0, "malformed": 0, "out_of_range": 0, "duplicate": 0}
    t0 = time.perf_counter()
    d = 0
    try:
        while not stop["flag"] and (args.days == 0 or d < args.days):
            day = start + dt.timedelta(days=d)
            df, need, risk = sim.step(day)
            ids = df["vehicle_id"].tolist()
            for i in range(n):
                truth.writerow([ids[i], day.isoformat(), int(need[i]), round(float(risk[i]), 4)])
            truth_f.flush()
            records = df.astype(object).where(df.notna(), None).to_dict("records")
            offsets = pyr.integers(0, ticks, n)
            fracs = pyr.random(n)
            day_start = time.perf_counter()
            for tick in range(ticks):
                if stop["flag"]:
                    break
                bodies = []
                for i in np.nonzero(offsets == tick)[0]:
                    sec = (tick + fracs[i]) / ticks * 86400.0
                    ts = (dt.datetime.combine(day, dt.time(0), tzinfo=dt.timezone.utc)
                          + dt.timedelta(seconds=float(sec))).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                    ev = {"source": "maintenance", "timestamp": ts, "snapshot_date": day.isoformat(), **records[i]}
                    r = pyr.random()
                    if r < args.malformed_rate:
                        stats["malformed"] += 1
                        s = json.dumps(ev, separators=(",", ":"))
                        if pyr.random() < 0.5:
                            bodies.append(s[:int(pyr.integers(10, max(11, len(s) - 5)))])
                        else:
                            ev.pop("vehicle_id", None)
                            bodies.append(json.dumps(ev, separators=(",", ":")))
                        continue
                    if r < args.malformed_rate + args.outofrange_rate:
                        stats["out_of_range"] += 1
                        ev["odometer_km"] = -1.0
                    body = json.dumps(ev, separators=(",", ":"))
                    bodies.append(body)
                    if pyr.random() < args.duplicate_rate:
                        stats["duplicate"] += 1
                        bodies.append(body)
                sink.send(bodies)
                sink.flush()
                stats["events"] += len(bodies)
                wait = day_start + (tick + 1) * (args.seconds_per_day / ticks) - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            d += 1
            print(f"[day {d:>4}] {day}  events so far={stats['events']:,}  "
                  f"high-risk vehicles today={int(need.sum())}", flush=True)
    finally:
        sink.close()
        truth_f.close()
        summary = {"vehicles": n, "simulated_days": d, "start_date": start.isoformat(), "seed": args.seed,
                   "wall_seconds": round(time.perf_counter() - t0, 1), "events_sent": stats["events"],
                   "faults_injected": {k: v for k, v in stats.items() if k != "events"}}
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
