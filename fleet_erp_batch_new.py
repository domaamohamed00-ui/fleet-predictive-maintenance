#!/usr/bin/env python3
"""
fleet_erp_batch.py  -  Simulated ERP extracts for the FMCG fleet telematics project.

Tables (one folder per table, one partition per business day):

    <out-dir>/<table>/load_date=YYYY-MM-DD/<table>.csv|parquet
    <out-dir>/_runs/load_date=YYYY-MM-DD/_SUCCESS          (row counts; written LAST)

    corridors       fixed logistics corridors (origin city -> destination city, waypoints)
    drivers         driver master
    vehicles        fleet master (DRY_VAN / REEFER_CHILLED / REEFER_FROZEN), fixed corridor + current driver
    customers       retail / wholesale customers, located in corridor destination cities
    products        FMCG SKUs with temperature class and spoilage risk tier
    shipments       one trip per vehicle per day (header, cargo totals, cold-chain targets, expenses)
    shipment_lines  products on board each shipment (the natural fact grain)

Day 0  : full master data + that day's shipments / shipment_lines.
Day N>0: ONLY master rows that changed that day (SCD1 / SCD2 demos) + new shipments / lines.

Properties that matter for the project:
  * Deterministic: same --seed + --day => identical files (idempotent reruns and backfills).
  * Temperature compatibility: dry vans carry AMBIENT only, chilled reefers CHILLED (+ some AMBIENT),
    frozen reefers FROZEN only. Cargo never exceeds vehicle capacity.
  * _SUCCESS is written last, so a pipeline can ignore half-written days.
    --simulate-failure crashes mid-run to exercise the "kill and retry" test.
  * Business date defaults to TODAY so ERP dates line up with the telemetry stream.

Run anywhere: locally, or in a Fabric notebook with --out-dir /lakehouse/default/Files/erp

    pip install numpy pandas faker pyarrow        (faker, pyarrow optional)
    python fleet_erp_batch.py --out-dir ./erp --day 0 --through 2
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from faker import Faker
except ImportError:  # names fall back to placeholders
    Faker = None

# --------------------------------------------------------------------------- reference data
# Edit freely: store/customer cities (name, lat, lon) and which of them host a distribution centre.
CITIES = [
    ("Cairo", 30.0444, 31.2357), ("Alexandria", 31.2001, 29.9187), ("Tanta", 30.7865, 31.0004),
    ("Mansoura", 31.0409, 31.3785), ("Zagazig", 30.5877, 31.5021), ("Ismailia", 30.5965, 32.2715),
    ("Suez", 29.9668, 32.5498), ("Port Said", 31.2653, 32.3019), ("Damietta", 31.4165, 31.8133),
    ("Beni Suef", 29.0661, 31.0994), ("Fayoum", 29.3084, 30.8428), ("Minya", 28.1099, 30.7503),
    ("Asyut", 27.1783, 31.1859), ("Sohag", 26.5591, 31.6948), ("Banha", 30.4659, 31.1848),
    ("Damanhur", 31.0341, 30.4682),
]
ORIGIN_CITIES = ["Cairo", "Alexandria", "Tanta", "Ismailia", "Minya"]   # corridor origins (DCs)

# (category, temp_class, spoilage_risk_tier, share of SKUs, unit weight kg range, unit price EGP range)
# Price ranges are placeholders - replace with your own.
CATEGORIES = [
    ("Dairy & Yogurt",      "CHILLED", "HIGH",   .14, (0.2, 1.0), (10, 120)),
    ("Chilled Ready Meals", "CHILLED", "HIGH",   .08, (0.3, 0.6), (40, 200)),
    ("Fresh Juice",         "CHILLED", "MEDIUM", .06, (0.5, 1.5), (15, 90)),
    ("Frozen Foods",        "FROZEN",  "MEDIUM", .12, (0.4, 2.0), (40, 250)),
    ("Ice Cream",           "FROZEN",  "HIGH",   .06, (0.1, 1.5), (15, 180)),
    ("Beverages",           "AMBIENT", "LOW",    .16, (0.3, 6.0), (8, 90)),
    ("Snacks",              "AMBIENT", "LOW",    .14, (0.05, 0.5), (5, 60)),
    ("Packaged Food",       "AMBIENT", "LOW",    .13, (0.2, 3.0), (15, 150)),
    ("Household",           "AMBIENT", "LOW",    .07, (0.3, 5.0), (30, 300)),
    ("Personal Care",       "AMBIENT", "LOW",    .04, (0.1, 1.0), (30, 350)),
]
TEMP_RANK = {"AMBIENT": 0, "CHILLED": 1, "FROZEN": 2}
RISK_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
TARGET_TEMP_C = {"CHILLED": 4.0, "FROZEN": -20.0}    # simulation thresholds, not regulatory ones
LIMIT_TEMP_C = {"CHILLED": 8.0, "FROZEN": -15.0}     # a reading above this = cold-chain breach
CHANNELS = ["HYPERMARKET", "SUPERMARKET", "CONVENIENCE", "WHOLESALE", "HORECA"]


# --------------------------------------------------------------------------- helpers
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def make_waypoints(lat1, lon1, lat2, lon2, rng, n=10):
    t = np.linspace(0, 1, n)
    lats = lat1 + (lat2 - lat1) * t
    lons = lon1 + (lon2 - lon1) * t
    wig = rng.normal(0, 0.015, n)
    wig[0] = wig[-1] = 0.0
    dlat, dlon = lat2 - lat1, lon2 - lon1
    norm = math.hypot(dlat, dlon) or 1.0
    lats = lats - dlon / norm * wig
    lons = lons + dlat / norm * wig
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lats, lons)]


def route_distance_km(wps):
    return sum(haversine_km(a[0], a[1], b[0], b[1]) for a, b in zip(wps[:-1], wps[1:]))


# --------------------------------------------------------------------------- day-0 master data
def build_initial(seed, args, start_ts):
    rng = np.random.default_rng(seed)
    fk = None
    if Faker:
        Faker.seed(seed)
        fk = Faker()

    def person():
        return fk.name() if fk else f"Person {int(rng.integers(10**6)):06d}"

    city_xy = {c[0]: (c[1], c[2]) for c in CITIES}

    # corridors: every origin city to `dest_per_origin` other cities
    rows, cid = [], 0
    for origin in ORIGIN_CITIES:
        olat, olon = city_xy[origin]
        others = [c for c in CITIES if c[0] != origin]
        k = min(args.dest_per_origin, len(others))
        for j in rng.choice(len(others), k, replace=False):
            dest = others[int(j)]
            wps = make_waypoints(olat, olon, dest[1], dest[2], rng)
            dist = route_distance_km(wps)
            cid += 1
            rows.append({
                "corridor_id": f"COR{cid:03d}", "corridor_name": f"{origin} - {dest[0]}",
                "origin_city": origin, "destination_city": dest[0],
                "distance_km": round(dist, 1), "planned_duration_min": int(dist / 55.0 * 60) + 15,
                "waypoints": json.dumps(wps), "updated_at": start_ts})
    corridors = pd.DataFrame(rows)

    # customers: round-robin over cities so every destination has customers
    nc = args.customers
    city_idx = np.arange(nc) % len(CITIES)
    customers = pd.DataFrame({
        "customer_id": [f"CUS{i+1:04d}" for i in range(nc)],
        "customer_name": [f"{CITIES[c][0]} Customer {i+1}" for i, c in enumerate(city_idx)],
        "channel": rng.choice(CHANNELS, nc),
        "city": [CITIES[c][0] for c in city_idx],
        "lat": [round(CITIES[c][1] + rng.normal(0, 0.03), 5) for c in city_idx],
        "lon": [round(CITIES[c][2] + rng.normal(0, 0.03), 5) for c in city_idx],
        "updated_at": start_ts})

    # products
    npr = args.products
    shares = np.array([c[3] for c in CATEGORIES])
    cat_i = rng.choice(len(CATEGORIES), npr, p=shares / shares.sum())
    words = [fk.word().title() if fk else f"Brand{i % 40}" for i in range(npr)]
    products = pd.DataFrame({
        "product_id": [f"SKU{i+1:04d}" for i in range(npr)],
        "product_name": [f"{CATEGORIES[c][0].split()[0]} {words[i]} {i+1:03d}" for i, c in enumerate(cat_i)],
        "product_category": [CATEGORIES[c][0] for c in cat_i],
        "temp_class": [CATEGORIES[c][1] for c in cat_i],
        "spoilage_risk_tier": [CATEGORIES[c][2] for c in cat_i],
        "unit_weight_kg": [round(float(rng.uniform(*CATEGORIES[c][4])), 3) for c in cat_i],
        "unit_price_egp": [round(float(rng.uniform(*CATEGORIES[c][5])), 2) for c in cat_i],
        "updated_at": start_ts})

    # drivers (110% of the fleet so spares exist) + vehicles
    nv = args.vehicles
    nd = int(nv * 1.1)
    drivers = pd.DataFrame({
        "driver_id": [f"DRV{i+1:05d}" for i in range(nd)],
        "driver_name": [person() for _ in range(nd)],
        "hire_date": [(start_ts - pd.Timedelta(days=int(rng.integers(100, 4000)))).date() for _ in range(nd)],
        "updated_at": start_ts})

    is_reefer = rng.random(nv) < args.reefer_share
    is_frozen = rng.random(nv) < args.frozen_share_of_reefer
    vtype = np.where(~is_reefer, "DRY_VAN", np.where(is_frozen, "REEFER_FROZEN", "REEFER_CHILLED"))
    cap = np.where(vtype == "DRY_VAN", rng.uniform(8, 12, nv), rng.uniform(5, 9, nv)).round(1)
    vehicles = pd.DataFrame({
        "vehicle_id": [f"V{i+1:05d}" for i in range(nv)],
        "vehicle_type": vtype,
        "capacity_tons": cap,
        "corridor_id": rng.choice(corridors["corridor_id"], nv),     # fixed corridor per vehicle
        "driver_id": rng.permutation(drivers["driver_id"].to_numpy())[:nv],
        "model_year": rng.integers(2016, 2025, nv),
        "updated_at": start_ts})

    return {"corridors": corridors, "drivers": drivers, "vehicles": vehicles,
            "customers": customers, "products": products}


# --------------------------------------------------------------------------- daily master changes
def apply_changes(state, day, seed, biz_ts):
    """Mutates `state` in place; returns only the changed rows per table."""
    rng = np.random.default_rng([seed, day])
    ts = biz_ts + pd.Timedelta(hours=2)
    changed = {}

    v = state["vehicles"]
    n = len(v)
    # driver swaps between pairs of vehicles -> SCD2 candidate (vehicles.driver_id)
    k = max(1, int(n * 0.005))
    idx = rng.choice(n, size=2 * k, replace=False)
    a, b = idx[:k], idx[k:]
    da = v.loc[a, "driver_id"].to_numpy().copy()
    db = v.loc[b, "driver_id"].to_numpy().copy()
    v.loc[a, "driver_id"] = db
    v.loc[b, "driver_id"] = da
    # corridor reassignments -> SCD2 candidate (vehicles.corridor_id)
    corr_ids = state["corridors"]["corridor_id"].tolist()
    mv = rng.choice(n, size=max(1, int(n * 0.003)), replace=False)
    for i in mv:
        cur = v.at[i, "corridor_id"]
        v.at[i, "corridor_id"] = rng.choice([c for c in corr_ids if c != cur])
    touched = sorted(set(idx.tolist()) | set(mv.tolist()))
    v.loc[touched, "updated_at"] = ts
    changed["vehicles"] = v.loc[touched].copy()

    # price changes -> SCD1 (overwrite)
    p = state["products"]
    pi = rng.choice(len(p), size=max(1, int(len(p) * 0.02)), replace=False)
    p.loc[pi, "unit_price_egp"] = (p.loc[pi, "unit_price_egp"] * rng.uniform(0.92, 1.08, len(pi))).round(2)
    p.loc[pi, "updated_at"] = ts
    changed["products"] = p.loc[pi].copy()

    # customer channel change -> SCD2 candidate
    c = state["customers"]
    ci = rng.choice(len(c), size=max(1, int(len(c) * 0.005)), replace=False)
    for i in ci:
        c.at[i, "channel"] = rng.choice([x for x in CHANNELS if x != c.at[i, "channel"]])
    c.loc[ci, "updated_at"] = ts
    changed["customers"] = c.loc[ci].copy()
    return changed


# --------------------------------------------------------------------------- daily transactions
def build_day_transactions(state, day, seed, biz_date):
    rng = np.random.default_rng([seed, 1000 + day])
    v, corr, cust, prods = (state[k] for k in ("vehicles", "corridors", "customers", "products"))
    dest = dict(zip(corr["corridor_id"], corr["destination_city"]))
    dur = dict(zip(corr["corridor_id"], corr["planned_duration_min"]))
    dist = dict(zip(corr["corridor_id"], corr["distance_km"]))
    cust_by_city = {c: g["customer_id"].to_numpy() for c, g in cust.groupby("city")}
    pool = {}
    for cls, g in prods.groupby("temp_class"):
        pool[cls] = (g["product_id"].to_numpy(), g["unit_weight_kg"].to_numpy(),
                     g["unit_price_egp"].to_numpy(), g["spoilage_risk_tier"].to_numpy())

    tag = f"{biz_date:%Y%m%d}"
    biz_ts = pd.Timestamp(biz_date)
    ships, lines = [], []
    for k, row in enumerate(v.itertuples(index=False)):
        cid_ = row.corridor_id
        customer_id = rng.choice(cust_by_city[dest[cid_]])
        n_lines = int(rng.integers(3, 9))
        if row.vehicle_type == "DRY_VAN":
            classes = ["AMBIENT"] * n_lines
        elif row.vehicle_type == "REEFER_FROZEN":
            classes = ["FROZEN"] * n_lines
        else:
            classes = ["CHILLED" if rng.random() < 0.85 else "AMBIENT" for _ in range(n_lines)]
            classes[0] = "CHILLED"

        recs = []   # [line_no, product_id, qty, unit_weight, price, class, risk]
        for ln, cls in enumerate(classes, 1):
            ids, w, pr, rk = pool[cls]
            j = int(rng.integers(len(ids)))
            recs.append([ln, ids[j], int(rng.integers(5, 61)), float(w[j]), float(pr[j]), cls, rk[j]])
        cap_kg = row.capacity_tons * 1000.0                # fill the vehicle to 40-90% of payload capacity
        total_kg = sum(r[2] * r[3] for r in recs)
        f = cap_kg * rng.uniform(0.4, 0.9) / total_kg
        for r in recs:
            r[2] = max(1, int(r[2] * f))

        ship_id = f"S{tag}{k:05d}"
        req = max((r[5] for r in recs), key=lambda c: TEMP_RANK[c])
        risk = max((r[6] for r in recs), key=lambda x: RISK_RANK[x])
        total_val = weight_kg = 0.0
        for ln, pid, qty, w, pr, cls, rk in recs:
            lv, lw = round(qty * pr, 2), round(qty * w, 2)
            total_val += lv
            weight_kg += lw
            lines.append({"shipment_id": ship_id, "line_no": ln, "product_id": pid,
                          "quantity_units": qty, "unit_price_egp": pr,
                          "line_value_egp": lv, "line_weight_kg": lw})

        is_reefer = row.vehicle_type != "DRY_VAN"
        rate = rng.uniform(14, 18) * (1.25 if is_reefer else 1.0)      # EGP/km placeholders
        expenses = dist[cid_] * 2 * rate + rng.uniform(300, 500)        # round trip + driver allowance
        start = biz_ts + pd.Timedelta(minutes=int(rng.integers(4 * 60, 10 * 60)))
        d = int(dur[cid_] * rng.uniform(0.9, 1.2))
        created = biz_ts + pd.Timedelta(hours=1)
        ships.append({
            "shipment_id": ship_id, "vehicle_id": row.vehicle_id, "driver_id": row.driver_id,
            "corridor_id": cid_, "customer_id": customer_id,
            "planned_start_ts": start, "planned_end_ts": start + pd.Timedelta(minutes=2 * d),
            "target_temp_c": TARGET_TEMP_C.get(req), "temp_limit_c": LIMIT_TEMP_C.get(req),
            "spoilage_risk_tier": risk, "cargo_weight_tons": round(weight_kg / 1000.0, 3),
            "total_cargo_value_egp": round(total_val, 2), "trip_expenses_egp": round(expenses, 2),
            "status": "PLANNED", "created_at": created, "updated_at": created})
    return {"shipments": pd.DataFrame(ships), "shipment_lines": pd.DataFrame(lines)}


# --------------------------------------------------------------------------- output
def write_table(df, name, load_date, out_dir, fmt):
    d = out_dir / name / f"load_date={load_date}"
    d.mkdir(parents=True, exist_ok=True)
    final = d / f"{name}.{fmt}"
    tmp = d / f".{name}.tmp"
    if fmt == "csv":
        df.to_csv(tmp, index=False)
    else:
        df.to_parquet(tmp, index=False)
    os.replace(tmp, final)          # atomic: readers never see a half-written file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="./erp")
    ap.add_argument("--format", choices=["csv", "parquet"], default="csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-date", default=dt.date.today().isoformat(),
                    help="business date of day 0 (default: today, so it lines up with the stream)")
    ap.add_argument("--day", type=int, default=0, help="first day index to write")
    ap.add_argument("--through", type=int, default=None, help="last day index to write (default = --day)")
    ap.add_argument("--vehicles", type=int, default=5000)
    ap.add_argument("--customers", type=int, default=600)
    ap.add_argument("--products", type=int, default=300)
    ap.add_argument("--dest-per-origin", type=int, default=10,
                    help="destinations per origin city (5 origins -> 5 x N corridors, max 15)")
    ap.add_argument("--reefer-share", type=float, default=0.38, help="fraction of fleet that is reefer")
    ap.add_argument("--frozen-share-of-reefer", type=float, default=0.38)
    ap.add_argument("--simulate-failure", action="store_true",
                    help="crash half-way through the day (no _SUCCESS marker) to test retry")
    args = ap.parse_args()
    through = args.day if args.through is None else args.through

    out = Path(args.out_dir)
    start = dt.date.fromisoformat(args.start_date)
    state = build_initial(args.seed, args, pd.Timestamp(start))

    for d in range(0, through + 1):
        biz = start + dt.timedelta(days=d)
        changed = None
        if d > 0:
            changed = apply_changes(state, d, args.seed, pd.Timestamp(biz))   # replay keeps state correct
        if d < args.day:
            continue
        tables = ({k: v.copy() for k, v in state.items()} if d == 0 else changed)
        tables.update(build_day_transactions(state, d, args.seed, biz))
        names = list(tables)
        counts = {}
        for i, name in enumerate(names):
            if args.simulate_failure and i == len(names) // 2:
                print(f"[day {d}] simulated failure after {i} tables (no _SUCCESS written)", file=sys.stderr)
                sys.exit(1)
            write_table(tables[name], name, biz.isoformat(), out, args.format)
            counts[name] = len(tables[name])
        run_dir = out / "_runs" / f"load_date={biz.isoformat()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "_SUCCESS").write_text(json.dumps(
            {"day_index": d, "load_date": biz.isoformat(), "seed": args.seed, "format": args.format,
             "rows": counts, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2))
        print(f"[day {d}] {biz} -> " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
