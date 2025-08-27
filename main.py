#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, json
import pandas as pd
import numpy as np
import datetime
import pytz
from supabase import create_client
from google_play_scraper import app as gp_app
from http.client import IncompleteRead

# ================== Setup ==================
IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.timezone("UTC")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ================== Cleaning Functions ==================
def package_wise_pairing(events_metadata_df: pd.DataFrame) -> pd.DataFrame:
    events_metadata_df = events_metadata_df.sort_values(["timestamp"]).reset_index(drop=True)
    events_metadata_df["Index"] = events_metadata_df.index
    events_metadata_df = events_metadata_df.sort_values(["package", "timestamp"]).reset_index(drop=True)

    events_metadata_df["prev_type"] = events_metadata_df.groupby("package")["type"].shift()
    events_metadata_df["next_type"] = events_metadata_df.groupby("package")["type"].shift(-1)
    events_metadata_df["broken"] = (
        (events_metadata_df["type"] == "paused") & (events_metadata_df["prev_type"] == "paused")
    ) | (
        (events_metadata_df["type"] == "resumed") & (events_metadata_df["next_type"] == "resumed")
    )
    linked_df = events_metadata_df[~events_metadata_df["broken"]].drop(
        columns=["prev_type", "broken", "next_type"]
    ).reset_index(drop=True)
    return linked_df


def chrono_pair(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["timestamp"]).reset_index(drop=True)
    df["prev_type"] = df["type"].shift()
    df["next_type"] = df["type"].shift(-1)
    df["prev_title"] = df["Title"].shift()
    df["next_title"] = df["Title"].shift(-1)
    df["prev_timestamp"] = df["timestamp"].shift()
    df["next_timestamp"] = df["timestamp"].shift(-1)
    df["broken"] = (
        (df["type"] == "paused") & (df["prev_type"] == "paused")
    ) | (
        (df["type"] == "resumed") & (df["next_type"] == "resumed")
    ) | (
        ((df["type"] == "resumed") & (df["next_type"] == "paused")) & (df["next_title"] != df["Title"])
    ) | (
        ((df["type"] == "paused") & (df["prev_type"] == "resumed")) & (df["prev_title"] != df["Title"])
    ) | (
        df["timestamp"] == df["prev_timestamp"]
    ) | (
        df["timestamp"] == df["next_timestamp"]
    )

    final_df = df[~df["broken"]].drop(
        columns=[
            "prev_type",
            "broken",
            "next_type",
            "prev_title",
            "next_title",
            "prev_timestamp",
            "next_timestamp",
        ]
    )
    return final_df


def calc_screentimes(df: pd.DataFrame) -> pd.DataFrame:
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(IST)
    df = df.sort_values("ts").reset_index(drop=True)

    df["next_type"] = df["type"].shift(-1)
    df["next_ts"] = df["ts"].shift(-1)
    df["next_Title"] = df["Title"].shift(-1)

    cond_screen = (df["type"] == "resumed") & (df["next_type"] == "paused")
    df.loc[cond_screen, "screentime"] = (df.loc[cond_screen, "next_ts"] - df.loc[cond_screen, "ts"]).dt.total_seconds()

    cond_off = (df["type"] == "paused") & (df["next_type"] == "resumed")
    df.loc[cond_off, "offscreentime"] = (df.loc[cond_off, "next_ts"] - df.loc[cond_off, "ts"]).dt.total_seconds()

    return df


def remove_short_pauses(df_time: pd.DataFrame, threshold_offscreen=3) -> pd.DataFrame:
    cond_off = (
        (df_time["type"] == "paused")
        & (df_time["next_type"] == "resumed")
        & (df_time["Title"] == df_time["next_Title"])
    )
    df_time["tiny_offscreen"] = cond_off & (df_time["offscreentime"] < threshold_offscreen)
    to_drop_off = list(df_time.index[df_time["tiny_offscreen"]])
    for idx in to_drop_off:
        if (idx + 1) in df_time.index and df_time.at[idx + 1, "type"] == "resumed" and df_time.at[idx + 1, "package"] == df_time.at[idx, "package"]:
            to_drop_off.append(idx + 1)

    df_clean_off = df_time.drop(index=list(set(to_drop_off))).drop(
        columns=["ts", "next_type", "next_ts", "next_Title", "screentime", "tiny_offscreen"], errors="ignore"
    )
    return df_clean_off


def remove_short_screentimes(df: pd.DataFrame, threshold_onscreen_min=3, threshold_onscreen_max=5 * 60 * 60) -> pd.DataFrame:
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(IST)
    df = df.sort_values(["timestamp"]).reset_index(drop=True)
    df["next_type"] = df["type"].shift(-1)
    df["next_ts"] = df["ts"].shift(-1)

    cond_screen = (df["type"] == "resumed") & (df["next_type"] == "paused")
    df.loc[cond_screen, "screentime"] = (df.loc[cond_screen, "next_ts"] - df.loc[cond_screen, "ts"]).dt.total_seconds()

    df["tiny_screentime"] = cond_screen & (
        (df["screentime"] < threshold_onscreen_min) | (df["screentime"] > threshold_onscreen_max)
    )
    to_drop_scr = list(df.index[df["tiny_screentime"]])
    for idx in to_drop_scr:
        if (idx + 1) in df.index and df.at[idx + 1, "type"] == "paused" and df.at[idx + 1, "package"] == df.at[idx, "package"]:
            to_drop_scr.append(idx + 1)

    df_clean = df.drop(index=list(set(to_drop_scr))).reset_index(drop=True).drop(
        columns=["ts", "next_type", "next_ts", "screentime", "tiny_screentime", "offscreentime", "Index"], errors="ignore"
    )
    return df_clean


def process_device(events_metadata_df: pd.DataFrame) -> pd.DataFrame:
    df_pckg_pair = package_wise_pairing(events_metadata_df)
    df_chrono_pair = chrono_pair(df_pckg_pair)
    df_time = calc_screentimes(df_chrono_pair)
    df_clean_off = remove_short_pauses(df_time, threshold_offscreen=3)
    df_clean = remove_short_screentimes(df_clean_off, threshold_onscreen_min=3, threshold_onscreen_max=5 * 60 * 60)
    final_df = calc_screentimes(df_clean)
    return final_df

# ================== Utility ==================
def safe_to_int_ms(x):
    try:
        return int(x)
    except:
        return None

def dedupe_rows(df_raw: pd.DataFrame, gap_seconds: int = 10) -> pd.DataFrame:
    df=df_raw.sort_values(by="timestamp", ascending=False)
    df['timestamp'] = df['timestamp'].astype(int)
    filtered_rows = []
    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]
        time_difference = (next_row['timestamp'] - current_row['timestamp']) / 1000
        if abs(time_difference) >= gap_seconds:
            filtered_rows.append(current_row)
    filtered_rows.append(df.iloc[-1])
    return pd.DataFrame(filtered_rows).reset_index(drop=True)

def explode_events(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_raw.iterrows():
        dev = r.get("device_id")
        ev = r.get("events")
        if ev is None:
            continue
        if isinstance(ev, str):
            try:
                ev = json.loads(ev)
            except Exception:
                continue
        if not isinstance(ev, dict):
            continue
        for pkg, ev_list in ev.items():
            if not isinstance(ev_list, (list, tuple)):
                continue
            for e in ev_list:
                ts = safe_to_int_ms(e.get("timestamp"))
                typ = str(e.get("type", "")).lower()
                if ts is None or not pkg or typ not in {"resumed", "paused"}:
                    continue
                rows.append({"device_id": dev, "timestamp": ts, "package": pkg, "type": typ})
    return pd.DataFrame(rows, columns=["device_id", "timestamp", "package", "type"])

def add_metadata(events_df: pd.DataFrame) -> pd.DataFrame:
    def lookup_app(pkg, max_retries=3):
        retries = 0
        while retries < max_retries:
            try:
                info = gp_app(pkg, lang='en', country='us')
                return pd.Series({
                    'Title': info.get('title'),
                    'Category': info.get('genre'),
                    'Content_Rating': info.get('contentRating'),
                })
            except IncompleteRead:
                retries += 1
                continue
            except Exception as e:
                return pd.Series({'Title': None, 'Category': str(e), 'Content_Rating': str(e)})
        return pd.Series({'Title': None, 'Category': 'Failed after retries', 'Content_Rating': None})

    pkg_list = events_df.package.drop_duplicates().reset_index(drop=True)
    package_lookup = pd.DataFrame({'package': pkg_list})
    metadata = package_lookup['package'].apply(lookup_app)
    package_lookup = pd.concat([package_lookup, metadata], axis=1)
    
    launchable_package_lookup = package_lookup[package_lookup.Title.notna()].reset_index(drop=True)
    events_metadata_df = events_df.merge(
        launchable_package_lookup[['package', 'Title', 'Category', 'Content_Rating']],
        on='package', how='left'
    ).dropna().reset_index(drop=True)

    events_metadata_df['Title'] = events_metadata_df['Title'].replace('WhatsApp Business', 'WhatsApp Messenger')
    events_metadata_df['package'] = events_metadata_df['package'].replace('com.whatsapp.w4b', 'com.whatsapp')

    launcher_packages_df = launchable_package_lookup[launchable_package_lookup['package'].str.contains('launcher', case=False, na=False)]
    UI_launcher_lst = list(launcher_packages_df.package)

    return events_metadata_df[~events_metadata_df.package.isin(UI_launcher_lst)]

# ================== Main Runner ==================
def run():
    print("Fetching last processed timestamps from cleaned_usage_data ...")
    cleaned_res = sb.table("cleaned_usage_data").select("device_id,timestamp", count="exact").execute()
    cleaned_df = pd.DataFrame(cleaned_res.data)
    if not cleaned_df.empty:
        last_processed = cleaned_df.groupby("device_id")["timestamp"].max().to_dict()
    else:
        last_processed = {}

    print("Fetching raw_usage_data from Supabase ...")
    raw = sb.table("usage_data").select("*").execute()
    df_raw = pd.DataFrame(raw.data)
    if df_raw.empty:
        print("No raw data found.")
        return

    # ================= Incremental filter =================
    def is_new(row):
        device_id = row["device_id"]
        ts = row["timestamp"]
        return ts > last_processed.get(device_id, 0)

    df_raw = df_raw[df_raw.apply(is_new, axis=1)]
    if df_raw.empty:
        print("No new usage data to process.")
        return

    print("Deduplicating raw rows ...")
    df_raw = dedupe_rows(df_raw, gap_seconds=10)

    print("Exploding events ...")
    events_df = explode_events(df_raw)
    if events_df.empty:
        print("No events found after explode.")
        return

    print("Fetching Google Play metadata ...")
    events_meta_df = add_metadata(events_df)
    if events_meta_df.empty:
        print("No launchable app events after metadata join.")
        return

    sparse_drop = ['0646964ca120f0a3', '0d466f06bb52e4a4', '2d837289c766497c', '7931a7138e8730d9']
    events_meta_df = events_meta_df[~events_meta_df["device_id"].isin(sparse_drop)].reset_index(drop=True)

    print("Pairing events and computing screentimes ...")
    all_cleaned = []
    for device_id in events_meta_df['device_id'].unique():
        print(f"Processing device_id: {device_id}")
        dev_df = events_meta_df[events_meta_df.device_id == device_id].copy()
        summary = process_device(dev_df)
        summary['device_id'] = device_id
        all_cleaned.append(summary)

    final_df = pd.concat(all_cleaned).reset_index(drop=True)

    # Replace NaN, inf, -inf with Python None (safe for Supabase)
    final_df = final_df.replace([np.nan, np.inf, -np.inf], None)

    # Ensure tz-aware timestamps are JSON serializable
    def safe_isoformat(x):
        if x is None or pd.isna(x):
            return None
        if isinstance(x, (pd.Timestamp, datetime.datetime)):
            return x.isoformat()
        return str(x)
    
    for col in ["ts", "next_ts"]:
        if col in final_df.columns:
            final_df[col] = final_df[col].apply(safe_isoformat)

    print(f"Saving {len(final_df)} new cleaned rows to Supabase ...")
    rows = final_df.to_dict(orient="records")
    if rows:
        sb.table("cleaned_usage_data").upsert(rows).execute()
        print("✅ Saved new cleaned data to Supabase.")
    else:
        print("No new cleaned rows to save.")

if __name__ == "__main__":
    run()
