import fastf1
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
logging.getLogger("fastf1").propagate = False

CACHE_DIR = './cache'
YEARS = [2022, 2023, 2024, 2025]

DROP_YEAR = 2025
DROP_EVENT = 'Miami Grand Prix'


def fetch_data():
    all_laps, all_telemetry, all_weather = [], [], []
    today = datetime.now(timezone.utc).date()

    for year in YEARS:
        schedule = fastf1.get_event_schedule(year)
        schedule['Session5DateUtc'] = pd.to_datetime(schedule['Session5DateUtc'])
        filtered = schedule[
            schedule['EventName'].notnull() &
            schedule['EventName'].str.contains('Grand Prix') &
            (schedule['Session5DateUtc'].dt.date < today)
        ]

        events = list(filtered[['EventName', 'Session5DateUtc']].itertuples(index=False, name=None))

        for gp, session_date in events:
            # print with left-aligned padding
            print(f"Fetching data for the {gp:<28} {session_date.date()}")
            session = fastf1.get_session(year, gp, 'R')
            try:
                session.load(telemetry=True, weather=True)
            except Exception as e:
                print(f"Could not load {year} {gp}: {e}")
                continue

            # Laps
            laps_df = session.laps.copy()
            laps_df['Year'] = year
            laps_df['EventName'] = gp
            all_laps.append(laps_df)

            # Weather
            try:
                weather_df = session.weather_data.copy()
                weather_df['Year'] = year
                weather_df['EventName'] = gp
                all_weather.append(weather_df)
            except Exception as e:
                print(f"Skipping weather for {year} {gp}: {e}")

            # Telemetry
            telemetry_dfs = []
            for idx, lap in session.laps.iterlaps():
                try:
                    tel = lap.get_car_data().add_track_status()
                    tel['Year'] = year
                    tel['EventName'] = gp
                    tel['Driver'] = lap.Driver
                    tel['LapNumber'] = lap.LapNumber
                    telemetry_dfs.append(tel)
                except Exception as e:
                    print(f"Skipping telemetry for {year} {gp}: {e}")
            if telemetry_dfs:
                telemetry_df = pd.concat(telemetry_dfs, ignore_index=True)
                all_telemetry.append(telemetry_df)

    laps_df = pd.concat(all_laps, ignore_index=True)
    weather_df = pd.concat(all_weather, ignore_index=True)
    telemetry_df = pd.concat(all_telemetry, ignore_index=True)
    return laps_df, telemetry_df, weather_df


def preprocess(laps_df, telemetry_df, weather_df):
    laps_df = laps_df.copy()
    telemetry_df = telemetry_df.copy()
    weather_df = weather_df.copy()
    # LapsTilPit
    laps_df['LapsTilPit'] = (
        laps_df.groupby(['Year', 'EventName', 'Driver', 'Stint'])['LapNumber']
        .transform(lambda x: x.max() - x + 1)
    )

    # Drop Miami 2025 event
    laps_df = laps_df[~((laps_df['Year'] == DROP_YEAR) & (laps_df['EventName'] == DROP_EVENT))]
    telemetry_df = telemetry_df[~((telemetry_df['Year'] == DROP_YEAR) & (telemetry_df['EventName'] == DROP_EVENT))]
    weather_df = weather_df[~((weather_df['Year'] == DROP_YEAR) & (weather_df['EventName'] == DROP_EVENT))]

    timedelta_cols = [
        'Time', 'LapTime',
        'Sector1Time', 'Sector2Time', 'Sector3Time',
        'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
        'LapStartTime'
    ]

    # Convert to timedelta
    for col in timedelta_cols:
        laps_df.loc[:, col] = pd.to_timedelta(laps_df[col], errors='coerce')

    weather_df.loc[:, 'Time'] = pd.to_timedelta(weather_df['Time'], errors='coerce')
    telemetry_df.loc[:, 'Time'] = pd.to_timedelta(telemetry_df['Time'])
    telemetry_df.loc[:, 'SessionTime'] = pd.to_timedelta(telemetry_df['SessionTime'])

    # Coerce telemetry numeric columns to numeric, drop invalid entries
    for col in ['Speed','Throttle','Brake','nGear','DRS','TrackStatus']:
        telemetry_df.loc[:, col] = pd.to_numeric(telemetry_df[col], errors='coerce')
    # Drop any remaining rows with non-numeric telemetry
    telemetry_df = telemetry_df.dropna(subset=['Speed','Throttle','Brake','nGear','DRS','TrackStatus'])

    mask = (
        laps_df['LapTime'].isna() &
        laps_df['Sector1Time'].notna() &
        laps_df['Sector2Time'].notna() &
        laps_df['Sector3Time'].notna()
    )
    # Fill LapTime as sum of sector times
    laps_df.loc[mask, 'LapTime'] = (
        laps_df.loc[mask, 'Sector1Time'] +
        laps_df.loc[mask, 'Sector2Time'] +
        laps_df.loc[mask, 'Sector3Time']
    )

    cols = ['Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime']
    # Fill possible nulls
    mask1 = laps_df['Sector1SessionTime'].isna() & laps_df['Sector1Time'].notna()
    laps_df.loc[mask1, 'Sector1SessionTime'] = laps_df.loc[mask1, 'LapStartTime'] + laps_df.loc[mask1, 'Sector1Time']

    mask2 = laps_df['Sector2SessionTime'].isna() & laps_df['Sector2Time'].notna() & laps_df['Sector1SessionTime'].notnull()
    laps_df.loc[mask2, 'Sector2SessionTime'] = laps_df.loc[mask2, 'Sector1SessionTime'] + laps_df.loc[mask2, 'Sector2Time']

    mask3 = laps_df['Sector3SessionTime'].isna() & laps_df['Sector3Time'].notna() & laps_df['Sector2SessionTime'].notnull()
    laps_df.loc[mask3, 'Sector3SessionTime'] = laps_df.loc[mask3, 'Sector2SessionTime'] + laps_df.loc[mask3, 'Sector3Time']

    laps_df = laps_df.dropna(subset=cols)
    return laps_df, telemetry_df, weather_df

def rpm_change_rate(rpm_series, time_series):
    dt = np.diff(time_series.values.astype('timedelta64[ns]').astype(np.float64) / 1e9)
    drpm = np.diff(rpm_series.values)
    valid = dt > 0
    return np.std(np.abs(drpm[valid] / dt[valid])) if np.any(valid) else np.nan

def aggregate_per_sector(laps_df_copy, telemetry_df, weather_df):
    # Build per-sector DataFrame
    sector_list = []
    for sec_num in [1, 2, 3]:
        sec_df = laps_df_copy.copy()
        sec_df['SectorNumber'] = sec_num
        sec_df['SectorTime'] = sec_df[f'Sector{sec_num}Time']
        sec_df['SectorSessionTime_Start'] = (
            sec_df['LapStartTime'] if sec_num == 1 else sec_df[f'Sector{sec_num-1}SessionTime']
        )
        sec_df['SectorSessionTime_End'] = sec_df[f'Sector{sec_num}SessionTime']
        sector_list.append(sec_df)
    per_sector = pd.concat(sector_list, ignore_index=True)

    # Build sector interval table
    cols_to_copy = [
        'Year', 'EventName', 'Team', 'Driver',
        'Stint', 'LapNumber', 'SectorNumber',
        'SectorTime','SectorSessionTime_Start','SectorSessionTime_End',
        'LapsTilPit', 'Compound', 'TyreLife', 'FreshTyre'
        ]
    sector_intervals = per_sector[cols_to_copy].copy()

    # Merge telemetry with sector intervals
    merged = telemetry_df.merge(
        sector_intervals,
        on=['Year', 'EventName', 'Driver', 'LapNumber'],
        how='right'
    )

    # Filter telemetry to those within the sector's time window
    in_sector = (
        (merged['SessionTime'] >= merged['SectorSessionTime_Start']) &
        (merged['SessionTime'] < merged['SectorSessionTime_End'])
    )
    merged = merged[in_sector]

    # Aggregate telemetry per sector
    group_keys = ['Year', 'EventName', 'Driver', 'LapNumber', 'SectorNumber']
    agg = merged.groupby(group_keys, as_index=False).agg(
        Speed_P10=        ('Speed', lambda x: np.percentile(x,10)),
        RPM_Std=          ('RPM', 'std'),
        Throttle_Median=  ('Throttle', 'median'),
        Throttle_ZeroPct= ('Throttle', lambda x: (x==0).mean()),
        Gear_Range=       ('nGear', lambda x: x.max()-x.min()),
        DRS_ActivePct=    ('DRS', lambda x: x.isin([10, 12, 14]).mean()),   # positive codes per documentation
        TrackStatus_Mean= ('TrackStatus', 'mean'),
        TrackStatus_Mode= ('TrackStatus', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
    )

    # Merge back onto sector_intervals to preserve empty sectors
    agg = sector_intervals.merge(
        agg,
        on=group_keys,
        how='left'
    )

    # Merge weather using asof on sector start time
    agg = agg.sort_values('SectorSessionTime_Start')
    weather_df = weather_df.sort_values('Time')
    agg = pd.merge_asof(
        agg,
        weather_df,
        left_on='SectorSessionTime_Start', right_on='Time',
        by=['Year', 'EventName'],
        direction='backward'
    )
    return agg


def main():
    data_path = 'data/f1_sector_data.pkl'
    fastf1.Cache.enable_cache(CACHE_DIR)
    laps_df, telemetry_df, weather_df = fetch_data()
    print("\nCleaning and preprocessing the data.")
    laps_df, telemetry_df, weather_df = preprocess(laps_df, telemetry_df, weather_df)
    print("\nAggregating and merging telemetry and weather data.")
    per_sector_df = aggregate_per_sector(laps_df, telemetry_df, weather_df)
    print(f"\nSaving data to {data_path}.")
    per_sector_df.to_pickle(data_path)

if __name__ == "__main__":
    main()