import polars as pl
import pandas as pd
from pathlib import Path

MENTION_COLS = ["GlobalEventID", "EventTimeDate", "MentionTimeDate", "MentionType", "MentionSourceName", "MentionIdentifier", "SentenceID", "Actor1CharOffset", "Actor2CharOffset", "ActionCharOffset", "InRawText", "Confidence", "MentionDocLen", "MentionDocTone", "MentionDocTranslationInfo", "Extras"]
EVENT_COLS = ["GlobalEventID", "Day", "MonthYear", "Year", "FractionDate", "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode", "Actor1EthnicCode", "Actor1Religion1Code", 
                "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code", "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode", "Actor2EthnicCode", 
                "Actor2Religion1Code", "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code", "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", 
                "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone", 
                "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID", 
                "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID", 
                "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID", 
                "DATEADDED", "SOURCEURL"]

def merge_events_and_mentions(day, cols_to_keep=None, write=True, drop_missing = False):
    year = day[:4]
    in_dir = Path(f"D:\\gdelt_data\\{year}_mentions_output")
    # import 20160101.mentions.parquet
    try:
        daily_mentions_parquet = pl.read_parquet(in_dir / f"{day}.mentions.parquet")
    except Exception as e:
        print(f"Error receiving mentions input file: {day}: {e}")
        return

    # import 20160101.export.CSV.zip
    try:
        daily_events = pd.read_csv(f"D:\\gdelt_data\\{year}_events\\{day}.export.CSV.zip", compression="zip", header=None, sep="\t", low_memory=False)
        daily_events_pl = pl.from_pandas(daily_events)
    except Exception as e:
        print(f"Error receiving events input file: {day}: {e}")
        return

    daily_events_pl.columns = EVENT_COLS
    daily_mentions_parquet.columns = MENTION_COLS

    try:
        merged = daily_mentions_parquet.join(
            daily_events_pl,
            left_on="GlobalEventID",
            right_on="GlobalEventID",
            how="inner"
        )
    except Exception as e:
        print(f"Error merging the files: {day}: {e}")
        return

    if cols_to_keep:
        merged = merged.select(cols_to_keep)

    if drop_missing:
        prev_len = len(merged)
        merged = merged.drop_nulls()
        merged = merged.drop_nans()
        print(f"Saved {100 * len(merged)/prev_len} % of the rows")


    if write == "csv":
        merged.write_csv(f"D:\\gdelt_data\\{year}_merged\\merged{day}.csv")
        return
    elif write == "parquet":
        merged.write_parquet(f"D:\\gdelt_data\\{year}_merged\\merged{day}.csv")
        return
    else:
        return merged