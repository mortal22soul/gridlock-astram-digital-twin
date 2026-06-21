import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardized feature engineering pipeline for ASTraM.
    Used consistently across model training notebooks and the Streamlit app.
    """
    # 1. Parse dates and sort chronologically
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], utc=True)
    df = df.dropna(subset=['start_datetime'])
    df = df.sort_values('start_datetime').reset_index(drop=True)

    # 2. Extract temporal features
    df['hour'] = df['start_datetime'].dt.hour
    df['dayofweek'] = df['start_datetime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Initialize columns
    df['corridor_count_1d'] = 0.0
    df['corridor_count_7d'] = 0.0
    df['corridor_count_30d'] = 0.0

    df['corridor'] = df['corridor'].fillna('Unknown')
    
    # Robustly compute rolling counts per group and assign back via original integer index
    # to perfectly prevent index-scrambling (which occurs if simply resetting index from groupby)
    # Use 'id' for rolling count if it exists, otherwise use an arbitrary column or create a dummy one
    count_col = 'id' if 'id' in df.columns else 'corridor'
    
    for corridor, group in df.groupby('corridor', dropna=False):
        group_time_idx = group.set_index('start_datetime')
        c1d = group_time_idx[count_col].rolling('1D').count().values - 1
        c7d = group_time_idx[count_col].rolling('7D').count().values - 1
        c30d = group_time_idx[count_col].rolling('30D').count().values - 1
        
        df.loc[group.index, 'corridor_count_1d'] = c1d
        df.loc[group.index, 'corridor_count_7d'] = c7d
        df.loc[group.index, 'corridor_count_30d'] = c30d

    # 4. Fill missing values for numerical features explicitly
    df['osm_lanes'] = df['osm_lanes'].fillna(2.0)
    # The dist_to_nearest_road_m defaults to 0.0 or median; fillna with median if NA
    if df['dist_to_nearest_road_m'].isna().any():
        median_val = df['dist_to_nearest_road_m'].median()
        df['dist_to_nearest_road_m'] = df['dist_to_nearest_road_m'].fillna(median_val if pd.notna(median_val) else 0.0)

    # 5. Calculate severity score (used mostly by the app's 3D heatmap, 1-3 scale)
    if 'requires_road_closure' in df.columns:
        conditions = [
            (df['priority'] == 'High') & (df['requires_road_closure'].astype(bool)),
            (df['priority'] == 'High'),
        ]
        df['severity_score'] = np.select(conditions, [3, 2], default=1)

    return df
