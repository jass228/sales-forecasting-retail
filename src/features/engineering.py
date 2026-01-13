"""
@author: Joseph A.
Description: Feature engineering module for sales data.
"""
import pandas as pd

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Create temporal features from the 'date' column.

    Args:
        df (pd.DataFrame): DataFrame containing a 'date' column.

    Returns:
        pd.DataFrame: DataFrame with temporal features added.
    """
    df = df.copy()

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_month'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek # Monday=0, Sunday=6
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter

    return df

def create_historical_features(
    df: pd.DataFrame,
    historical_means: dict[str, pd.DataFrame] | None = None
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """ Create historical mean features.

    Args:
        df (pd.DataFrame): DataFrame with columns 'agency', 'sku', 'date', and 'volume'.
        historical_means (dict[str, pd.DataFrame] | None, optional): Dictionary of historical means. 
        Defaults to None.

    Returns:
        tuple[pd.DataFrame, dict[str, pd.DataFrame]]: DataFrame with historical features added 
        and dictionary of historical means.
    """

    df = df.copy()
    if historical_means is None:
        historical_means = {}

        # Average by agency/sku/month
        historical_means['by_agency_sku_month'] = (
            df.groupby(['agency', 'sku', 'month'])['volume']
            .mean()
            .reset_index()
            .rename(columns={'volume': 'mean_volume_agency_sku_month'})
        )

        # Average by agency/sku
        historical_means['by_agency_sku'] = (
            df.groupby(['agency', 'sku'])['volume']
            .mean()
            .reset_index()
            .rename(columns={'volume': 'mean_volume_agency_sku'})
        )

        # Average by sku/month (seasonality patterns)
        historical_means['by_sku_month'] = (
            df.groupby(['sku', 'month'])['volume']
            .mean()
            .reset_index()
            .rename(columns={'volume': 'mean_volume_sku_month'})
        )

    # Merging historical means
    df = df.merge(
        historical_means['by_agency_sku_month'],
        on=['agency', 'sku', 'month'],
        how='left'
    )

    df = df.merge(
        historical_means['by_agency_sku'],
        on=['agency', 'sku'],
        how='left'
    )

    df = df.merge(
        historical_means['by_sku_month'],
        on=['sku', 'month'],
        how='left'
    )

    return df, historical_means

def encode_categorical_features(
    df: pd.DataFrame,
    encoders: dict[str, dict] | None = None
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """ Encode categorical features 'agency' and 'sku' into integers.

    Args:
        df (pd.DataFrame): DataFrame with columns 'agency' and 'sku'.
        encoders (dict[str, dict] | None, optional): Dictionary of encoders. Defaults to None.

    Returns:
        tuple[pd.DataFrame, dict[str, dict]]: DataFrame with encoded categorical features 
        and dictionary of encoders.
    """
    df = df.copy()

    if encoders is None:
        encoders = {}

        # Encoding 'agency' -> integer
        agencies = df['agency'].unique()
        encoders['agency'] = {agency: idx for idx, agency in enumerate(agencies)}

        # Encoding 'sku' -> integer
        skus = df['sku'].unique()
        encoders['sku'] = {sku: idx for idx, sku in enumerate(skus)}

    # Apply encodings
    df['agency_encoded'] = df['agency'].map(encoders['agency'])
    df['sku_encoded'] = df['sku'].map(encoders['sku'])

    return df, encoders

def prepare_features(
    df: pd.DataFrame,
    historical_means: dict | None = None,
    encoders: dict | None = None
) -> tuple[pd.DataFrame, dict, dict]:
    """ Pipeline to prepare features for modeling.

    Args:
        df (pd.DataFrame): Raw DataFrame.
        historical_means (dict | None, optional): Dictionary of historical means. Defaults to None.
        encoders (dict | None, optional): Dictionary of encoders. Defaults to None.

    Returns:
        tuple[pd.DataFrame, dict, dict]: DataFrame with prepared features,
        historical means, and encoders.
    """
    # 1. Create temporal features
    df = create_temporal_features(df)

    # 2. Create historical features
    df, historical_means = create_historical_features(df, historical_means)

    # 3. Encode categorical features
    df, encoders = encode_categorical_features(df, encoders)

    return df, historical_means, encoders

def get_feature_columns() -> list[str]:
    """ Return the list of feature column names.

    Returns:
        list[str]: List of feature column names.
    """
    return [
        # Identifiers encoded
        'agency_encoded',
        'sku_encoded',
        # Temporal features
        'year',
        'month',
        'day',
        'day_of_week',
        'week_of_year',
        'quarter',
        # Historical features
        'mean_volume_agency_sku_month',
        'mean_volume_agency_sku',
        'mean_volume_sku_month',
        # Dataset features (already present in the dataset)
        'avg_max_temp',
        'price_actual',
        'discount_in_percent',
    ]

def get_target_column() -> str:
    """ Return the target column name.

    Returns:
        str: Target column name.
    """
    return 'volume'
