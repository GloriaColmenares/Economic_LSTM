# Standard library imports
import logging
import tempfile
import warnings
from pathlib import Path
from datetime import datetime

# Third-party imports
import requests
import pandas as pd

# Local imports
from data.data_reading import get_afrr_data_for_years
from helper.helper_functions import ensure_path_exists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def split_pos_neg_dataframes(df):
    """
    Split the combined aFRR data into separate DataFrames for positive and negative products.
    
    Args:
        df (pandas.DataFrame): The combined DataFrame with a PRODUCT column
        
    Returns:
        tuple: (pos_df, neg_df) - DataFrames containing only positive and negative products
    """
    try:
        if df.empty:
            logger.warning("Cannot split empty DataFrame")
            return pd.DataFrame(), pd.DataFrame()
            
        if 'PRODUCT' not in df.columns:
            logger.error("DataFrame does not contain a 'PRODUCT' column")
            return pd.DataFrame(), pd.DataFrame()
        
        # Create masks for positive and negative products
        pos_mask = df['PRODUCT'].str.startswith('POS_', na=False)
        neg_mask = df['PRODUCT'].str.startswith('NEG_', na=False)
        
        # Split the dataframes
        pos_df = df[pos_mask].copy().reset_index(drop=True)
        neg_df = df[neg_mask].copy().reset_index(drop=True)
        
        # Log results
        logger.info(f"Split DataFrame into positive ({len(pos_df)} rows) and negative ({len(neg_df)} rows) products")
        
        # Check for any rows that didn't match either pattern
        other_rows = len(df) - len(pos_df) - len(neg_df)
        if other_rows > 0:
            logger.warning(f"Found {other_rows} rows that don't match POS_ or NEG_ patterns")
        
        return pos_df, neg_df
        
    except Exception as e:
        logger.error(f"Error splitting DataFrame by product type: {e}")
        return pd.DataFrame(), pd.DataFrame()
    

def add_date_with_hour(df):
    """
    Add a DATE column that combines DATE_FROM with the hour extracted from PRODUCT.
    
    Args:
        df (pandas.DataFrame): DataFrame containing DATE_FROM and PRODUCT columns
        
    Returns:
        pandas.DataFrame: DataFrame with added DATE column
    """
    try:
        # Make a copy of the input DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Check if required columns exist
        if 'DATE_FROM' not in result_df.columns or 'PRODUCT' not in result_df.columns:
            logger.error("DataFrame must contain both DATE_FROM and PRODUCT columns")
            return df
        
        # Extract hour from PRODUCT column using regular expressions
        # Pattern looks for two digits after POS_ or NEG_ and before _
        result_df['HOUR'] = result_df['PRODUCT'].str.extract(r'(?:POS_|NEG_)(\d{2})_\d{2}')
        
        # Convert to integer
        result_df['HOUR'] = pd.to_numeric(result_df['HOUR'], errors='coerce')
        
        # Convert DATE_FROM to datetime if it's not already
        result_df['DATE_FROM'] = pd.to_datetime(result_df['DATE_FROM'])
        
        # Create DATE by adding the extracted hour to DATE_FROM
        result_df['DATE'] = result_df.apply(
            lambda row: row['DATE_FROM'] + pd.Timedelta(hours=int(row['HOUR'])) 
                        if pd.notna(row['HOUR']) else row['DATE_FROM'], 
            axis=1
        )
        
        # Check if all dates were created successfully
        missing_dates = result_df['DATE'].isna().sum()
        if missing_dates > 0:
            logger.warning(f"Could not create DATE for {missing_dates} rows")
        else:
            logger.info(f"Successfully created DATE column for all {len(result_df)} rows")
            
        return result_df
        
    except Exception as e:
        logger.error(f"Error adding DATE with hour: {e}")
        # Return original dataframe on error
        return df
    

def merge_pos_neg_dataframes(pos_df, neg_df):
    """
    Merge positive and negative aFRR DataFrames into a single DataFrame with labeled columns.
    The PRODUCT column will be standardized to remove POS_ and NEG_ prefixes.
    
    Args:
        pos_df (pandas.DataFrame): DataFrame containing positive aFRR data
        neg_df (pandas.DataFrame): DataFrame containing negative aFRR data
        
    Returns:
        pandas.DataFrame: Merged DataFrame with labeled columns
    """
    try:
        # Make copies to avoid modifying originals
        pos_copy = pos_df.copy() if not pos_df.empty else pd.DataFrame()
        neg_copy = neg_df.copy() if not neg_df.empty else pd.DataFrame()
        
        # Check if either DataFrame is empty
        if pos_copy.empty and neg_copy.empty:
            logger.error("Both positive and negative DataFrames are empty")
            return pd.DataFrame()
        elif pos_copy.empty:
            logger.warning("Positive DataFrame is empty, only using negative data")
            return neg_copy
        elif neg_copy.empty:
            logger.warning("Negative DataFrame is empty, only using positive data")
            return pos_copy
            
        # Ensure both DataFrames have a proper index (DATE)
        if not isinstance(pos_copy.index, pd.DatetimeIndex):
            logger.error("Positive DataFrame must have DATE as index")
            return pd.DataFrame()
            
        if not isinstance(neg_copy.index, pd.DatetimeIndex):
            logger.error("Negative DataFrame must have DATE as index")
            return pd.DataFrame()
        
        # Standardize the PRODUCT column - remove POS_ and NEG_ prefixes
        if 'PRODUCT' in pos_copy.columns:
            pos_copy['PRODUCT'] = pos_copy['PRODUCT'].str.replace('POS_', '', regex=False)
            
        if 'PRODUCT' in neg_copy.columns:
            neg_copy['PRODUCT'] = neg_copy['PRODUCT'].str.replace('NEG_', '', regex=False)
        
        # Add prefix to price columns before merging
        price_columns = [col for col in pos_copy.columns if col != 'PRODUCT']
        for col in price_columns:
            pos_copy.rename(columns={col: f"POS_{col}"}, inplace=True)
                
        price_columns = [col for col in neg_copy.columns if col != 'PRODUCT']
        for col in price_columns:
            neg_copy.rename(columns={col: f"NEG_{col}"}, inplace=True)
        
        # Merge DataFrames on the index (DATE)
        # If both have PRODUCT, check if they match and use one
        if 'PRODUCT' in pos_copy.columns and 'PRODUCT' in neg_copy.columns:
            # Check if PRODUCT values match for the same DATE
            product_mismatch = 0
            for idx in set(pos_copy.index).intersection(set(neg_copy.index)):
                if idx in pos_copy.index and idx in neg_copy.index:
                    pos_prod = pos_copy.at[idx, 'PRODUCT'] if not pd.isna(pos_copy.at[idx, 'PRODUCT']) else None
                    neg_prod = neg_copy.at[idx, 'PRODUCT'] if not pd.isna(neg_copy.at[idx, 'PRODUCT']) else None
                    if pos_prod and neg_prod and pos_prod != neg_prod:
                        product_mismatch += 1
                        
            if product_mismatch > 0:
                logger.warning(f"Found {product_mismatch} dates with mismatched PRODUCT values")
            
            # First merge without PRODUCT to avoid conflicts
            pos_without_prod = pos_copy.drop(columns=['PRODUCT'])
            neg_without_prod = neg_copy.drop(columns=['PRODUCT'])
            
            merged_df = pd.merge(
                pos_without_prod,
                neg_without_prod,
                left_index=True,
                right_index=True,
                how='outer'
            )
            
            # Then add PRODUCT column, preferring non-null values from positive data
            merged_df['PRODUCT'] = None
            for idx in merged_df.index:
                pos_prod = pos_copy.at[idx, 'PRODUCT'] if idx in pos_copy.index else None
                neg_prod = neg_copy.at[idx, 'PRODUCT'] if idx in neg_copy.index else None
                merged_df.at[idx, 'PRODUCT'] = pos_prod if pd.notna(pos_prod) else neg_prod
        else:
            # Simple merge if only one or neither has PRODUCT
            merged_df = pd.merge(
                pos_copy,
                neg_copy,
                left_index=True,
                right_index=True,
                how='outer'
            )
        
        # Sort by the index (DATE)
        merged_df.sort_index(inplace=True)
        
        # Move PRODUCT column to the front if it exists
        if 'PRODUCT' in merged_df.columns:
            cols = ['PRODUCT'] + [col for col in merged_df.columns if col != 'PRODUCT']
            merged_df = merged_df[cols]
            
        logger.info(f"Merged DataFrame has {len(merged_df)} rows with columns: {', '.join(merged_df.columns)}")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging positive and negative DataFrames: {e}")
        return pd.DataFrame()


def create_processed_afrr_dataframe(start_year, end_year=None, output_path=None):
    """
    Create the processed aFRR DataFrame with both positive and negative products
    and DATE as index, and optionally save it to a file.
    
    Args:
        start_year (int): The first year to include
        end_year (int, optional): The last year to include. If None, uses current year.
        save_path (str or Path, optional): Path where to save the DataFrame. 
            If None, the DataFrame is not saved to a file.
        
    Returns:
        pandas.DataFrame: Merged DataFrame with labeled columns
    """
    # Get combined data for the specified years
    combined_df = get_afrr_data_for_years(start_year, end_year)
    
    # Split into positive and negative product DataFrames
    pos_df, neg_df = split_pos_neg_dataframes(combined_df)
    
    # Add DATE column to each DataFrame
    pos_df_with_date = add_date_with_hour(pos_df)
    neg_df_with_date = add_date_with_hour(neg_df)
    
    # Prepare DataFrames with DATE as index and only required columns
    columns_to_keep = [
        'DATE', 
        'PRODUCT',
        'GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]',
        'GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]'
    ]
    
    # Filter columns and set index
    pos_filtered = pos_df_with_date[columns_to_keep].set_index('DATE')
    neg_filtered = neg_df_with_date[columns_to_keep].set_index('DATE')
    
    # Merge positive and negative DataFrames
    merged_df = merge_pos_neg_dataframes(pos_filtered, neg_filtered)
    
    # Save the DataFrame if path is provided
    if output_path is not None:
        try:
            # Convert string path to Path object if necessary
            if isinstance(output_path, str):
                output_path = Path(output_path)

            # Ensure output directory exists
            ensure_path_exists(output_path)
            
            file_name = "processed_afrr_data.csv"
            output_file = output_path / file_name
            merged_df.to_csv(output_file)
            logger.info(f"DataFrame saved to CSV (default): {output_file}")
        except Exception as e:
            logger.error(f"Failed to save DataFrame to {output_path}: {e}")

    return merged_df
