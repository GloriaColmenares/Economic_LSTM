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
from paths import BASE_INPUT_PATH
from helper.helper_functions import download_yearly_file, download_monthly_files, read_excel_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def read_afrr_capacity_market_overview_data(year):
    """
    Download and read aFRR capacity market overview data for a specified year
    using the direct download URL.
    
    Args:
        year: The year to read data for (integer or string)
        
    Returns:
        pandas.DataFrame: The loaded data or empty DataFrame on error
    """    
    try:
        # Base URL for direct downloads
        base_url = "https://www.regelleistung.net/apps/cpp-publisher/api/v1/download/tenders/files"
        
        # Create a temporary directory to store downloaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_list = []
            
            current_year = datetime.now().year
            year_int = int(year)
            
            # For current or future years, use monthly files directly
            if year_int >= current_year:
                success = download_monthly_files(year_int, temp_path, base_url, file_list)
            else:
                # Try yearly file first, fall back to monthly if needed
                success = download_yearly_file(year_int, temp_path, base_url, file_list)
                if not success:
                    success = download_monthly_files(year_int, temp_path, base_url, file_list)
            
            # Read the downloaded files
            if file_list:
                logger.info(f"Attempting to read {len(file_list)} downloaded files")
                return read_excel_files(file_list)
            else:
                logger.warning(f"No files were downloaded for year {year}")
                return pd.DataFrame()
                
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error when downloading aFRR capacity market data for year {year}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing aFRR capacity market data for year {year}: {e}")
        return pd.DataFrame()
    

def get_afrr_data_for_years(start_year, end_year=None):
    """
    Retrieve and combine aFRR capacity market overview data for a specified range of years.
    
    Args:
        start_year (int): The first year to include in the dataset
        end_year (int, optional): The last year to include. If None, uses current year.
    
    Returns:
        pandas.DataFrame: Combined data from specified years, or empty DataFrame on error
    """
    try:
        # If end_year is not specified, use current year
        if end_year is None:
            end_year = datetime.now().year
        
        # Validate inputs
        start_year = int(start_year)
        end_year = int(end_year)
        
        if start_year > end_year:
            logger.error(f"Invalid year range: start_year ({start_year}) > end_year ({end_year})")
            return pd.DataFrame()
            
        logger.info(f"Retrieving aFRR capacity market data for years {start_year} through {end_year}")
        
        # Initialize an empty DataFrame to store all results
        all_data = pd.DataFrame()
        
        # Iterate through the specified years
        for year in range(start_year, end_year + 1):
            logger.info(f"Retrieving data for year {year}")
            
            # Get data for the current year
            year_data = read_afrr_capacity_market_overview_data(year)
            
            # If we got data back, add a year column and append to the result
            if not year_data.empty:
                # Add year as a column if not already present
                if 'Year' not in year_data.columns:
                    year_data['Year'] = year
                
                # Append to the combined DataFrame
                logger.info(f"Adding {len(year_data)} rows from year {year}")
                all_data = pd.concat([all_data, year_data], ignore_index=True)
            else:
                logger.warning(f"No data available for year {year}")
        
        logger.info(f"Total combined dataset has {len(all_data)} rows from {start_year} to {end_year}")
        return all_data
    
    except Exception as e:
        logger.error(f"Error combining aFRR data for years {start_year}-{end_year}: {e}")
        return pd.DataFrame()


def read_exogenous_factors_data():
    """
    Reads exogenous factors data from a predefined Excel file and processes it.

    The function loads data from a specific sheet in the Excel file, renames columns,
    and prepares the data for further analysis.

    Returns:
        pandas.DataFrame: A DataFrame containing processed exogenous factors data.
    """
    # Define the file path and sheet name for the Excel file containing exogenous factors data
    file_path = BASE_INPUT_PATH / 'aFRR_market_and_exogenous_factors_20190101_20250228.xlsx'
    sheet_name = '4-hourly'
    
    # Read the specified sheet from the Excel file, skipping the first two rows
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=2, engine="openpyxl")
    
    # Check if the 'Date' column exists in the DataFrame
    if 'Date' in df.columns:
        # Drop the 'Date' column as it is not needed
        df = df.drop(columns=['Date'])
    else:
        # Log a warning if the 'Date' column is not found and return an empty DataFrame
        logger.warning("'Date' column not found in the DataFrame. Skipping drop operation.")
        return pd.DataFrame()
    
    # Define the initial column name to be processed
    init_col_name = 'Excess capacity.1'
    if init_col_name in df.columns:
        # Create a new DataFrame with only the relevant columns
        df = pd.DataFrame(df[['4-hour time slice', init_col_name]])
        
        # Rename the column to a more descriptive name
        col_name = 'Cumulated capacity more expensive than electricity price - Excess capacity'
        df = df.rename(columns={init_col_name: col_name})
    else:
        # Log a warning if the initial column is not found and return an empty DataFrame
        logger.warning(f"'{init_col_name}' column not found in the DataFrame. Returning an empty DataFrame.")
        return pd.DataFrame()
    
    # Rename the '4-hour time slice' column to 'PRODUCT' for consistency
    df = df.rename(columns={'4-hour time slice': 'PRODUCT'})
