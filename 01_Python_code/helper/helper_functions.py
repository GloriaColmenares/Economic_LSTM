# Standard library imports
import logging
import tempfile
import warnings
from pathlib import Path
from datetime import datetime

# Third-party imports
import requests
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


# Folder and path handling functions
# _____________________________________________________________________________________________

def create_timestamped_folder(output_path=None):
    """
    Create a folder with current date and time as its name.
    
    Args:
        output_path (str or Path, optional): Base directory where the folder will be created.
            If None, creates folder in current working directory.
    
    Returns:
        Path: Path object pointing to the created folder
    """
    # Get current date and time formatted as YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base path if not provided
    if output_path is None:
        output_path = Path.cwd()
    else:
        output_path = Path(output_path)
    
    # Ensure the base output path exists
    ensure_path_exists(output_path)

    # Create the full folder path
    folder_path = output_path / timestamp
    
    # Create the folder
    folder_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output folder: {folder_path}")
    return folder_path


def ensure_path_exists(path):
    """
    Checks if a directory path exists and creates it if not.
    
    Args:
        path (str or Path): The directory path to check/create
        
    Returns:
        Path: The Path object of the now-existing directory
    """
    try:
        # Convert to Path object if it's a string
        if isinstance(path, str):
            path = Path(path)
        
        # Check if path exists and is a directory
        if path.exists():
            if not path.is_dir():
                logger.warning(f"Path exists but is not a directory: {path}")
                return path
            logger.debug(f"Directory already exists: {path}")
        else:
            # Create directory and all parent directories
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
        
        return path
    
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


# data_reading helper functions
# _____________________________________________________________________________________________

def download_yearly_file(year, temp_path, base_url, file_list):
    """Helper function to download a yearly file"""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    file_name = f"RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_{start_date}_{end_date}.xlsx"
    direct_url = f"{base_url}/{file_name}"
    
    logger.info(f"Attempting to download yearly file: {direct_url}")
    
    response = requests.get(direct_url)
    
    if response.status_code == 200 and is_valid_excel(response.content):
        file_path = temp_path / file_name
        with open(file_path, 'wb') as f:
            f.write(response.content)
        file_list.append(file_path)
        logger.info(f"Successfully downloaded yearly file for {year}")
        return True
    else:
        logger.warning(f"Failed to download yearly data for {year}: HTTP {response.status_code}")
        return False


def download_monthly_files(year, temp_path, base_url, file_list):
    """Helper function to download monthly files"""
    logger.info(f"Attempting to download monthly files for {year}")
    
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # Determine months to download
    end_month = current_month if int(year) == current_year else 12
    success = False
    
    for month in range(1, end_month + 1):
        # Determine last day of month
        if int(year) == current_year and month == current_month:
            last_day = datetime.now().day
        elif month in [4, 6, 9, 11]:
            last_day = 30
        elif month == 2:
            # Simple leap year check
            last_day = 29 if (int(year) % 4 == 0 and (int(year) % 100 != 0 or int(year) % 400 == 0)) else 28
        else:
            last_day = 31
            
        monthly_start = f"{year}-{month:02d}-01"
        monthly_end = f"{year}-{month:02d}-{last_day:02d}"
        monthly_file = f"RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_{monthly_start}_{monthly_end}.xlsx"
        monthly_url = f"{base_url}/{monthly_file}"
        
        logger.info(f"Attempting to download monthly file: {monthly_url}")
        monthly_response = requests.get(monthly_url)
        
        if monthly_response.status_code == 200 and is_valid_excel(monthly_response.content):
            monthly_path = temp_path / monthly_file
            with open(monthly_path, 'wb') as f:
                f.write(monthly_response.content)
            file_list.append(monthly_path)
            logger.info(f"Successfully downloaded monthly file for {monthly_start} to {monthly_end}")
            success = True
        else:
            logger.warning(f"Failed to download monthly file for {monthly_start} to {monthly_end}: HTTP {monthly_response.status_code}")
    
    return success


def is_valid_excel(content):
    """Check if the content is a valid Excel file"""
    # Simple check: Excel files start with specific magic bytes
    excel_signatures = [
        b'PK\x03\x04',  # .xlsx (zip file)
        b'\xd0\xcf\x11\xe0',  # .xls (OLE file)
    ]
    
    for sig in excel_signatures:
        if content.startswith(sig):
            return True
    
    # If the content is HTML, it's likely an error page
    if b'<!DOCTYPE html>' in content[:100] or b'<html' in content[:100]:
        logger.warning("Received HTML content instead of Excel file")
        return False
    
    return False


def read_excel_files(file_list):
    dataframes = []
    for file in file_list:
        try:
            logger.info(f"Reading file: {file}")
            excel_file = pd.read_excel(file, sheet_name=None)
            for sheet_name, df in excel_file.items():
                dataframes.append(df)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
