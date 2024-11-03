from ydata_profiling import ProfileReport
import numpy as np
import pandas as pd
import os
from childhope.common.logger import setup_logger

logger = setup_logger("ydata_profiling_EDA")

if __name__=="__main__":
   
    path = os.path.join(os.path.dirname(__file__), "../../data/anonimized_vitals.csv")
    logger.info(f"Reading data from {path}")
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "../../data/anonimized_vitals.csv"), index_col=0)
    
    # Generate the profile report
    logger.info("Generating EDA reports")
    
    profile = ProfileReport(data, title="General EDA")
    profile.to_file(os.path.join(os.path.dirname(__file__), "general_eda_report.html"))
    
    data = data[data['datetime'].notnull()]
    data['datetime'] = pd.to_datetime(data['datetime'])
    profile_ts = ProfileReport(data, title="Time Series EDA", tsmode=True, sortby='datetime')
    profile_ts.to_file(os.path.join(os.path.dirname(__file__), "ts_eda_report.html"))
    
    logger.info("EDA reports generated successfully and saved to disk.")
    

