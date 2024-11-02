from ydata_profiling import ProfileReport
import numpy as np
import pandas as pd
import os
from childhope.common.logger import setup_logger

logger = setup_logger("ydata_profiling_EDA")

if __name__=="__main__":
   
    path = os.path.join(os.path.dirname(__file__), "../../data/anonimized_vitals.csv")
    logger.info(f"Reading data from {path}")
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "../../data/anonimized_vitals.csv"))
    
    # Generate the profile report
    logger.info("Generating EDA report")
    profile = ProfileReport(data, title="ChildHope EDA")
    profile.to_file(os.path.join(os.path.dirname(__file__), "eda_report.html"))
    logger.info("EDA report generated successfully and saved to eda_report.html")