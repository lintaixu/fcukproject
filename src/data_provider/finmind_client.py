import os
import requests
import pandas as pd
from typing import Optional, Dict, Any

class FinMindClient:
    BASE_URL = "https://api.finmindtrade.com/api/v4"
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("FINMIND_TOKEN")
        if not self.token:
            raise ValueError("FINMIND_TOKEN environment variable not set.")
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def get_data(self, dataset: str, data_id: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        url = f"{self.BASE_URL}/data"
        params = {
            "dataset": dataset,
            "data_id": data_id,
            "start_date": start_date,
        }
        if end_date:
            params["end_date"] = end_date
            
        resp = requests.get(url, params=params, headers=self.headers)
        data = resp.json()
        
        if data.get("status") != 200:
            print(f"Error fetching {dataset} for {data_id}: {data.get('msg', 'Unknown error')}")
            return pd.DataFrame()
            
        df = pd.DataFrame(data["data"])
        return df

    def get_taiwan_stock_info(self) -> pd.DataFrame:
        return self.get_data("TaiwanStockInfo", "", "2000-01-01")

    def get_price(self, data_id: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        return self.get_data("TaiwanStockPrice", data_id, start_date, end_date)
