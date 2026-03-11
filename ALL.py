import requests
import pandas as pd

def get_taiwan_stock_categories():
    print("正在獲取上市與上櫃公司產業分類資料...")
    
    # 1. 取得上市公司資料 (TWSE OpenAPI)
    url_twse = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    res_twse = requests.get(url_twse)
    df_twse = pd.DataFrame(res_twse.json())
    
    # 篩選我們需要的欄位並重新命名
    df_twse = df_twse[['公司代號', '公司簡稱', '產業別']]
    df_twse.columns = ['股票代號', '股票名稱', '產業分類']
    df_twse['市場別'] = '上市'

    # 2. 取得上櫃公司資料 (TPEx OpenAPI)
    url_tpex = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
    res_tpex = requests.get(url_tpex)
    df_tpex = pd.DataFrame(res_tpex.json())
    
    # 使用上櫃公司的英文欄位名稱
    # SecuritiesCompanyCode = 公司代號
    # CompanyAbbreviation = 公司簡稱  
    # SecuritiesIndustryCode = 產業代碼（不是 Registration，那是註冊地）
    df_tpex = df_tpex[['SecuritiesCompanyCode', 'CompanyAbbreviation', 'SecuritiesIndustryCode']]
    df_tpex.columns = ['股票代號', '股票名稱', '產業分類']
    df_tpex['市場別'] = '上櫃'

    # 3. 合併上市與上櫃資料
    df_all = pd.concat([df_twse, df_tpex], ignore_index=True)
    
    # 資料清理
    # 去除產業分類前後的空格
    df_all['產業分類'] = df_all['產業分類'].astype(str).str.strip()
    
    # 統計原始資料
    total_count = len(df_all)
    special_count = len(df_all[df_all['產業分類'].isin(['－', ''])])
    
    # 排除沒有產業別的特殊代碼（ETF、特別股等）
    df_all = df_all[~df_all['產業分類'].isin(['－', ''])]
    df_all = df_all.sort_values(by='股票代號').reset_index(drop=True)
    
    print(f"\n資料統計：")
    print(f"  原始總筆數：{total_count}")
    print(f"  特殊股票（無產業分類）：{special_count}")
    print(f"  有效股票筆數：{len(df_all)}")

    return df_all

if __name__ == "__main__":
    # 執行並取得資料表
    stock_df = get_taiwan_stock_categories()
    
    # 顯示前 10 筆確認結果
    print("\n前 10 筆股票資料：")
    print(stock_df.head(10))
    print("-" * 60)
    print(f"總共取得 {len(stock_df)} 檔股票的分類資料！")
    
    # 顯示產業分類統計
    print("\n產業分類統計（前 10 名）：")
    industry_stats = stock_df['產業分類'].value_counts().head(10)
    for code, count in industry_stats.items():
        print(f"  產業代碼 {code}: {count} 檔")
    
    # 顯示市場別統計
    print("\n市場別統計：")
    market_stats = stock_df['市場別'].value_counts()
    for market, count in market_stats.items():
        print(f"  {market}: {count} 檔")
    
    # 將結果輸出成 CSV 檔案 (utf-8-sig 確保在 Excel 或其他系統開啟不會亂碼)
    stock_df.to_csv("taiwan_stock_categories.csv", index=False, encoding="utf-8-sig")
    print("\n✓ 檔案已成功儲存為 taiwan_stock_categories.csv")