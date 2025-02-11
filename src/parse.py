import pandas as pd

def parse_data(data_path):
    # Load data
    df = pd.read_csv(data_path)
    
    # Convert ApplicationResult to binary
    def convert_to_binary(result):
        if pd.isna(result):
            return 0
        return 1 if 'PASSED' in result.upper() else 0
    
    # Create binary features
    df['HasCriminalRecord'] = (
        (df['CriminalFederalCount'] > 0) | 
        (df['CriminalFelonyCount'] > 0) | 
        (df['CriminalMisdemeanorCount'] > 0) | 
        (df['Failed_Criminal'] == 1)
    ).astype(int)
    
    df['HasEvictionHistory'] = (
        (df['EvictionCount'] > 0) | 
        (df['Failed_Eviction'] == 1)
    ).astype(int)
    
    # Fill NaN values with 0 or median
    df['FICOScore'] = df['FICOScore'].fillna(df['FICOScore'].median())
    df['AssetMonthlyValue'] = df['AssetMonthlyValue'].fillna(0)
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(0)
    df['RentToIncomeRatio'] = df['RentToIncomeRatio'].fillna(df['RentToIncomeRatio'].median())
    
    df['ApplicationResult'] = df['ApplicationResult'].apply(convert_to_binary)
    
    # Extract relevant columns
    values = df[[
        'MonthlyIncome',
        'FICOScore', 
        'RentToIncomeRatio',
        'HasCriminalRecord',
        'HasEvictionHistory',
        'AssetMonthlyValue',
        'ApplicationResult'
    ]].values.tolist()
    
    return values