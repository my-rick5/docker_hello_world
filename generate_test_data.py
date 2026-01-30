import pandas as pd
import random

def create_mock_csv(filename="test_data.csv", num_rows=20):
    data = []
    for i in range(num_rows):
        sqft = random.randint(500, 4500)
        # Simple rule: expensive if over 2200 sqft
        is_expensive = 1 if sqft > 2200 else 0
        data.append({
            'id': i,  # Our Primary Key
            'sqft': sqft,
            'is_expensive': is_expensive
        })
    
    # Add a couple of duplicates with the same IDs but maybe different data
    # This tests if your 'keep="last"' logic works
    data.append({'id': 1, 'sqft': 9999, 'is_expensive': 1}) 
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created {filename} with {len(df)} rows (including duplicates).")

if __name__ == "__main__":
    create_mock_csv()
