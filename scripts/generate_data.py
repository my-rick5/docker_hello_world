import pandas as pd
import numpy as np

# Create 100 rows of realistic housing data
data = {
    'id': range(1, 101),
    'sqft': np.random.randint(800, 4000, 100),
    'price': []
}

# Simple logic: $200 per sqft + some random noise
for s in data['sqft']:
    data['price'].append(int(s * 200 + np.random.normal(0, 10000)))

df = pd.DataFrame(data)
df.to_csv("test_data.csv", index=False)
print("Realistic dataset 'test_data.csv' created!")
