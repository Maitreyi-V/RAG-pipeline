import pandas as pd
df = pd.read_csv("annotations.csv")
print(df.columns.tolist())
print(df[df['video_file'].str.startswith('kannada')].head(5))