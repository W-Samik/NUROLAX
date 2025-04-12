import torchaudio

df = pd.read_csv("C:\Users\pc\Desktop\py_code\hills_prj\torgo_data\data.csv")  # Replace with actual dataframe

for idx, row in df.iterrows():
    try:
        waveform, sr = torchaudio.load(row['file_path'])
        print(f"OK: {row['file_path']}")
    except Exception as e:
        print(f"FAILED: {row['file_path']} | Error: {str(e)}")