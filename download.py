import tarfile
import zipfile
import sys
import os
import wget
import requests
import pandas as pd
import pickle

os.makedirs("data/", exist_ok=True)
if sys.argv[1] == "physio":
    url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"
    wget.download(url, out="data")
    with tarfile.open("data/set-a.tar.gz", "r:gz") as t:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(t, path="data/physio")

elif sys.argv[1] == "pm25":
    url = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/STMVL-Release.zip"
    urlData = requests.get(url).content
    filename = "data/STMVL-Release.zip"
    with open(filename, mode="wb") as f:
        f.write(urlData)
    with zipfile.ZipFile(filename) as z:
        z.extractall("data/pm25")
        
    def create_normalizer_pm25():
        df = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
            index_col="datetime",
            parse_dates=True,
        )
        test_month = [3, 6, 9, 12]
        for i in test_month:
            df = df[df.index.month != i]
        mean = df.describe().loc["mean"].values
        std = df.describe().loc["std"].values
        path = "./data/pm25/pm25_meanstd.pk"
        with open(path, "wb") as f:
            pickle.dump([mean, std], f)
    create_normalizer_pm25()
