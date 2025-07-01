import argparse
import os
from huggingface_hub import snapshot_download
import zipfile


def download(args):
    try:
        path = os.path.join(args.path, "VB-DemandEx")
        os.mkdir(path)
    except FileExistsError:
        pass
    print("Downloading dataset...")
    snapshot_download(repo_id="NikolaiKyhne/VB-DemandEx", repo_type="dataset", local_dir=rf"{path}")
    print("Dataset downloaded successfully")

def extract(args):

    path = os.path.join(args.path, "VB-DemandEx")
    print(path)

    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)

        if zipfile.is_zipfile(file_path):
            print(f"Extracting {file_name} in {path}")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path)
            os.remove(file_path)  # Optional: delete the zip file after extraction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=r'')
    args = parser.parse_args()
    
    download(args)
    extract(args)

main()
