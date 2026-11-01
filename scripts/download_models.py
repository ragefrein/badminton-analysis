"""Fetch public pretrained weights only when not already available."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
MODELS = (
    ("yolo11s-pose.pt", "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt",
     "1060bda4a27012060eca246f9b2adeea22eabb045a1e58f8d229be29b7ebc2ba"),
    ("osnet_x0_25_msmt17.pt", "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF",
     "6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99"),
)


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    folder = ROOT/"models"
    folder.mkdir(exist_ok=True)
    for name, url, expected in MODELS:
        target = folder/name
        if target.exists():
            if checksum(target) != expected:
                raise RuntimeError(f"Checksum berbeda untuk {name}; periksa model sebelum digunakan.")
            print(f"OK {name}")
            continue
        temporary = target.with_suffix(".part")
        if "drive.google.com" in url:
            import gdown
            gdown.download(url, str(temporary), quiet=False)
        else:
            urllib.request.urlretrieve(url, temporary)
        if checksum(temporary) != expected:
            raise RuntimeError(f"Checksum unduhan {name} tidak cocok; file tidak diaktifkan.")
        temporary.replace(target)
        print(f"OK {name}")


if __name__ == "__main__":
    main()
