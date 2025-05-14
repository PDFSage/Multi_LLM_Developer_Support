import urllib.request
import zipfile
import io
from pathlib import Path

def main():
    url = "https://files.pythonhosted.org/packages/3c/4c/3889bc332a6c743751eb78a4bada5761e50a8a847ff0e46c1bd23ce12362/openai-1.78.1-py3-none-any.whl"
    data = urllib.request.urlopen(url).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.filename.endswith(".py"):
                path = Path("openai_source") / info.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(path, "wb") as dst:
                    dst.write(src.read())

if __name__ == "__main__":
    main()
