import urllib.request
import tarfile
import io
from pathlib import Path

def main():
    url = "https://files.pythonhosted.org/packages/f4/19/da5a085ce419c33b9e6ae308005efad9bfa1b10f59f449d075bba1f16a64/google_genai-1.15.0.tar.gz"
    data = urllib.request.urlopen(url).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        for member in t.getmembers():
            if member.isfile() and member.name.endswith(".py"):
                path = Path("gemini_source") / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with t.extractfile(member) as src, open(path, "wb") as dst:
                    dst.write(src.read())

if __name__ == "__main__":
    main()
