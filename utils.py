from pathlib import Path
from os import listdir
from os.path import isfile, join
import htmllistparse
import requests


def file_names(url_or_local, extension):
    is_local = Path(url_or_local).exists() and Path(url_or_local).is_dir()
    is_url = is_local == False

    if is_url:
        try:
            cwd, listing = htmllistparse.fetch_listing(url_or_local)
        except requests.exceptions.HTTPError as err:
            status_code = err.response.status_code
            print(status_code)
            return [], []
        else:
            names = []
            for dir in listing:
                name = dir.name
                if name.endswith(extension):
                    names.append(name)
            return cwd, sorted(names)


    # remove current dir from the subdir list
    if is_local:
        cwd = Path(url_or_local)
        cwd_name = cwd.name
        names = []
        files = [f for f in listdir(url_or_local) if isfile(join(url_or_local, f))]
        for file in files:
            name = Path(file).name
            if name.endswith(extension):
                names.append(name)
        return cwd, sorted(names)
