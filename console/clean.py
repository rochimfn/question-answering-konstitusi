#! /usr/bin/env python
import shutil
from pathlib import Path
from sys import argv


def main():
    """
    usage:   clean.py [-h] [-y]

    optional arguments:
        -h, --help
        -y, -Y      Delete all directory and file in .cache (except .gitignore)
    """

    delete = False
    if len(argv) == 2:
        if any((argv[1] == '-h', argv[1] == '--help')):
            print(main.__doc__)
        if any((argv[1] == '-y', argv[1] == '-Y')):
            delete = True
    elif len(argv) == 1:
        confirmation = input('Apakah anda yakin untuk menghapus seluruh cache? [N/y] ')
        if any((confirmation == 'y',
                confirmation == 'yes',
                confirmation == 'Y',
                confirmation == 'Yes')):
            delete = True

    if delete:
        cache_path = Path('.cache')
        to_delete = filter(lambda p: p.name != '.gitignore', cache_path.iterdir())
        for path in to_delete:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()
        print('Clean cache complete!')


if __name__ == '__main__':
    main()
