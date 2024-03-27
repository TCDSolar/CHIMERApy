import os

from chimerapy.chimera import chimera_legacy
from parfive import Downloader

INPUT_FILES = {
    'aia171': 'https://solarmonitor.org/data/2016/09/22/fits/saia/saia_00171_fd_20160922_103010.fts.gz',
    'aia193': 'https://solarmonitor.org/data/2016/09/22/fits/saia/saia_00193_fd_20160922_103041.fts.gz',
    'aia211': 'https://solarmonitor.org/data/2016/09/22/fits/saia/saia_00211_fd_20160922_103046.fts.gz',
    'hmi_mag': 'https://solarmonitor.org/data/2016/09/22/fits/shmi/shmi_maglc_fd_20160922_094640.fts.gz'
}


def test_chimera(tmp_path):
    files = Downloader.simple_download(INPUT_FILES.values(), path=tmp_path)
    os.chdir(tmp_path)
    chimera_legacy()
