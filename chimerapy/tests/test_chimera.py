import os
from pathlib import Path

from parfive import Downloader

from chimerapy.chimera import chimera_legacy

INPUT_FILES = {
    "aia171": "https://solarmonitor.org/data/2016/09/22/fits/saia/saia_00171_fd_20160922_103010.fts.gz",
    "aia193": "https://solarmonitor.org/data/2016/09/22/fits/saia/saia_00193_fd_20160922_103041.fts.gz",
    "aia211": "https://solarmonitor.org/data/2016/09/22/fits/saia/saia_00211_fd_20160922_103046.fts.gz",
    "hmi_mag": "https://solarmonitor.org/data/2016/09/22/fits/shmi/shmi_maglc_fd_20160922_094640.fts.gz",
}


def test_chimera(tmp_path):
    Downloader.simple_download(INPUT_FILES.values(), path=tmp_path)
    os.chdir(tmp_path)
    chimera_legacy()

    test_summary_file = Path(__file__).parent / 'test_ch_summary.txt'
    with test_summary_file.open('r') as f:
        test_summary_text = f.read()

    ch_summary_file = tmp_path / "ch_summary.txt"
    with ch_summary_file.open("r") as f:
        ch_summary_text = f.read()

    assert ch_summary_text == test_summary_text
