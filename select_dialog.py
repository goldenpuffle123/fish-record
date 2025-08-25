from PySide6.QtWidgets import QFileDialog, QApplication
import logging

def get_dir(msg: str = "") -> str:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    folder = QFileDialog.getExistingDirectory(
        caption=msg
    )
    if not folder:
        raise FileNotFoundError("Select dialog cancelled, folder not found")
    return folder

def get_filepath(msg: str = "", dir: str = None, filter: str = None) -> str:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    filepath = QFileDialog.getOpenFileName(
        caption=msg,
        dir=dir,
        filter=filter
    )[0]
    if not filepath:
        raise FileNotFoundError("Select dialog cancelled, file not found")
    return filepath

def get_filepaths(msg: str = "", dir: str = None, filter: str = None, pair: bool = False) -> list[str]:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    filepaths = QFileDialog.getOpenFileNames(
        caption=msg,
        dir=dir,
        filter=filter
    )[0]
    if not filepaths:
        raise FileNotFoundError("Select dialog cancelled, files not found")

    if pair:
        if len(filepaths) == 2:
            if any(x in filepaths[0] for x in ["cam-1", "cam_1"]): # hardcoded!!!
                filepaths.reverse()
        else:
            raise NameError("Pair select error: not pair or wrong naming")
    
    return filepaths