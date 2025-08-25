import PySpin
from typing import Optional


class CameraDriver:
    def __init__(self,
                 frame_rate: float = 100,
                 exposure_time: float = 5000,
                 serial_primary: str = "24048471") -> None:
        
        self.system = None
        self.cam_list = None
        self.num_cams = None

        #constants
        self.ACQUISITION_FRAME_RATE = frame_rate        # Check limit of camera specs
        self.AUTOEXPOSUREUPPERLIMIT = exposure_time     # Must be less than frame rate (otherwise bottleneck)
        self.SERIAL_PRIMARY = serial_primary            # Primary camera serial number

    def initialize_cameras(self) -> None:
        try:
            self.system: PySpin.System = PySpin.System.GetInstance()
            self._pyspin_cam_list: PySpin.CameraList = self.system.GetCameras()  # PySpin CameraList object
            self.cam_list: list[PySpin.Camera] = list(self._pyspin_cam_list)     # Pythonic list of PySpin Cameras
            self.num_cams: int = len(self._pyspin_cam_list)

            if self.num_cams == 0:
                print('\tNo devices detected.')
                quit()

            # Print details
            for i, cam in enumerate(self.cam_list):
                device_vendor_name = cam.TLDevice.DeviceVendorName.ToString()
                device_model_name = cam.TLDevice.DeviceModelName.GetValue()
                device_serial = cam.TLDevice.DeviceSerialNumber.ToString()
                print(f"\tDevice {i}: {device_vendor_name} {device_model_name} {device_serial}")
            del cam   # Recommended to delete this cam pointer after loop

            print(f"\t{self.num_cams} cams")
            
        except PySpin.SpinnakerException as ex:
            print(f"Error initializing cameras: {ex}")

    def set_config_all(self) -> None:
        try:
            if self.num_cams == 0:
                print("\tNo devices detected.")
                quit()

            # Common parameters e.g., image settings, acquisition settings
            for cam in self.cam_list:
                # Load default user settings
                cam.Init()

                cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)  # Reset to default settings first
                cam.UserSetLoad()

                cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
                cam.AutoExposureExposureTimeUpperLimit.SetValue(self.AUTOEXPOSUREUPPERLIMIT) # Clunky setting but needed... supposedly
                
                cam.AcquisitionFrameRateEnable.SetValue(True)
                cam.AcquisitionFrameRate.SetValue(self.ACQUISITION_FRAME_RATE)

                cam.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_NewestFirst)  # Can be changed
            del cam

        except PySpin.SpinnakerException as ex:
            print(f"Error setting general config: {ex}")

    def set_config_sync_two(self) -> None:
        try:
            if self.num_cams != 2:
                print("Cannot run sync config, need two cameras")
                self.release_all()
                quit()
            
            if self.cam_list[0].TLDevice.DeviceSerialNumber.ToString() != self.SERIAL_PRIMARY:  # Swap to correct order (want primary first)
                self.cam_list[1], self.cam_list[0] = self.cam_list[0], self.cam_list[1]
        
            # Config of primary according to 
            # https://www.teledynevisionsolutions.com/support/support-center/application-note/iis/configuring-synchronized-capture-with-multiple-cameras/
            self.cam_list[0].LineSelector.SetValue(PySpin.LineSelector_Line1)
            self.cam_list[0].LineMode.SetValue(PySpin.LineMode_Output)

            self.cam_list[0].LineSelector.SetValue(PySpin.LineSelector_Line2)
            self.cam_list[0].V3_3Enable.SetValue(True)

            #Config of secondary
            self.cam_list[1].TriggerSource.SetValue(PySpin.TriggerSource_Line3)
            self.cam_list[1].TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
            self.cam_list[1].TriggerMode.SetValue(PySpin.TriggerMode_On)

        except PySpin.SpinnakerException as ex:
            print(f"Error setting sync config: {ex}")

    def release_all(self) -> None:
        try:
            if self._pyspin_cam_list:
                for cam in self._pyspin_cam_list:
                    if cam.IsStreaming():
                        cam.EndAcquisition()
                    if cam.IsInitialized():
                        cam.DeInit()
                    del cam
                self.cam_list.clear()
                del self.cam_list
                
                self._pyspin_cam_list.Clear()
                del self._pyspin_cam_list
            if self.system:
                self.system.ReleaseInstance()
                del self.system
            print("\tReleased all camera resources")

        except PySpin.SpinnakerException as ex:
            print(f"Error releasing: {ex}")
    
    @staticmethod
    def get_resolution(cam: PySpin.Camera) -> Optional[tuple[int, int]]:
        try:
            return cam.Width.GetValue(), cam.Height.GetValue()
        except PySpin.SpinnakerException as ex:
            print(f"Error getting resolution: {ex}")
            return None, None
        
    @staticmethod
    def get_resolution_list(cam_list: list[PySpin.Camera]) -> list[tuple[int, int]]:
        list = []
        for cam in cam_list:
            res = CameraDriver.get_resolution(cam)
            if res:
                list.append(res)
        return list