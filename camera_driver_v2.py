import PySpin
import typing


class CameraDriver:
    def __init__(self):
        self.system = None
        self.cam_list = None
        self.num_cams = None

        #constants
        self.ACQUISITION_FRAME_RATE = 100
        self.NUM_IMAGES = 5000
        self.AUTOEXPOSUREUPPERLIMIT = 5000
        self.SERIAL_PRIMARY = "24048471"

    def initialize_cameras(self):
        try:
            # Init cams from system
            self.system: PySpin.System = PySpin.System.GetInstance()
            self._pyspin_cam_list: PySpin.CameraList = self.system.GetCameras()
            self.cam_list: list[PySpin.Camera] = list(self._pyspin_cam_list)
            self.num_cams: int = len(self.cam_list)

            if self.num_cams == 0:
                print('\tNo devices detected.')
                quit()

            # Print details
            for i, cam in enumerate(self.cam_list):
                device_vendor_name = cam.TLDevice.DeviceVendorName.ToString()
                device_model_name = cam.TLDevice.DeviceModelName.GetValue()
                device_serial = cam.TLDevice.DeviceSerialNumber.ToString()
                print(f"\tDevice {i}: {device_vendor_name} {device_model_name} {device_serial}")
            del cam

            print(f"\t{self.num_cams} cams")
            
        except PySpin.SpinnakerException as ex:
            print(f"Error initializing cameras: {ex}")
    
    def set_config_all(self):
        try:
            if self.num_cams == 0:
                print("\tNo devices detected.")
                quit()

            # Common parameters e.g., image settings, acquisition settings
            for cam in self.cam_list:
                # Load default user settings
                cam.Init()

                cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
                cam.UserSetLoad()

                cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
                cam.AutoExposureExposureTimeUpperLimit.SetValue(self.AUTOEXPOSUREUPPERLIMIT) # Clunky setting but needed... supposedly
                
                cam.AcquisitionFrameRateEnable.SetValue(True)
                cam.AcquisitionFrameRate.SetValue(self.ACQUISITION_FRAME_RATE)

                cam.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_NewestFirst)
            del cam

        except PySpin.SpinnakerException as ex:
            print(f"Error setting general config: {ex}")

    def set_config_sync_two(self):
        try:
            if self.num_cams != 2:
                print("cannot run sync config, need two cameras")
                self.release_all()
                quit()
            
            if self.cam_list[0].TLDevice.DeviceSerialNumber.ToString() != self.SERIAL_PRIMARY:  # Swap to correct order
                self.cam_list[1], self.cam_list[0] = self.cam_list[0], self.cam_list[1]
        
            # Config of primary according to 
            # https://www.teledynevisionsolutions.com/support/support-center/application-note/iis/configuring-synchronized-capture-with-multiple-cameras/
            self.cam_list[0].LineSelector.SetValue(PySpin.LineSelector_Line1)
            self.cam_list[0].LineMode.SetValue(PySpin.LineMode_Output)

            self.cam_list[0].LineSelector.SetValue(PySpin.LineSelector_Line2)
            self.cam_list[0].V3_3Enable.SetValue(True)
            # self.cam_list[0].TriggerMode.SetValue(PySpin.TriggerMode_On)

            #Config of secondary
            self.cam_list[1].TriggerSource.SetValue(PySpin.TriggerSource_Line3)
            self.cam_list[1].TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
            self.cam_list[1].TriggerMode.SetValue(PySpin.TriggerMode_On)

        except PySpin.SpinnakerException as ex:
            print(f"Error setting sync config: {ex}")
    
    def release_all(self):
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
            print("released all")

        except PySpin.SpinnakerException as ex:
            print(f"Error releasing: {ex}")
    
    @staticmethod
    def get_resolution(cam: PySpin.Camera) -> typing.Optional[tuple[int, int]]:
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