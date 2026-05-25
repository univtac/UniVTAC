from typing import List
from enum import Enum
import numpy as np
import time
from gsmini.msg import Shear


class TactileModality(Enum):
    IMAGE = "image"
    DEPTH = "depth"
    SHEAR = "shear"


class TactileRecorder:
    def __init__(
        self,
        name: str,
        modalities: List[TactileModality],
        init_node=True,
        is_debug=False,
    ):
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        from rospy.numpy_msg import numpy_msg
        from rospy_tutorials.msg import Floats

        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.modalities = modalities
        for modality in modalities:
            assert modality in TactileModality
        self.modality_names = [modality.value for modality in modalities]
        if init_node:
            rospy.init_node("tactile_recorder", anonymous=True)

        topics = rospy.get_published_topics()
        topics = [i for sublist in topics for i in sublist]
        for modality in self.modalities:
            modality_name = modality.value
            topic_name = f"/puppet_{name}/gsmini_{modality_name}"
            if topic_name not in topics:
                raise ValueError(f"topic {topic_name} not found")
            setattr(self, f"{modality_name}_obs", None)
            setattr(self, f"{modality_name}_secs", None)
            setattr(self, f"{modality_name}_nsecs", None)
            if modality == TactileModality.IMAGE:
                callback_func = self.obs_cb_image
            elif modality == TactileModality.DEPTH:
                callback_func = self.obs_cb_depth
            elif modality == TactileModality.SHEAR:
                callback_func = self.obs_cb_shear

            if modality_name == "image" or modality_name == "depth":
                rospy.Subscriber(topic_name, Image, callback_func)
            elif modality_name == "shear":
                rospy.Subscriber(topic_name, Shear, callback_func)
            else:
                rospy.Subscriber(topic_name, numpy_msg(Floats), callback_func)

        time.sleep(0.5)

    def tactile_obs_cb(self, tac_modality: TactileModality, data):
        shape, dtype = TactileRecorder.get_modality_shape_dtype(tac_modality)
        if tac_modality == TactileModality.IMAGE:
            img = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
            assert img.shape == shape, print(img.shape)
            assert img.dtype == dtype, print(img.dtype)
            setattr(self, f"{tac_modality.value}_obs", img)
        elif tac_modality == TactileModality.DEPTH:
            img = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
            assert img.shape == shape, print(img.shape)
            assert img.dtype == dtype, print(img.dtype)
            setattr(self, f"{tac_modality.value}_obs", img)
        elif tac_modality == TactileModality.SHEAR:
            data: Shear
            shear = np.array(data.markers).reshape((data.n, data.m, 2)) - np.array(
                data.initial
            ).reshape((data.n, data.m, 2))
            shear = shear.astype(np.float32)
            assert shear.shape == shape, print(shear.shape)
            assert shear.dtype == dtype, print(shear.dtype)
            setattr(self, f"{tac_modality.value}_obs", shear)
        else:
            raise NotImplementedError

        setattr(self, f"{tac_modality.value}_secs", data.header.stamp.secs)
        setattr(self, f"{tac_modality.value}_nsecs", data.header.stamp.nsecs)

    def obs_cb_image(self, data):
        return self.tactile_obs_cb(TactileModality.IMAGE, data)

    def obs_cb_depth(self, data):
        return self.tactile_obs_cb(TactileModality.DEPTH, data)

    def obs_cb_shear(self, data):
        return self.tactile_obs_cb(TactileModality.SHEAR, data)

    def get_obs(self):
        obs_dict = dict()
        for modality in self.modalities:
            modality_name = modality.value
            obs_dict[modality_name] = getattr(self, f"{modality_name}_obs")
        return obs_dict

    @staticmethod
    def get_modality_shape_dtype(tac_modality: TactileModality):
        if tac_modality == TactileModality.IMAGE:
            return (240, 320, 3), np.uint8
        elif tac_modality == TactileModality.DEPTH:
            return (240, 320), np.float32
        elif tac_modality == TactileModality.SHEAR:
            return (7, 9, 2), np.float32
        else:
            raise NotImplementedError


if __name__ == "__main__":
    recorder = TactileRecorder("left", [TactileModality.IMAGE, TactileModality.DEPTH])
    for key, value in recorder.get_obs().items():
        print(key, value.shape)
