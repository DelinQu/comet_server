import time
import logging
import torch as th
from omnigibson.learning.utils.array_tensor_utils import torch_to_numpy
from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
from typing import Optional

class WebsocketPolicy:
    """
    Websocket policy for controlling the robot over a websocket connection.
    """

    def __init__(
        self,
        *args,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs,
    ) -> None:
        logging.info(f"Creating websocket client policy with host: {host}, port: {port}")
        self.policy = WebsocketClientPolicy(host=host, port=port)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        # convert observation to numpy
        obs = torch_to_numpy(obs)
        return self.policy.act(obs).detach().cpu()

    def reset(self) -> None:
        self.policy.reset()


if __name__ == "__main__":
    policy = WebsocketPolicy(host="localhost", port=8000)
    logging.warning("Created websocket policy")

    example = {
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": th.randint(256, size=(224, 224, 3), dtype=th.uint8), # unit 8
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": th.randint(256, size=(224, 224, 3), dtype=th.uint8), # unit 8
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": th.randint(256, size=(224, 224, 3), dtype=th.uint8), # unit 8
        "robot_r1::proprio": th.rand(256),
    }

    task_ids = [
        * [1] * 32, # picking_up_trash
        * [0] * 32, # turning_on_radio
        * [3] * 32, # cleaning_up_plates_and_food
        * [4] * 32, # can_meat
    ]

    for i in range(128):
        task_id = task_ids[i]
        example["task_id"] = th.tensor(task_id)

        start_time = time.time()
        action = policy.forward(example)
        end_time = time.time()
        
        logging.warning(f"iter {i:2d}: {end_time - start_time} sec")
    
    logging.warning(action.shape)
