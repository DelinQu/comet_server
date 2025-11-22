import dataclasses
import enum
import logging
import socket
import tyro
import asyncio
import time
import traceback
import websockets.sync.client
import websockets
from copy import deepcopy

from omnigibson.macros import gm
from omnigibson.learning.utils.network_utils import (
    WebsocketPolicyServer,
    Packer,
    unpackb,
)

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config as _config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONFIG = {
    # policy config
    "control_mode": "receeding_horizon",
    "max_len": 32,
    "action_horizon": 5,
    "temporal_ensemble_max": 3,
    "fine_grained_level": 0,
    "mount_base": "/root/models",
    # mapping
    0: {
        "task_name": "turning_on_radio",
        "policy_dir": "2025-11-12-18-33-57_pi05_b1k-turning_on_radio_cs32_bs64_lr2.5e-6_step15k_re_jax/14999",
    },
    1: {
        "task_name": "picking_up_trash",
        "policy_dir": "17-36-51_pi05_b1k-pt12_cs32_bs32_lr2.5e-5_step100k_gpu80_jax/75000",
    },
    2: {
        "task_name": "putting_away_Halloween_decorations",
        "policy_dir": "11-31-51_pi05_b1k-putting_away_Halloween_decorations_cs32_bs32_lr2.5e-6_step15k_sft_pt50_merge_25k_jax/14999",
    },
    3: {
        "task_name": "cleaning_up_plates_and_food",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    4: {
        "task_name": "can_meat",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    5: {
        "task_name": "setting_mousetraps",
        "policy_dir": "03-32-55_pi05_b1k-pt50_merge_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/20000",
    },
    6: {
        "task_name": "hiding_Easter_eggs",
        "policy_dir": "2025-11-16-15-49-17_pi05_b1k-hiding_Easter_eggs_official_rollsamplelower_lr2.5e-6_step20k_sft_resume_jax/19999",
    },
    7: {
        "task_name": "picking_up_toys",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    8: {
        "task_name": "rearranging_kitchen_furniture",
        "policy_dir": "03-32-55_pi05_b1k-pt50_merge_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/45000",
    },
    9: {
        "task_name": "putting_up_Christmas_decorations_inside",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    10: {
        "task_name": "set_up_a_coffee_station_in_your_kitchen",
        "policy_dir": "2025-11-16-15-49-17_pi05_b1k-set_up_a_coffee_station_in_your_kitchen_official_rollsamplelower_lr2.5e-6_step20k_sft_resume_jax/15000",
    },
    11: {
        "task_name": "putting_dishes_away_after_cleaning",
        "policy_dir": "2025-11-16-15-41-28_pi05_b1k-putting_dishes_away_after_cleaning_official_rollsamplehigher_lr2.5e-6_step20k_sft_resume_jax/19999",
    },
    12: {
        "task_name": "preparing_lunch_box",
        "policy_dir": "2025-11-16-15-41-27_pi05_b1k-preparing_lunch_box_official_rollsamplehigher_lr2.5e-6_step20k_sft_resume_jax/19999",
    },
    13: {
        "task_name": "loading_the_car",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    14: {
        "task_name": "carrying_in_groceries",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/35000",
    },
    15: {
        "task_name": "bringing_in_wood",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    16: {
        "task_name": "moving_boxes_to_storage",
        "policy_dir": "2025-11-12-19-00-46_pi05_b1k-pt10_merge1112_cs32_bs64_lr2.5e-5_step50k_gpu160_jax/40000",
    },
    17: {
        "task_name": "bringing_water",
        "policy_dir": "2025-11-12-15-45-53_pi05_b1k-pt10_re-pt12-49k-m_cs32_bs64_lr2.5e-6_step50k_gpu160_jax/45000",
    },
    18: {
        "task_name": "tidying_bedroom",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    19: {
        "task_name": "outfit_a_basic_toolbox",
        "policy_dir": "2025-11-16-15-49-18_pi05_b1k-outfit_a_basic_toolbox_official_rollsamplelower_lr2.5e-6_step20k_sft_resume_jax/15000",
    },
    20: {
        "task_name": "sorting_vegetables",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    21: {
        "task_name": "collecting_childrens_toys",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    22: {
        "task_name": "putting_shoes_on_rack",
        "policy_dir": "17-36-51_pi05_b1k-pt12_cs32_bs32_lr2.5e-5_step100k_gpu80_jax/85000",
    },
    23: {
        "task_name": "boxing_books_up_for_storage",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    24: {
        "task_name": "storing_food",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    25: {
        "task_name": "clearing_food_from_table_into_fridge",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    26: {
        "task_name": "assembling_gift_baskets",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    27: {
        "task_name": "sorting_household_items",
        "policy_dir": "2025-11-12-22-39-15_pi05_b1k-pt50_merge1112_hq_re_cs32_bs64_lr2.5e-6_step50k_gpu400_jax/15000",
    },
    28: {
        "task_name": "getting_organized_for_work",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    29: {
        "task_name": "clean_up_your_desk",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    30: {
        "task_name": "setting_the_fire",
        "policy_dir": "06-20-28_pi05_b1k-pt12_merge_cs32_bs64_lr2.5e-5_step50k_gpu160_jax/40000",
    },
    31: {
        "task_name": "clean_boxing_gloves",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    32: {
        "task_name": "wash_a_baseball_cap",
        "policy_dir": "06-20-28_pi05_b1k-pt12_merge_cs32_bs64_lr2.5e-5_step50k_gpu160_jax/40000",
    },
    33: {
        "task_name": "wash_dog_toys",
        "policy_dir": "11-31-51_pi05_b1k-wash_dog_toys_cs32_bs32_lr2.5e-6_step15k_sft_pt50_merge_25k_jax/14999",
    },
    34: {
        "task_name": "hanging_pictures",
        "policy_dir": "2025-11-15-03-13-35_pi05_b1k-hanging_pictures_roll1114_cs32_bs64_lr2.5e-6_step15k_re-pt50-49k_jax/14999",
    },
    35: {
        "task_name": "attach_a_camera_to_a_tripod",
        "policy_dir": "06-20-28_pi05_b1k-pt12_merge_cs32_bs64_lr2.5e-5_step50k_gpu160_jax/25000",
    },
    36: {
        "task_name": "clean_a_patio",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    37: {
        "task_name": "clean_a_trumpet",
        "policy_dir": "20-36-46_pi05_b1k-clean_a_trumpet_cs32_bs32_lr2.5e-5_step30k_jax/25000",
    },
    38: {
        "task_name": "spraying_for_bugs",
        "policy_dir": "2025-11-16-15-41-30_pi05_b1k-spraying_for_bugs_official_rollsamplehigher_lr2.5e-6_step20k_sft_resume_jax/15000",
    },
    39: {
        "task_name": "spraying_fruit_trees",
        "policy_dir": "06-46-38_pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/49999",
    },
    40: {
        "task_name": "make_microwave_popcorn",
        "policy_dir": "2025-11-10-12-57-44_pi05_b1k-pt7_cs32_bs64_lr2.5e-5_step50k_gpu80_jax/49999",
    },
    41: {
        "task_name": "cook_cabbage",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    42: {
        "task_name": "chop_an_onion",
        "policy_dir": "03-32-55_pi05_b1k-pt50_merge_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/45000",
    },
    43: {
        "task_name": "slicing_vegetables",
        "policy_dir": "2025-11-16-15-41-27_pi05_b1k-slicing_vegetables_official_rollsamplehigher_lr2.5e-6_step20k_sft_resume_jax/19999",
    },
    44: {
        "task_name": "chopping_wood",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    45: {
        "task_name": "cook_hot_dogs",
        "policy_dir": "2025-11-12-16-22-00_pi05_b1k-pt10_merge1112_re-pt12-49k-m_cs32_bs64_lr2.5e-6_step50k_gpu160_jax/40000",
    },
    46: {
        "task_name": "cook_bacon",
        "policy_dir": "06-46-38_pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/49999",
    },
    47: {
        "task_name": "freeze_pies",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/35000",
    },
    48: {
        "task_name": "canning_food",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
    49: {
        "task_name": "make_pizza",
        "policy_dir": "10-00-32_pi05_b1k-pt50_merge1110_cs32_bs64_lr2.5e-5_step50k_gpu400_jax/40000",
    },
}


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


class WebsocketPolicyServerMulti(WebsocketPolicyServer):
    def __init__(self, *args, **kwargs):
        self._task_id = kwargs.pop("task_id", 0)
        self._task_id_lock = asyncio.Lock()
        super().__init__(*args, **kwargs)

    def update_policy(self, task_id: int = 0) -> None:
        policy_dir = f"{CONFIG['mount_base']}/{CONFIG[task_id]['policy_dir']}"
        task_name = CONFIG[task_id]["task_name"]

        if self._policy is not None and (CONFIG[task_id]["policy_dir"] == CONFIG[self._task_id]["policy_dir"]):
            logger.warning(f"*** Update task {task_id} policy from {self._task_id} ***")
            
            pi05_policy = self._policy.policy
            policy_metadata = pi05_policy.metadata

            policy = B1KPolicyWrapper(
                pi05_policy,
                task_name=task_name,
                control_mode=CONFIG["control_mode"],
                max_len=CONFIG["max_len"],
                action_horizon=CONFIG["action_horizon"],
                temporal_ensemble_max=CONFIG["temporal_ensemble_max"],
                fine_grained_level=CONFIG["fine_grained_level"],
            )

        else:
            logger.warning(f"*** Delete task {self._task_id} policy, and create task {task_id} from ckpt {policy_dir} ***")
            
            del self._policy
            
            pi05_policy = _policy_config.create_trained_policy(
                _config.get_config("comet_submission"),
                policy_dir,
                default_prompt=None,
            )
            policy_metadata = pi05_policy.metadata

            policy = B1KPolicyWrapper(
                pi05_policy,
                task_name=task_name,
                control_mode=CONFIG["control_mode"],
                max_len=CONFIG["max_len"],
                action_horizon=CONFIG["action_horizon"],
                temporal_ensemble_max=CONFIG["temporal_ensemble_max"],
                fine_grained_level=CONFIG["fine_grained_level"],
            )

        self._policy = policy
        self._task_id = task_id
        self._metadata = policy_metadata

    async def _handler(self, websocket):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                result = unpackb(await websocket.recv())

                if "reset" in result:
                    self._policy.reset()
                    continue

                # check if policy matches task_id
                task_id = int(result.get("task_id", self._task_id))
                if task_id != self._task_id or self._policy is None:
                    async with self._task_id_lock:
                        self.update_policy(task_id)
                        # await websocket.send(packer.pack(self._metadata))

                obs = deepcopy(result)

                infer_time = time.monotonic()
                action = self._policy.act(obs)
                infer_time = time.monotonic() - infer_time

                action = {
                    "action": action.cpu().numpy(),
                }
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break

            except Exception:
                logger.error(
                    f"Error in connection from {websocket.remote_address}:\n{traceback.format_exc()}"
                )
                if gm.DEBUG:
                    await websocket.send(traceback.format_exc())
                try:
                    # Try new websockets API first
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                except AttributeError:
                    # Fallback for older websockets versions
                    await websocket.close(code=1011, reason="Internal server error")
                raise


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str | None = "comet_submission"

    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000"). Must be specified upon launching
    dir: str | None = None


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # If provided, will be used to retrieve the prompt of the task. Must be specified upon launching
    task_name: str | None = None

    # Port to serve the policy on.
    port: int = 8000

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # Specifies the fine-grained level of the policy.
    fine_grained_level: int = 0

    # Specifies the control mode of the policy.
    control_mode: str = "receeding_horizon"  # receeding_horizon | temporal_ensemble

    # Specifies the action horizon of the policy.
    max_len: int = 32  # receeding horizon | receeding temporal mode

    action_horizon: int = 5  # temporal ensemble mode

    temporal_ensemble_max: int = 3  # receeding temporal mode

    task_id: int = None # Must be specified upon launching


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config),
        args.policy.dir,
        default_prompt=args.default_prompt,
    )


def main(args: Args) -> None:
    logging.info(f"Using task_name: {args.task_name}")

    # policy = create_policy(args)
    # policy_metadata = policy.metadata

    # policy = B1KPolicyWrapper(
    #     policy,
    #     task_name=args.task_name,
    #     control_mode=args.control_mode,
    #     max_len=args.max_len,
    #     action_horizon=args.action_horizon,
    #     temporal_ensemble_max=args.temporal_ensemble_max,
    #     fine_grained_level=args.fine_grained_level,
    # )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServerMulti(
        policy=None,
        host="0.0.0.0",
        port=args.port,
        # metadata=policy_metadata,
        task_id=args.task_id,
    )

    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
