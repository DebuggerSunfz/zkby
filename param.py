import argparse

MultiAgent = False
Max_step_num = 1500

parser = argparse.ArgumentParser()
parser.add_argument("--excute_path", type=str, help="Checkpoint path for inference",
                    default="D:\\py_files\\platform\\20221130\\Windows\\Mono\\Windows\\ZK.exe")
parser.add_argument("--checkpoint", type=str, help="Checkpoint path for inference",)
parser.add_argument("--save_folder", type=str, help="save_folder path for inference",)
parser.add_argument("--train_num", type=int, default=0, help="Training iteration number (Default: 10)")
parser.add_argument("--num_gpus", default=7, action="store_true", help="Use GPU (Default: True)")

parser.add_argument("--render", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--frame_feature_size", type=int, default=43)
parser.add_argument("--randomseed", type=int, default=12345)
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--num_workers", type=int, default=50)
parser.add_argument("--evaluation_num_workers", type=int, default=25)
parser.add_argument("--evaluation_duration", type=int, default=10)
parser.add_argument("--evaluation_interval", type=int, default=10)
