import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import json

from .metrics import EvalTools


class ImagePairDataset(Dataset):
    def __init__(self, root_dir, gt_dir=None, transform=None):
        self.root_dir = root_dir
        if gt_dir is None:
            self.gt_dir = root_dir
        else:
            self.gt_dir = gt_dir
        self.transform = transform or T.ToTensor()
        self.image_pairs = self._get_image_pairs()

    def _get_image_pairs(self):
        pred_files = [f for f in os.listdir(self.root_dir) if f.endswith('_pred.png')]
        pairs = []
        for pred_file in pred_files:
            base_name = pred_file.replace('_pred.png', '')
            gt_file = f"{base_name}_gt.png"
            gt_path = os.path.join(self.gt_dir, gt_file)
            pred_path = os.path.join(self.root_dir, pred_file)
            if os.path.exists(gt_path):
                pairs.append((pred_path, gt_path))
        return pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pred_path, gt_path = self.image_pairs[idx]
        pred_image = Image.open(pred_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")
        return self.transform(pred_image), self.transform(gt_image), pred_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_folder', 
                        type=str, 
                        default="outputs/matchnerf_3v_tnt_full/test/tnt", 
                        help='Path to folder with *_pred.png images')
    parser.add_argument('--gt_folder', 
                        type=str, 
                        default="outputs/matchnerf_3v_tnt_full/test/tnt", 
                        help='Path to folder with *_gt.png images')
    args = parser.parse_args()

    eval_tools = EvalTools(device="cuda")

    # Set your folder
    pred_folder = args.pred_folder

    # Create dataset and dataloader
    dataset = ImagePairDataset(pred_folder, gt_dir=args.gt_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    scores_dict = {}
    avg_logger = {}

    for pred, gt, pred_path in tqdm(dataloader):
        pred_rgb_nb = pred[0].permute(1, 2, 0).detach().cpu().numpy()
        gt_rgb_nb = gt[0].permute(1, 2, 0).detach().cpu().numpy()  # h,w,3

        eval_tools.set_inputs(pred_rgb_nb, gt_rgb_nb)
        cur_metrics = eval_tools.get_metrics(return_full=False)

        for m, m_val in cur_metrics.items():
            if m not in avg_logger:
                avg_logger[m] = []
            avg_logger[m].append(m_val)

        img_name_list = str(os.path.basename(pred_path[0])).split("_")
        scene_name = img_name_list[0]
        view_idx = int(img_name_list[1][4:])
        src_idx = [int(img_name_list[2][3:]), int(img_name_list[3]), int(img_name_list[4])]
        if scene_name not in scores_dict:
            scores_dict[scene_name] = []
        
        cur_score_dict = {"view_idx": view_idx, "src_idx": src_idx, 
                          "metrics": {k: float(v) for k, v in cur_metrics.items()}}
        scores_dict[scene_name].append(cur_score_dict)

    with open(os.path.join(pred_folder, "0scores.json"), "w") as f:
        json.dump(scores_dict, f)

    # report the average scores
    print(args.pred_folder)
    for m, m_list in avg_logger.items():
        m_avg = sum(m_list) / len(m_list)
        print(m, m_avg)


if __name__ == "__main__":
    main()
