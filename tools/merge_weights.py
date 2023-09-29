import torch
import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path_mlr', default='')
    parser.add_argument('--weight_path_ovod', default='')
    parser.add_argument('--output_path', default='')
    args = parser.parse_args()
    print("loading mlr model weights ", args.weight_path_mlr)
    checkpoint_mlr = torch.load(args.weight_path_mlr, map_location="cpu")
    print("loading ovod model weights ", args.weight_path_ovod)
    checkpoint_det = torch.load(args.weight_path_ovod, map_location="cpu")
    for name in tqdm.tqdm(checkpoint_mlr["model"].keys(), ncols=90):
        new_name = "ovmlr_model." + name
        checkpoint_det["model"][new_name] = checkpoint_mlr["model"][name]
    torch.save(checkpoint_det, args.output_path)
    print("saved to ", args.output_path)
    

