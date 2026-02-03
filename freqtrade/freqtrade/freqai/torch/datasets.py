import torch


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys, window_size):
        self.xs = xs
        self.ys = ys
        self.window_size = window_size

    def __len__(self):
        return len(self.xs) - self.window_size

    def __getitem__(self, index):
        idx_rev = len(self.xs) - self.window_size - index - 1
        window_x = self.xs[idx_rev : idx_rev + self.window_size, :]
        # Beware of indexing, these two window_x and window_y are aimed at the same row!
        # this is what happens when you use :
        window_y = self.ys[idx_rev + self.window_size - 1, :].unsqueeze(0)
        return window_x, window_y
