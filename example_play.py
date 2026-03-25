import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

from deltanet import DeltaBlock
#from Kaczmarz_one_step_Slimpajama import DeltaBlock
from slimpajama_data import build_slimpajama_byte_splits, load_memmap_splits
 
"""
-    this is an example on how to use the architecture. It's byte per byte text generation, it uses a dumb model without mlp bloc, it's capacity limited
+Byte-level next-token training example for DeltaBlock.
+
+This script:
+1) Streams SlimPajama once with a pseudorandom seed.
+2) Writes 70M bytes/tokens into local train/val/test .bin files
+   (10M val + 10M test + 50M train).
+3) Trains and evaluates DeltaBlock on those persisted datasets.
 """
 
def tensor_to_char(tensor):
    index = torch.argmax(tensor).item()
    return chr(index)
 
def char_to_tensor(char, device):
    tensor = torch.zeros(256, device=device)
    tensor[ord(char)] = 1.0
    return tensor
 

def generated(tensor):
    new_tensor = torch.zeros(256, device=tensor.device, dtype=tensor.dtype)
    index = torch.argmax(tensor)
    new_tensor[index] = 1.0
    return new_tensor
 


def generate(model, prompt, seq_len=128):
    model.eval()
    device = next(model.parameters()).device
    S = None

    with torch.no_grad():
        for c in prompt:
            x = char_to_tensor(c, device=device)
            x, S = model.step(x, S)
            print(c, end="")
            x = x.squeeze(0).squeeze(0)

        print(tensor_to_char(x), end="")
        for _ in range(seq_len):
            x, S = model.step(x, S)
            x = generated(x)
            x = x.squeeze(0).squeeze(0)
            print(tensor_to_char(x), end="")
    print()


class ByteMemmapDataset(Dataset):
    def __init__(self, memmap_array, seq_len):
        self.data = memmap_array
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx + self.seq_len].astype("int64"))
        y = torch.from_numpy(self.data[idx + 1 : idx + self.seq_len + 1].astype("int64"))
        return x, y


def one_hot_batch(token_batch):
    return torch.nn.functional.one_hot(token_batch, num_classes=256).float()


def evaluate(model, dataloader, criterion, device, eval_steps=100):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for x_tokens, y_tokens in dataloader:
            x_tokens = x_tokens.to(device)
            y_tokens = y_tokens.to(device)

            inputs = one_hot_batch(x_tokens)
            targets = one_hot_batch(y_tokens)

            output = model(inputs)
            loss = criterion(output, targets)
            total_loss += loss.item()
            steps += 1

            if steps >= eval_steps:
                break

    return total_loss / max(1, steps)


def main():
    # Data config
    data_dir = "./data/slimpajama_70m"
    seed = 12345
    total_tokens = 70_000_000
    val_tokens = 10_000_000
    test_tokens = 10_000_000

    # Training config
    batch_size = 4
    seq_len = 128
    num_epochs = 2
    train_steps_per_epoch = 300
    eval_steps = 100
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Preparing/loading SlimPajama streaming split files...")
    build_slimpajama_byte_splits(
        data_dir,
        seed=seed,
        total_tokens=total_tokens,
        val_tokens=val_tokens,
        test_tokens=test_tokens,
    )

    train_mm, val_mm, test_mm = load_memmap_splits(data_dir)

    train_ds = ByteMemmapDataset(train_mm, seq_len=seq_len)
    val_ds = ByteMemmapDataset(val_mm, seq_len=seq_len)
    test_ds = ByteMemmapDataset(test_mm, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    model = DeltaBlock(256, 2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for step, (x_tokens, y_tokens) in enumerate(train_loader, start=1):
            x_tokens = x_tokens.to(device)
            y_tokens = y_tokens.to(device)

            inputs = one_hot_batch(x_tokens)
            targets = one_hot_batch(y_tokens)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 20 == 0:
                print(
                    f"epoch {epoch + 1}/{num_epochs} step {step}/{train_steps_per_epoch} "
                    f"train_loss={running_loss / 20:.6f}"
                )
                running_loss = 0.0

            if step >= train_steps_per_epoch:
                break

        val_loss = evaluate(model, val_loader, criterion, device, eval_steps=eval_steps)
        print(f"epoch {epoch + 1}/{num_epochs} validation_loss={val_loss:.6f}")

    test_loss = evaluate(model, test_loader, criterion, device, eval_steps=eval_steps)
    print(f"test_loss={test_loss:.6f}")

    print("Generated sample:")
    generate(model, "The ")


if __name__ == "__main__":
    main()
