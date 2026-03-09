import torch
import torch.nn as nn
import torch.optim as optim
from deltanet_altered import DeltaBlock

"""
    this is an example on how to use the architecture. It's byte per byte text generation, it uses a dumb model without mlp bloc, it's capacity limited
"""

def text_to_tensor(texts):
    B = len(texts)  # Number of text samples
    L = max(len(text) for text in texts)  # Max length of the texts
    D = 256  # Number of possible ASCII characters (0-255 inclusive)
    out = torch.zeros((B, L, D))  # Create a zero tensor with shape (B, L, D)
    
    for b in range(B):  # Iterate over batch
        for c in range(len(texts[b])):  # Iterate over characters in the b-th text
            out[b, c, ord(texts[b][c])] = 1.0  # Set the corresponding ASCII index to 1.0
    
    return out
    
def tensor_to_text(tensor):
    B, L, D = tensor.shape  # Get the shape of the tensor
    texts = []
    
    for b in range(B):  # Iterate over batch
        text = ""
        for l in range(L):  # Iterate over sequence length
            char_idx = torch.argmax(tensor[b, l]).item()  # Get the index of the max value
            if char_idx != 0:  # Ignore padding (assuming 0 represents no character)
                text += chr(char_idx)  # Convert index back to character
        texts.append(text)  # Append reconstructed string to the list
    
    return texts

def tensor_to_char(tensor):
    index = torch.argmax(tensor)
    return chr(index)

def char_to_tensor(char):
    tensor = torch.zeros(256)
    tensor[ord(char)]=1.0
    return tensor

def generated(tensor):
    new_tensor = torch.zeros(256)
    index = torch.argmax(tensor)
    new_tensor[index] = 1.0
    return new_tensor

def generate(model, prompt, seq_len=10):
    model.eval() # Set to evaluation mode
    S = None
    last_output = None
    
    with torch.no_grad():
        # Process the prompt to build the hidden state S
        for c in prompt:
            print(c, end="")
            x = char_to_tensor(c)
            last_output, S = model.step(x, S)
        
        # Predict the next character after the prompt
        x = last_output 
        
        # Generate new characters
        for i in range(seq_len):
            # Convert continuous output to a discrete "one-hot" character vector
            char = tensor_to_char(x)
            print(char, end="")
            
            x_input = char_to_tensor(char)
            x, S = model.step(x_input, S)

# --- Training Setup ---

inputs = text_to_tensor(["hello world", "bonjour", "test"])
target = inputs.roll(shifts=-1, dims=1)

# Note: Since the Projection Rule update doesn't use the 'alpha' 
# or 'beta' logic in the same way, you can keep expand=2 
# but you may want to monitor if the model converges faster.
model = DeltaBlock(256, expand=2)

criterion = nn.MSELoss()
# The Projection Rule can be sensitive; if loss becomes NaN, 
# try reducing the learning rate to 0.0005
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epoch = 500

for epoch in range(num_epoch):
    model.train()
    optimizer.zero_grad()
    # Ensure chunk size divides the sequence length (L=11 in "hello world")
    # or just use chunk=1 for simple sequence processing.
    output = model(inputs, chunk=1) 
    loss = criterion(output, target)
    loss.backward()
    
    # Optional: Gradient clipping is highly recommended with the Projection Rule
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"epoch:{epoch}/{num_epoch}, loss:{loss.item():.6f}")

print("\nFinal generated text:")
print(tensor_to_text(output))

print("\nPrompting the model with 'h':")
generate(model, "h", seq_len=10)
print("\n")
