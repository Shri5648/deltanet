import torch
import torch.nn as nn
import torch.optim as optim
from deltanet import DeltaBlock

# ... (Keep text_to_tensor, tensor_to_text, tensor_to_char, char_to_tensor, generated as they are) ...

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
    
    if epoch % 50 == 0:
        print(f"epoch:{epoch}/{num_epoch}, loss:{loss.item():.6f}")

print("\nFinal generated text:")
print(tensor_to_text(output))

print("\nPrompting the model with 'h':")
generate(model, "h", seq_len=10)
print("\n")
