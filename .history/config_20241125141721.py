# Very Lightweight
learning_rate = 1e-3
nepochs = 2        # Very few epochs
batch_size = 128   # Larger batch size for faster processing
max_len = 2        # Short sequences
hidden_size = 2    # Tiny hidden size
num_layers = 1     # Single layer

# Lightweight
#learning_rate = 1e-4
#nepochs = 10        # Very few epochs
#batch_size = 64   # Larger batch size for faster processing
#max_len = 8        # Too short for meaningful context
#hidden_size = 8    # Too small to capture complexity
#num_layers = 1     # Single layer might be insufficient

# Suggested better hyperparameters
#learning_rate = 1e-4      # Slightly faster learning
#nepochs = 10             # More training time
#batch_size = 32          # Smaller for better generalization
#max_len = 128            # Longer sequences for better context
#hidden_size = 64       # More capacity to learn patterns
#num_layers = 3

# Heavy
# learning_rate = 1e-3      # Slightly faster learning
# nepochs = 10            # More training time
# batch_size = 32          # Smaller for better generalization
# max_len = 128            # Longer sequences for better context
# hidden_size = 256       # More capacity to learn patterns
# num_layers = 3         # More layers for complex patterns