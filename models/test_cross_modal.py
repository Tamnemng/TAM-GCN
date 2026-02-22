import torch
from models.resnet_gcn_attention import ResNet_GCN_Attention
import graph.ucla

def test_model():
    print("Initializing ResNet_GCN_Attention...")
    
    # Initialize the model
    model = ResNet_GCN_Attention(
        num_class=10,
        num_point=20,
        num_person=1,
        graph='graph.ucla.Graph',
        graph_args={'labeling_mode': 'spatial'},
        in_channels=3,
        drop_out=0,
        adaptive=True,
        freeze_gcn=True
    )
    
    print("Model initialized successfully!")
    
    # Create dummy input data
    N = 4  # Batch size
    C_ske = 3 # Skeleton channels (x, y, z)
    T = 52 # Number of frames
    V = 20 # Number of joints
    M = 1  # Number of persons
    
    C_rgb = 3 
    H = 224
    W = 224
    
    # Dummy skeleton: [N, C, T, V, M]
    dummy_ske = torch.randn(N, C_ske, T, V, M)
    
    # Dummy RGB: [N, 3, 224, 224]
    dummy_rgb = torch.randn(N, C_rgb, H, W)
    
    print(f"Propagating dummy skeleton with shape: {dummy_ske.shape}")
    print(f"Propagating dummy rgb with shape: {dummy_rgb.shape}")
    
    try:
        # Forward pass
        output = model(dummy_ske, dummy_rgb)
        print(f"Forward pass successful! Output shape: {output.shape}")
        
        assert output.shape == (N, 10), f"Expected shape {(N, 10)}, got {output.shape}"
        print("All tests passed!")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_model()
