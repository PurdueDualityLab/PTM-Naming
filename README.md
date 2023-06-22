# PTM-Naming
# Example usage of generating an ordered list of NN layers from a pytorch model:
'''
class TestNN(nn.Module):
    # Replace this class with your model
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace),
            nn.Linear(128, 128),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

gen = OrderedListGenerator(TestMLP(), torch.randn(2, 128))

# Return a list[NodeInfo] object
l = gen.get_ordered_list()

# Print the above list
gen.print_ordered_list

# For printing a list with all the layer connections
gen.print_connection()

'''