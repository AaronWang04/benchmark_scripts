import torch
import torch.nn.functional as F
from torch import ops
from torch.ops import aten
from torch._inductor.decomposition import decompositions
from torch.fx import symbolic_trace
from torch._decomp import get_decompositions

# Get the GELU decomposition
print('GELU decomposition:')
if aten.gelu in decompositions:
    print('Found in decompositions')
    
# Let's trace and decompose manually
def fn(x):
    return F.gelu(x, approximate='tanh')

# Get decomposition table
decomp_table = get_decompositions([aten.gelu])
print('Decomposition table:', decomp_table)

# Apply decomposition
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import decomposition_table

x = torch.randn(10, 10)
with torch._decomp.decomposition_table.context(decomp_table):
    gm = make_fx(fn)(x)
    
print('\nDecomposed GELU graph:')
print(gm.code)