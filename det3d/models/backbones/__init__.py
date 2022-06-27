import importlib
spconv_spec = importlib.util.find_spec("spconv")
found = spconv_spec is not None

if found:
    from .scn import SpMiddleResNetFHD
    from .dcn import *
    from .pcnvgg import *
    from .pcnres18 import *
    from .pcnres34 import *
else:
    print("No spconv, sparse convolution disabled!")

