import os
if 'use_kubernets.backend' in os.listdir('.'):
    from .kubernetes_backend import *
else:
    from .atlas_backend import *
    # from .local_backend import *
