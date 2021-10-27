import os
local_backend = os.getenv('LOCAL_BACKEND')
if local_backend:
    from .local_backend import *
else:
    if 'use_kubernets.backend' in os.listdir('.'):
        from .kubernetes_backend import *
    else:
        from .atlas_backend import *
