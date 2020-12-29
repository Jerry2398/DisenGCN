file_path = ''

lr = 0.001  # [1e-8, 1]
weight_decay = 1e-5  # [1e-10, 1]
iter = 100

in_dims = []
layer_num = 3
channels = [16, 8, 4]
C_dims = [8, 4, 2]
in_dims.append(128)
for i in range(1,layer_num):
    in_dims.append(channels[i] * C_dims[i])

dropout = 0.10
iterations = 7
beta = 1
out_dim = 7
