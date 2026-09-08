OS： Ubuntu 24.04.4 LTS (GNU/Linux 6.17.0-22-generic x86_64)

nvidia-smi:

+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.84                 Driver Version: 595.84         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX PRO 6000 Blac...    Off |   00000000:01:00.0 Off |                  Off |
| 52%   84C    P1            494W /  600W |   30563MiB /  97887MiB |     97%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

可通过`ssh gpu-server`命令直接连接到GPU服务器上。