在GPU服务器上训练CURL算法时，当环境步达到3-4万后，发生如下问题：

```bash
[curl-sac-train] ep=36 env_steps=36000 ep_return=23.607 ep_env=1000 q=1.505e+21 curl=3.817 pi=-1.271e+10 temp=0.01799
[curl-sac-train] ep=37 env_steps=37000 ep_return=24.108 ep_env=1000 q=1.425e+24 curl=6.238 pi=-7.212e+11 temp=0.01367
[curl-sac-train] ep=38 env_steps=38000 ep_return=9.771 ep_env=1000 q=2.193e+25 curl=6.238 pi=-3.199e+12 temp=0.01171
[curl-sac-train] ep=39 env_steps=39000 ep_return=38.529 ep_env=1000 q=1.121e+26 curl=6.238 pi=-7.421e+12 temp=0.0104
[curl-sac-train] ep=40 env_steps=40000 ep_return=38.939 ep_env=1000 q=3.519e+26 curl=6.238 pi=-1.326e+13 temp=0.009383
```

可见q和pi的数值直接变得非常大（完整日志见`data\curl-sac-20260908-145018`）。

文献官方团队实现的CURL源码位于`curl/`，现在根据官方实现的源码结合本地实现，分析问题原因。