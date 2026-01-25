## 查看GPU参数

` watch -n 1 nvidia-smi` 每秒更新

- Memory-Usage：已用显存 / 总显存
- GPU-Util：GPU利用率
- Processes：进程

## 查看CPU参数

` htop ` 

- 顶部仪表盘：CPU监视器、Mem内存监视器、Swp交换区监视器

- Tasks：任务监视器显示了当前系统中任务和线程总数、内核线程数

- Load average：负载平均值显示了过去 1 分钟、5 分钟和 15 分钟内的平均 CPU 负载

- Uptime：显示了系统开机运行时间

- 下部主进程界面：![](/blogs/20260101-00/7d26c2c94bb42d59.png)

  其中S列进程状态含义：![](/blogs/20260101-00/3a476022ff86b505.png)

其他的有需要了再写

## 日志重定向

用于解决screen后台运行输出过多问题，` python xxx.py 2>&1 | tee output.log` 可配合`tail -f output.log` 实时看日志

## screen在attach状态掉线无法重进

先`screen -d xxxx` 再`screen -r xxxx` 便可解决