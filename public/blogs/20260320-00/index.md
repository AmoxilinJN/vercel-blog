1. 用于当前终端

`export PYTHONPATH=$PYTHONPATH:/home/usrname/models`

2. 用于当前用户

编辑`~/.bashrc`文件，末尾添加`export PYTHONPATH=$PYTHONPATH:/home/usrname/models`，修改完后`source ~/.bashrc`

3. 用于所有用户

修改`/etc/profile`文件，末尾添加`export PYTHONPATH=$PYTHONPATH:/home/usrname/models`，修改完后`source /etc/profile`

检查：`echo $PYTHONPATH`