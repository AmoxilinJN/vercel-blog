单次可行方案：跳过系统限制
```
pip install [替换为你要安装的包的名字] --break-system-packages
```
长久可行方案：移除提示文件（危）
```
sudo mv /usr/lib/python3.11/EXTERNALLY-MANAGED /usr/lib/python3.11/EXTERNALLY-MANAGED.bak
```
使用 pipx（夯）
```
sudo apt install pipx
pipx ensurepath
pipx install package_name
```
使用 venv（夯）