# A-cv2-GUI-implementation-of-GAME_OF_LIFE

- 黑色是死的，白色活的

- 空格键 开始/暂停 （启动时默认暂停）

- 暂停状态下鼠标左键单击切换单个细胞的生死，运行状态左键没用

- 鼠标滚轮缩放

- 鼠标右键按住拖拽进行视窗移动

- 加号和减号调整单位时间大小
<h3 style="color:red;">Caution</h3>
- s键：在当前目录下生成"_-_--__---___tmpfile.json"文件，可用于命令行参数读取
- 命令行参数：
-  - 输入```python game_of_life.py a_file_path.json```读取“地图”
-  - - 地图格式：```[[0,0],[0,0]]```（最好是方阵，否则会被截断部分）
-  - 如果输入的是数字（而且当前路径下没有以该数字命名的文件），则设置为 该数字*该数字 大小的全死方阵

