<font size=4>首先先说总结：程序要调用某函数时，首先call该函数的plt表地址，在plt表中call到该函数的got表地址，若该函数未被调用过，则从got表call回plt表，在一阵搜索后获得该函数的实际地址然后调用，同时将该地址写入got表中，待下次调用该函数时，在got表中便直接得到函数地址，不再call回plt表。

PLT：**程序链接表**（Procedure Link Table），用来存储外部函数的入口点，换言之，程序会到PLT表中寻找外部函数的地址。PLT存储在代码段内，在运行之前就已经确定并且不会被修改。

GOT：**全局偏移量表**（Global Offset Table），用于存储外部函数在内存中的确切地址。GOT存储在数据段内，可以在程序运行过程中被修改（开了Full RELRO就没有读写权限了）。

举个例子，首先我们来到调用puts函数这里
![](/blogs/20250305-00/4909e6d780beb781.png)
跟进查看
![](/blogs/20250305-00/c148eebb7a3f60ed.png)
发现该地址便是puts函数的plt表地址，我们接着跟进
![](/blogs/20250305-00/14de948d2caf2438.png)
发现来到got表地址，在动态调试中，这里应该是存储着plt表中jmp指令的下一条指令

```
push 0x**
jmp 0x********<PTL[0]>
---
push 0x********[_GLOBAL_OFFSET_TABLE_+4]<GOT[1]>
jump 0x********<GOT[2]><_dl_runtime_resolve>

```

其中，第一条指令push的值在源码中称为reloc_offset，可以理解为函数的代号

第二条指令是跳到PLT[0]的位置，该位置储存了上文写出的两条指令和没写的传参调用_dl_fixup函数的指令

push指令将GOT[1]中的内容推入栈中，改内容为link_map，它存储了解析每个函数的地址的关键数据

jump指令跳到GOT[2]，此处保存了_dl_runtime_resolve的地址， _dl_runtime_resolve是个解析函数，用来解析每个外部函数的地址，其参数有2：link_map和reloc_offset。该函数也只是一段汇编代码，核心函数是 _dl_fixup(struct link_map *1, ElfW(Word) reloc_offset)

进入_dl_runtime_resolve解析完毕后，GOT表中便存上了puts函数的真实地址，走出函数后便会立刻进入到puts函数执行

这里用一张图进行总结：
![](/blogs/20250305-00/f6ae8f9522e8d65f.png)
