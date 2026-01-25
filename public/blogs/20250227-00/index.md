<font size=4>RC4总结起来就是将明文与生成的密钥流进行异或加密，得到密文<br>
现在分析RC4生成密钥流的流程：

1. 初始化数组S和K

```c
keylength = byteKey.length;
for (int i = 0; i < 256; i++) {
    S[i] = i;
    K[i] = byteKey[i % keylength];
}
```

​	可以看到数组S被0~255按序填充，而数组K则被原始密钥重复填充直至第255位

2. 初始化置换数组S

```c
   int j = 0;
   for (int i = 0; i < 256; i++) {
       j = (j + S[i] + K[i]) % 256;
       int temp = S[i];
       S[i] = S[j];
       S[j] = temp;
   }
```

​	此段代码将数组S进行乱序排列

3. 生成密钥流

```c
   char[] cipherText = new char[plainText.length];
   int i = 0;
   int j = 0;
   int key;
   int plainTextLen = 0;
   while (plainTextLen < plainText.length) {
       i = (i + 1) % 256;
       j = (j + S[i]) % 256;
       int temp = S[i];
       S[i] = S[j];
       S[j] = temp;
       key = S[(S[i] + S[j]) % 256];
       cipherText[plainTextLen] = (char) (plainText[plainTextLen] ^ key);
       plainTextLen++;
   }
```

   首先将定位变量i和j置零，进行操作，每操作一次，i加1，j取本身与S[i]之和，S[i]与S[j]互换，取定位变量S[i]+S[j]，定位此数组S中的数，取为key

   将key与明文异或加密得到密文，加密完成。

由于异或运算的可逆性，我们只需将密文与密钥流再次异或便可得到明文，因此拿到密钥后我们的任务便是用相同的方法得到密钥流

```c
void decrypt(char *key,char *enc,char *dec)
{
    int len = strlen(enc);
    int S[len]={0};
    int K[len]={0};
    int KEY;
    int enclen=0;
    for(int i=0; i < 256; i++){
        S[i]=i;
        K[i]=key[i % strlen(key)];
    }
    for(int i=0, j=0; i < 256; i++){
        j = (j + S[i] + K[i]) % 256;
        int temp = S[i];
        S[i] = S [j];
        S[j] = temp;
    }
    int i=0, j=0;
    while(enclen < len){
        i = (i + 1) % 256;
        j = (j + S[i]) % 256;
        int temp = S[i];
        S[i] = S[j];
        S[j] = temp;
        KEY = S[(S[i] + S[j]) % 256];
        dec[enclen] = (char)(enc[enclen] ^ KEY);
        enclen++;
    }
}
```

*Fin*

