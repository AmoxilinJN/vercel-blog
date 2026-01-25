TEA分组长度为64位，密钥长度为128位(4个32位无符号整数)，采用Feistel网络，一般是32轮循环加密，特征是0x9e3779b9或-0x61C88647(可能会变)，加密特征是三个小括号两两异或，括号内一个左移4位(*16)一个右移5位，下面是代码实现：

```
void Encrypt(long* v,long* k)
{
	unsigned long y=v[0],z=v[1],sum=0,delta=0x9e3779b9,n=32;		//v[0]和v[1]是待加密信息
	while(n-->0)
	{
		sum+=delta;
		y+=((z<<4)+k[0])^(z+sum)^((z>>5)+k[1]);
		z+=((y<<4)+k[2])^(y+sum)^((y>>5)+k[3]);				//k[0]到k[3]是密钥
	}
	v[0]=y;
	v[1]=z;
}
```

解密即加密逆过程，如下：

```
void Decrypt(long* v,long* k)
{
	unsigned long n=32,sum,y=v[0],z=v[1],delta=0x9e3779b9;
	sum=delta<<5;				//等同于sum=delta*32,即加密中32轮过后的sum值
	while(n-->0)
	{
		z-=((y<<4)+k[2])^(y+sum)^((y>>5)+k[3]);
		y-=((z<<4)+k[0])^(z+sum)^((z>>5)+k[1]);
		sum-=delta;
	}
	v[0]=y;
	v[1]=z;
}
```

此外TEA还有改进算法XTEA,XXTEA，且待下回