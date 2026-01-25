## & , * and new

1. `&`常用于函数传参“引用传递”，即对形参的改变同步映射到实参，e.g.

```c++
void squareN(int& N){
	N*=N;
}

int main(){
	int num=5;
	squareN(num);	//num = 25
	return 0;
}
```

2. `Type* const`地址不可变，内容可变；`const Type*`地址可变，内容不可变；`const Type* const`地址、内容都不可变

```c++
auto ptr1 = new std::pair<int, double>();
auto ptr2 = new std::pair<int, double>;
double* ptr3 = new double[3]{106, 3.14, 3};
delete ptr1,ptr2;
delete[] ptr3;
```

3. 指针1被初始化为0，而指针2未被初始化
4. 指针可改变指向对象，引用不可变，e.g.

```c++
double* ptr = &pi;
ptr = &e;	//ptr指向e

double& ref = pi;
ref = e;	//pi = e
```



