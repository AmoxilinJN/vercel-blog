## Class Templates

首先举个使用模板的例子

```c++
//vector.h
template <tpyename T>
class vector {
    public:
    	void push_back(const T& v);{ /*...*/}
    	size_t size() const;{ /*...*/}	//cosnt表示该函数不会修改类的状态
    	T& operator[](size_t index) const;{ /*...*/}	//[]重载运算符
    /* ... */
};
//函数实现可以在定义实现，也可以在后面实现
/*
template <typename T>
void vector<T>::push_back(const T& v){}
template <typename T>
size_t vector<T>::size() const{}
template <typename T>
T& vector<T>::operator[](size_t index) const{}
*/
```

1. 类模板不是类，只有填上参数才是一个类，填入两个不同参数的类模板是两个不同的类
2. 使用类模板会显著增加编译时间
3. 类模板的定义和实现无法分开编译，也就是说，无法通过将定义写在`.h`中以及将实现写在`.cpp`中进行单独编译，因此要在头文件就写好实现（g++编译器编译标准库模板便是如此）
4. `template <typename T> or template <class T>` 中`class`和`typename`作用相同，`typename`只是增强了可读性
5. `typename N = std::allocator<T>`可定义默认参数，若未给定具体参数则为T类型
6. `vector`是一个模板，而`vector<T>`则是一个类型名，因此`vector::size()`是非法的

## Sequence Containers

序列容器类似python中的`list`,即一堆元素的线性集合，c++中常用`std::vector<T>`和`std::deque<T>`

1. `std::vector`
   - `std::vector<T> v`创建空向量v
   - `std::vector<T> v(n)`创建大小为n的向量v
   - `std::vector<T> v(n,e)`同上，元素值为e
   - `v.push_back(e)`向量v末尾增加(或者说如字面意思，压入)一个元素e
   - `v.pop_back()`将向量v末尾元素删除(弹出)，无返回值
   - `v.empty()`判断向量v是否为空
   - `T e = v[i] or v[i] = e`赋值操作，无越界检查
   - `T e = v.at(i) or v.at(i) = e`同上，越界报错
   - `v.clear()`清空向量v
2. `std::deque<T>`
   - 同`std::vector`
   - `d.push_front(e)`向量d开头增加(压入)一个元素e
   - `d.pop_front()`将向量d开头元素删除(弹出)，无返回值

