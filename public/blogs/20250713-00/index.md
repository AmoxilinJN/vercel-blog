## auto

1. `auto`不能识别含两个及以上double的类型，e.g.`std::pair<double,double> or std::vector<double>`
2. 若类型名太长且引用次数多，可使用别名`using`
3. `auto`常用于for循环计数变量，e.g.

```
std::vector<int> vec {1,2,3,4};
for(auto elem : vec){
	std::cout << elem << " ";
}
```

3. `auto`常用于函数传参，使函数可以接受任意类型的参数，e.g.

```
void  borges(auto cuento){
	std::cout << cuento << std::endl;
}

borges(std::string{"el sur"});//auto deduced as `std::string`
```

## 初始化变量

1. 初始化变量可以使用`std::vector`或`std::map`进行统一初始化，也可以进行结构化绑定(Structured Binding)（需要已知变量数，不可用于`std::vector`一类）

```
//统一初始化
#include <vector>
int main() {
    std::vector<int> v{1, 2, 3, 4, 5};
    return 0;
}
---
#include <iostream>
#include <map>
int main() {
    std::map<std::string, int> ages{
        {"Alice", 25},
        {"Bob", 30},
        {"Charlie", 35}
    };

    // Accessing map elements
    std::cout << "Alice's age: " << ages["Alice"] << std::endl;
    std::cout << "Bob's age: " << ages.at("Bob") << std::endl;
    return 0;
}
```
...
```
//结构化绑定
auto [var1, var2, ..., varN] = expression;
---
std::tuple<std::string, std::string, std::string> getClassInfo() {
    std::string className = "CS106L";
    std::string location = "online";
    std::string language = "C++";
    return {className, location, language};
}
int main() {
    auto [className, location, language] = getClassInfo();
    std::cout << "Join us " << location << " for " << className << " to learn " << language << "!" << std::endl;
    return 0;
}
```

