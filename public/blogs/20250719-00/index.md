## Associative Containers

关联容器类似于 Python 中的字典和集合，每个元素都有一个 unique key

### Ordered Containers

有序容器将元素通过 key 进行自动排序，使用 `std::map<K,V>` 和 `std::set<T>` 进行映射和存储，要求 `K` 和 `T` 必须支持运算符 `<` 操作（因为函数实现用了红黑树算法，搜索时用二分法查找，需要比较结果来判断走 left 还是走 right）；因此自定义类型可通过三种方法实现要求：

1. 运算符重载，重新定义运算符 `<`，e.g.

```c++
   //定义一个没有数字的比较
   bool operator<(const MyType& a,const MyType& b){
       //Return true if `a` is less than `b`
   }
```

2. 定义一个函子(*functor*)，适用于不想为了自定义类型而重载全局运算符 `<`，e.g.

```c++
   struct Less{
       bool operator()(const MyType& a,const MyType& b){
           //Return true if `a` is less than `b`
       }
   };
   std::map<MyType,double,Less>my_map;
   std::set<MyType,Less>my_set;
```

   该法通过向 `map` 和 `set` 传递比较模板参数实现，如果未提供参数则默认使用标准库的 `std::less<T>`，该模板定义使用了运算符 `<`

3. 使用 lambda 函数，类似第二条，e.g.

```c++
   auto comp = [](const MyType& a,const MyType& b){
       //Return true if `a` is less than `b`
   };
   std::map<MyType,double,decltype(comp)>my_map(comp);	//decltype(comp)推断出编译时comp的类型，
   std::set<MyType,decltype(comp)>my_set(comp);	//感觉类似auto
```

#### `std::map<K,V>`

- `std::map<K,V> m` 创建空映射

- `std::map<K,V> m{{k1,v1}{k2,v2}...}` 统一初始化键值对

- `auto v = m[k] or m[k] = v` 若 `k` 不在映射中，则 `v` 为默认值，即将默认值插入 `m` 中

  - 这里的默认值与 V 的类型有关，`V--default：bool--false；int,size_t,float,double--0；container type--empty container`

  - 利用特性 e.g.

```c++
    //统计字符串中每个字符的数量
    std::string quote = "Peace if possible, truth at all costs";
    std::map<char,size_t> counts;
    for(char c : quote){
        counts[c]++;	//如果字符x第一次出现，则counts[x]=0，然后++运算记一次
    }
```



- `auto v = m.at(k)` 同上，`k` 不在映射中报错

- `m.insert({k,v}) or m.insert(p)` 插入键值对(`std::pair p`)

- `m.erase(k)` 删除 `k`，`k` 不一定存在于映射中

- `if(m.count(k)) ... or if(m.contains(k)) ...` 检查 `k` 是否在映射中

- `m.empty()` 检查 `m` 是否为空

#### `std::set<T>`

- `std::set<T> s` 创建空集合
- `s.insert(e)` 向 `s` 中添加 `e` 元素（加多少次都只有一个 `e` 元素，不重复性）
- `s.erase(e)` 删除 `e`，`e` 不一定在 `s` 中
- `if(s.count(e)) ... or if(s.contains(e)) ...` 检查 `e` 是否在 `s` 中
- `s.empty()` 检查 `s` 是否为空

### Unordered Containers

无序容器通过哈希表进行元素查找，效率高，但遍历元素效率不如关联容器，且使用内存比关联容器多；使用 `std::unordered_map<K,V>` 和 `std::unordered_set<T>`，`K` 和 `T` 必须与哈希函数（输入任意类型，输出 `size_t` 类型）或相等函数相关，因此自定义类型也是对哈希函数、相等函数分别有三种方法实现：

##### Hash function

1. 为自定义类型创建一个专门的 `std::hash` 函子，这是无序容器哈希元素的默认方法，e.g.

```c++
   template<>
   struct std::hash<MyType>{
       std::size_t operator()(const MyType& o) const noexcept{
           //Calculate and return the hash of `o` ...
       }
   };
   std::unordered_map<MyType,std::string> my_map;
```

2. 定义一个自定义函子，避免改变哈希函数，e.g.

```c++
   struct MyHash{
       std::size_t operator()(const MyType& o) const noexcept{
           //Calculate and return the hash of `o` ...
       }
   };
   std::unordered_map<MyType,std::string,MyHash> my_map;
```

3. 使用 lambda 函数，e.g.

```c++
   auto hash = [](const MyType& o){
       //Calculate and return the hash of `o` ...
   };
   std::unordered_map<MyType,std::string,decltype(hash)>my_map(10,hash);
   std::unordered_set<MyType,decltype(hash)>my_set(10,hash);
```

- 可使用第三方库(`boost::hash_combine`)来组合哈希值以获得在整数上的良好分布，构造 hash 函数 e.g.

```c++
  template <typename T>
  struct std::hash<std::vector<T>>{
      std::size_t operator()(const std::vector<T>& vec) const{
          std::size_t seed = vec.size();
          for(const auto& elem : vec){
              size_t h = element_hash(elem);
              h = ((h >> 16) ^ h) * 0x45d9f3b;
        		h = ((h >> 16) ^ h) * 0x45d9f3b;
        		h = (h >> 16) ^ h;
              seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
          }
          return seed;
      }
      std::hash<T> element_hash{};
  };
```

##### Equality function

相等函数默认使用运算符 `==`，因此自定义类型同上：

1. 重载运算符 `==`，e.g.

```c++
   bool operator==(const MyType& a,const MyType& b){
       //Return true if `a` equals `b`
   }
```

2. 定义函子，e.g.

```c++
   struct Equal{
       bool operator()(const MyType& a,const MyType& b){
           //Return true if `a` equals `b`
       }
   };
   std::unordered_map<MyType,double,std::hash<MyType>,Equal> my_map;
   std::unordered_set<MyType,std::hash<MyType>,Equal> my_set;
```

3. 使用 lambda 函数，e.g.

```c++
   auto equals = [](const MyType& a,const MyType& b){
       //Return true if `a` equals `b`
   };
   std::unordered_map<MyType,double,std::hash<MyType>,decltype(equals)>my_map(10,{},equals);
   std::unordered_set<MyType,std::hash<MyType>,decltype(equals)>my_set(10,{},equals);
```

#### `std::unordered_map<K,V>`

- 同 `std::map`
- `m.load_factor()` 返回映射当前的负载因子(*load factor*)
- `m.max_load_factor(lf)` 将允许的最大负载因子设为 `lf`
- `m.rehash(b)` 确保 `m` 至少有 `b` 个 buckets，并根据新的 bucket count 将元素分配到 buckets 中

#### `std::unordered_set<T>`

- 同 `std::set`
- 与 `unordered_map` 类似

