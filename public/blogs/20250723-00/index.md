> 由于markdown只有六级标题，因此思维导图只能展开六层，所以有些最底层我就直接略过或用`——`表示了

![](/blogs/20250723-00/094690bb939d7b0e.svg)

# OSI protocol stack(协议栈)

## Application

### HTTP

#### 请求、响应

#### cookie

#### Web缓存器(代理服务器)

#### CDN

### SMTP、POP3、IMAP

### FTP

#### 控制连接

#### 数据连接

### DNS(UDP)

#### 层次结构

##### 根DNS服务器

##### TLD(顶级域)服务器

##### 权威DNS服务器

##### 本地DNS服务器

#### DNS缓存

##### RR(资源记录)

###### (Name,Value,Type,TTL)

### P2P

#### 最稀缺优先(rarest first)

#### 一报还一报(tit-for-tat)

#### DHT(分布式散列表)[(键，值)对]

##### 环形DHT

##### 对等方扰动

### NFS(UDP)

### socket(套接字)

## Transport

### FSM(有限状态机)

### TCP

#### SYN、SYN/ACK、ACK

#### 估算RTT

##### SampleRTT

##### EstimatedRTT(EWMA，指数加权移动平均)

##### DevRTT(RTT偏差)

#### 拥塞控制

##### GBN协议(滑动窗口协议)

##### SR协议(选择重传协议)

##### 快速重传

##### TFRC(TCP友好速率控制)

#### TCP拥塞控制算法

##### 慢启动

###### cwnd(拥塞窗口)、ssthresh(慢启动阈值)、safe区域、dangerous区域、TCP管道

###### AIMD(加性增，乘性减)

##### 拥塞避免

##### 快速恢复

###### TCP Tahoe

###### TCP Reno

#### 公平性

##### Reno算法

##### Vegas算法

##### CUBIC算法

##### AIMD的公平性收敛

### UDP

### DCCP(数据报拥塞控制协议)

### SCTP(流控制传输协议)

## Network

### IP

#### 数据报格式

#### IPv4编址

##### CIDR(无类别域间路由选择)

##### DHCP(动态主机配置协议)[即插即用协议]

###### DHCP discover

###### DHCP offer

###### DHCP request

###### DHCP ACK

#### NAT(网络地址转换)

##### 基础型NAT

###### 静态NAT

###### 动态NAT

##### NAPT

###### 完全锥型NAT

###### 受限锥型NAT

###### 端口受限NAT

###### 对称型NAT

##### NAT打洞/穿透

###### 建立UDP隧道

###### 注册信息包

###### 直接发送UDP报文

#### IPv6

### ICMP

#### ping

#### traceroute

### 路由选择算法

#### 单播

##### 根据全局或分散分类

###### 全局式路由选择算法——LS(链路状态)算法——Dijkstra算法

###### 分散式路由选择算法——DV(距离向量)算法——Bellman-Ford算法、路由选择环路和毒性逆转

##### 根据静态或动态分类

###### 静态路由选择算法

###### 动态路由选择算法

##### 根据负载敏感度分类

###### 负载敏感算法

###### 负载迟钝算法(RIP、OSPF、BGP)

#### 多播

##### 广播路由选择算法(broadcast routing)

###### 无控制洪泛(flooding)

###### 受控洪泛 之 序号控制洪泛

###### 受控洪泛 之 RPF(反向路径转发)，或称RPB(反向路径广播)

###### STP(生成树协议)

##### 组播路由选择算法(multicast routing)

###### IGMP

###### DVMRP(距离向量组播路由选择协议)

### AS

#### 热土豆路由选择(hot potato routing)

#### RIP(AS内部)[UDP]

#### OSPF(AS内部)

#### IS-IS(AS内部)

#### BGP4(AS之间)

## Data Link

### 差错检测、纠正

#### FEC(向前纠错)

#### 奇偶校验

#### checksum(检验和)

#### CRC(循环冗余检测)

### 多路访问协议

#### 信道划分协议

##### TDM(时分多路复用)

##### FDM(频分多路复用)

##### CDMA(码分多址)

#### 随机接入协议(MAC协议)

##### ALOHA

##### CSMA、CSMA/CD

#### 轮流协议

##### 轮询协议

##### 令牌传递协议

### 交换局域网

#### ARP

##### MAC地址(局域网地址、以太网地址、物理地址)

###### NIC唯一标识符

#### Ethernet(以太网技术)

##### 10BASE-T、100BASE-T、1000BASE-T、10GBASE-T

##### 以太网帧、使用CRC校验

#### 链路层交换机(即插即用设备)

##### 过滤、转发

##### 自学习

##### 交换机毒化

#### VLAN(虚拟局域网)

### 链路虚拟化

#### MPLS(多协议标签交换)

##### VPN

## Physical