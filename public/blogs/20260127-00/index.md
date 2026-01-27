## 直观理解

1. 有一个Generator，一个Discriminator，Generator吃进一个vector，吐出一张image，Discriminator吃进一个image，吐出一个scalar
2. 用database训练Discriminator，使Discriminator吃进database的image时吐出高scalar，吃进Generator吐出的image时吐出低scalar
3. 将Discriminator吐出的scalar返回给Generator，让他进行优化，使吐出的image能让Discriminator给出高scalar
4. 根据Generator对DIscriminator进行优化，使其吃进优化后Generator生成的image后吐出低scalar
5. 如此反复对Generator和Discriminator进行优化，最终Generator生成的image跟database的image相差不大

- 举个例子，Generator是学生，Discriminator是老师，学生要画二次元头像，老师见过大量二次元头像，学生现在是一年级，画了一张二次元头像给一年级老师，老师指出一个问题，打出低分，学生改进，升到二年级了，画了一张改进过的二次元头像给二年级老师，老师又指出一个问题，打出低分，学生再改进，如此如此，直到学生博士毕业，画的二次元头像与导师所见的头像相差不大了，导师打出高分

- 如果只用Generator，不用Discriminator，那就类似VAE的decoder，是以像素级别判断生成图片，也就是没有“大局观”，很难判断准确，导致生成图像不准；如果只用Discriminator生成，那就是求解`x'=arg max D(x)`，很难求，因此生成消耗资源很大，但是他可以从整张图片出发进行判断，有“大局观”，因此判断准确，所以Generator+Discriminator效果会更好

  