# Homework2 Report

**姓名：罗睿卿		学号：21214935**

## Task 1 Implementing DQN

### 1.1 算法理解

`DQN`算法通过深度神经网络来拟合`Agent`的`动作-价值函数(Q函数)`，是在`tabular Q-learning`的基础上，结合深度学习来解决$observation\_space \times action\_space$ 过大的问题。将当前的观测值输入Q函数，输入每个动作的价值。在实作中，通常将输出的动作价值转化为每个动作执行的概率分布，通过采样这个概率分布来选取动作。

为了`exploration`和`exploitation`之间的平衡，DQN通常采用 $\epsilon-greedy$ 作为动作选取策略。在训练刚开始时，`exploration`的权重通常会调得很大，比如设置为`1`；随着训练的深入，$\epsilon$ 会逐渐减小，即`expolration`的权重减小，`exploitaiton`的权重逐渐增大，到一个很小的数（为了保留一定的随机性），比如`0.001`。

DQN是off-line policy，训练的数据可以收集到replaybuffer重复使用。

#### 1.2.1 naive-DQN

`Naive-DQN`是上述DQN算法的实现。

#### 1.2.2 double DQN

为了应对`Naive-DQN`中存在的**高估**和**自举**问题，采用double-DQN算法。`double-DQN`唯一的不同在于使用目标网络来估计下一个状态的Q值，缓解了自举，并且消除了高估。

#### 1.2.3 dueling DQN

对决网络将学习Q值，转化为学习**V值(Q值的期望)**和**一个对应动作的优势函数D**，即$Q=V+D$ 。同时为了防止 $V+D$ 在训练过程中适应Q值而无限制的变化，于是将 `D` 减去它的平均值(或最大值)加以限制。即 $Q = V + D - mean(D)$ 。


### 1.2 PongNoFrameskip-v4

增加配置`double_dqn=True`来表示使用`double DQN`，默认为`True`。

增加配置`dueling_dqn=True`来表示使用`dueling DQN`，默认为`True`。

增加配置`exploration_fraction`，`start_e`和`end_e`作为`epsilon-greedy`的设置。`start_e`和`end_e`分别为起始和结束的 $\epsilon$，`exploration_fraction`为采用`epsilon-greedy`策略的比例。

#### 1.2.1 naive DQN

- train:

  ```shell
  # 先修改 main.py 中对应的配置，然后运行
  python main.py --env_name=PongNoFrameskip-v4 --double_dqn=False --dueling_dqn=False
  ```

- reward

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220509211508919.png" alt="image-20220509211508919" style="zoom:67%;" />

- TD_loss

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220509211530739.png" alt="image-20220509211530739" style="zoom:67%;" />

#### 1.2.2 double DQN

- train:

  ```shell
  python main.py --env_name=PongNoFrameskip-v4 --dueling_dqn=False
  ```

- reward

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220508150004177.png" alt="image-20220508150004177" style="zoom:67%;" />

- TD_loss

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220508150021622.png" alt="image-20220508150021622" style="zoom:67%;" />

#### 1.2.3 Dueling network + double DQN

- train:

  ```shell
  python main.py --env_name=PongNoFrameskip-v4
  ```

### 1.3 BreakoutNoFrameskip-v4

#### 1.3.1 naive DQN

- train:

  ```shell
  python main.py --env_name=BreakoutNoFrameskip-v4 --double_dqn=False --dueling_dqn=False
  ```

- reward

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220510200416724.png" alt="image-20220510200416724" style="zoom:67%;" />


- TD_loss

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220510200437959.png" alt="image-20220510200437959" style="zoom:67%;" />

#### 1.3.2 double DQN

- train:

  ```shell
  python main.py --env_name=BreakoutNoFrameskip-v4 --double_dqn=True --dueling_dqn=False

- reward

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220508145411094.png" alt="image-20220508145411094" style="zoom:67%;" />

- TD_loss

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220508145547088.png" alt="image-20220508145547088" style="zoom:67%;" />

#### 1.3.3 Dueling network + double DQN

- train:

  ```shell
  python main.py --env_name=BreakoutNoFrameskip-v4 --double_dqn=True --dueling_dqn=True

- reward

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220512103629817.png" alt="image-20220512103629817" style="zoom:67%;" />

- TD_loss

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220512103658270.png" alt="image-20220512103658270" style="zoom:67%;" />

## Task 2 Implementing Policy Gradient

### 2.1 算法理解

策略梯度方法不同DQN，DQN是学习一个动作价值函数，通过动作的价值来选择动作。策略梯度方法直接学习一个策略$\pi$，将观测输入到策略网络，输出每个动作的概率分布，通过采样这个概率分布来选择动作。

policy-gradient是on-line policy，使用当前policy采样得到的轨迹数据，一旦更新policy之后就不能再使用了。

#### 2.1.1 reinforce

reinforce使用蒙特卡洛来近似V，先使智能体收集一个完整的轨迹，计算出实际的回报值。通过这个实际的回报去优化策略$\pi$。

#### 2.1.2 A2C

由于reinforce算法需要收集一个完整的轨迹才能更新一次policy，数据利用率很低，更新很慢。A2C使用另外一个神经网络V来拟合每一步的期望回报，这样就可以每走一步就更新一次策略网络了。

### 2.2 CartPole-v0

#### 2.2.1 REINFORCE

- train:

  ```shell
  # 先修改 main.py 中对应的配置，然后运行
  python main.py --env_name=CartPole-v0
  ```

- reward:

<img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220508153756185.png" alt="image-20220508153756185" style="zoom:67%;" />

#### 2.2.2 证明：advantage function 不会影响策略梯度

策略梯度方法通过最大化目标函数 $J(\theta) = \mathbb E_S(V_{\pi}(S))$ ，训练出策略网络 $\pi(a|s;\theta)$ 。通过策略梯度 $\triangledown_\theta J(\theta)$ 来更新参数 $\theta$ 。

已知带有`advantage function`的策略梯度为：
$$
\triangledown_\theta J(\theta) = \mathbb E_S[\mathbb E_{A\sim\pi(\cdot|S;\theta)}[(Q_\pi(S,A) -b)\cdot\triangledown_\theta \ln \pi(A|S;\theta)]]
$$
其中 $b$ 为动作价值函数 $Q_\pi(S, A)$ 的 baseline 。**设 $b$ 是任意的函数，只要与动作 $A$ 无关即可**。 **$b$ 通常取为状态价值函数 $V\pi(S)$** 。上述策略梯度可以写为：
$$
\triangledown_\theta J(\theta) = \mathbb E_S[\mathbb E_{A\sim\pi(\cdot|S;\theta)}[Q_\pi(S,A)\cdot\triangledown_\theta \ln \pi(A|S;\theta)]] + \mathbb E_S[\mathbb E_{A\sim\pi(\cdot|S;\theta)}[b\cdot\triangledown_\theta \ln \pi(A|S;\theta)]]
$$
其中第一项即为原本的策略梯度定理。如果 $b$ 对策略梯度没有影响，那么第二项应该等于。现在我们证明：
$$
\mathbb E_S[\mathbb E_{A\sim\pi(\cdot|S;\theta)}[b\cdot\triangledown_\theta \ln \pi(A|S;\theta)]] = 0
$$
当 $S = s$ 时， 括号内为：
$$
\begin{eqnarray}    \label{eq}
{\mathbb E_{A\sim\pi(\cdot|s;\theta)}[b\cdot\triangledown_\theta \ln \pi(A|s;\theta)]}&=& {b \cdot\mathbb E_{A\sim\pi(\cdot|s;\theta)}[\frac {\partial \ln \pi (A|s;\theta)}{\partial \theta}]}    \\
&=& {b \cdot \sum_{a\in \mathcal A}\pi(a|s;\theta) \frac {\partial \ln \pi (a|s;\theta)}{\partial \theta}}     \\
&=& {b \cdot \sum_{a\in \mathcal A}\pi(a|s;\theta) \frac {1}{\pi(a|s;\theta)} \frac {\partial \pi (a|s;\theta)}{\partial \theta}}     \\
&=& b \cdot \sum_{a\in \mathcal A}\frac {\partial \pi (a|s;\theta)}{\partial \theta}
\end{eqnarray}
$$
上式右边的连加是关于 $a$ 求的，而偏导是关于 $\theta$ 求的，因此可以把连加放入偏导内部：
$$
\mathbb E_{A\sim\pi(\cdot|s;\theta)}[b\cdot\triangledown_\theta \ln \pi(A|s;\theta)] = b \cdot \sum_{a\in \mathcal A}\frac {\partial \pi (a|s;\theta)}{\partial \theta} = b \cdot \frac {\partial}{\partial \theta}\sum_{a\in \mathcal A}\pi (a|s;\theta)
$$
而：
$$
\sum_{a\in \mathcal A}\pi (a|s;\theta) = 1
$$
因此：
$$
\mathbb E_{A\sim\pi(\cdot|s;\theta)}[b\cdot\triangledown_\theta \ln \pi(A|s;\theta)] = b \cdot \frac{\partial 1}{\partial \theta} = 0
$$

#### 2.2.3 A2C

- train

  ```shell
  # 先修改 main.py 中对应的配置，然后运行
  python main.py --env_name=CartPole-v0
  ```

- reward

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220511202356512.png" alt="image-20220511202356512" style="zoom:67%;" />



## Task 3 Implementing DDPG

### 3.1 LunarLanderContinuous-v2

- train

  ```shell
  # 先修改 main.py 中对应的配置，然后运行
  python main.py --env_name=LunarLanderContinuous-v2
  ```

- reward

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220513125928661.png" alt="image-20220513125928661" style="zoom:67%;" />
