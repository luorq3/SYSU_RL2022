# Homework2 Report

**姓名：罗睿卿		学号：21214935**

## Task 1 Implementing DQN

### 1.1 PongNoFrameskip-v4

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

- reward
- TD_loss

### 1.2 BreakoutNoFrameskip-v4

#### 1.2.1 naive DQN

- train:

  ```shell
  python main.py --env_name=BreakoutNoFrameskip-v4 --double_dqn=False --dueling_dqn=False
  ```

- reward

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220510200416724.png" alt="image-20220510200416724" style="zoom:67%;" />


- TD_loss

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220510200437959.png" alt="image-20220510200437959" style="zoom:67%;" />

#### 1.2.2 double DQN

- train:

  ```shell
  python main.py --env_name=BreakoutNoFrameskip-v4 --double_dqn=True --dueling_dqn=False

- reward

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220508145411094.png" alt="image-20220508145411094" style="zoom:67%;" />

- TD_loss

  <img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220508145547088.png" alt="image-20220508145547088" style="zoom:67%;" />

#### 1.2.3 Dueling network + double DQN

- train:

  ```shell
  python main.py --env_name=BreakoutNoFrameskip-v4 --double_dqn=True --dueling_dqn=True

- reward

- TD_loss

## Task 2 Implementing Policy Gradient

### 2.1 CartPole-v0

#### 2.1.1 REINFORCE

- train:

  ```shell
  # 先修改 main.py 中对应的配置，然后运行
  python main.py --env_name=CartPole-v0
  ```

- reward:

<img src="/Users/luoruiqing/Library/Application Support/typora-user-images/image-20220508153756185.png" alt="image-20220508153756185" style="zoom:67%;" />

#### 2.1.2 A2C







