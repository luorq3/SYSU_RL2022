# Homework1 Report

**姓名：罗睿卿		学号：21214935**

## Task1 Frozen Lake MDP

### 1.Policy iteration

#### 1.1 Introduction

**策略迭代**是基于动态规划思想的用来解决MDP问题的一种方法，属于`Model_based`算法。算法主要分为两部分：**策略评估**以及**策略提升**。

**策略评估**是指，固定住当前的策略不变，然后估计此策略下，各个状态的价值。即给定策略函数，来估计状态价值函数。

**策略改进**是指，得到当前策略下的状态价值后，进一步计算动作状态价值函数`Q`，通过对`Q做贪心搜索来改进策略。



#### 1.2 Algorithm

##### 1.2.1 Policy evaluation

策略评估的理论依据是`Bellman Expectation Equation`：
$$
V_{\pi}(s)=\mathbb{E}_{\pi}[r_{t+1} + \gamma V_{\pi}(s_{t+1})|s_t = s]
$$
通过上述的算法，迭代足够的次数后，$V_{\pi}$ 可以收敛到一定的范围，这样就完成了对策略 $\pi$ 的评估。

##### 1.2.2 Policy improvement

当基于一个策略的状态价值函数收敛之后，就需要对这个策略进行提升。策略提升的理论依据是`Bellman Optimality Equation`：
$$
V^*(s)=\max_a(R(s,a) + \gamma \sum_{s^{\prime} \in S}p(s^{\prime}|s,a)V^*(s^{\prime}))
$$
贝尔曼最优方程通过贪心的思想，用价值函数最大的动作来替代前一个策略。



#### 1.3 Method

利用策略迭代解决Frozen Lake问题的代码实现如下：

1. 初始化一个初始策略（此处为全0），对于这个初始策略，重复执行`策略评估`和`策略提升`这两个步骤，直到**策略收敛**：

   ```python
   def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
     # Initialize the policy
     policy = np.zeros(nS, dtype=int)
     
     while True:
       # Policy evaluation
       value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
       # Policy improvement
       new_policy = policy_improvement(P, nS, nA, value_function, policy, gamma)
           
       # If the policy converges, break out loops
       if np.all(policy == new_policy):
         break
       else:  # Else, if the policy not covverges yet, iterate over new policy
         policy = new_policy.copy()
   	
     # Policy converges, return the policy and value function
     return value_function, policy
   ```

2. 执行**策略评估**，返回基于当前策略的价值函数：

   ```python
   def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
   	# Initialize the value function based on given policy
   	value_function = np.zeros(nS)
   	# Temporary value function, storage `next value function` temporarily
   	value_function_ = value_function.copy()
   
   	while True:
   		for s in range(nS):
   			a = policy[s]
   
   			val = 0
   			for prob, s_, r, done in P[s][a]:
   				val += prob * (r + gamma * value_function[s_])
   
   			value_function_[s] = val
   
   		if np.abs(np.max(value_function_ - value_function)) < tol:
   			break
   		else:
   			value_function = value_function_.copy()
   
   	return value_function
   ```

3. 进行**策略改进**，返回最新的策略：

    ```python
    def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    
    	new_policy = np.zeros(nS, dtype='int')
    
    	for s in range(nS):
    		action_val = np.zeros(nA)
    		for a in range(nA):
    			val = 0
    			for prob, s_, r, done in P[s][a]:
    				val += prob * (r + gamma * value_from_policy[s_])
    			action_val[a] = val
    		new_policy[s] = np.argmax(action_val)
    
    	return new_policy
    ```



### 2.Value iteration

#### 2.1 Introduction

**价值迭代**是基于动态规划思想的用来解决MDP问题的`Model_based`算法。

**策略迭代**的目标是去优化一个策略，价值函数只是辅助策略收敛，不参与决策。价值迭代与之不同，根据价值函数去选择动作。



#### 2.2 Algorithm

价值迭代采用`Bellman Expectation Equation`计算**状态价值函数**：
$$
Q_{k+1}(s,a) = R(s,a) + \gamma \sum_{s^{\prime} \in S}p(s^{\prime}|s,a)V_k(s^\prime)
\\
V_{k+1}(s) = \max_a Q_{k+1}(s,a)
$$
在迭代后提取最优策略：
$$
\pi(s) = \arg \max_a R(s,a) + \gamma \sum_{s^{\prime} \in S}p(s^{\prime}|s,a)V_{k+1}(s^\prime)
$$



#### 2.3 Method

利用价值迭代解决Frozen Lake问题的代码实现如下：

```python
def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)

	# 用于记录更新之后的value_function
	value_function_ = value_function.copy()

	# 价值迭代
	while True:
		# 依次更新每个状态的最优价值
		for s in range(nS):
			# 计算每个动作的价值
			action_values = np.zeros(nA)
			for a in range(nA):
				state_value = 0
				for prob, s_, r, done in P[s][a]:
					state_value += prob * (r + gamma * value_function[s_])
				action_values[a] = state_value
				# 选出最优的动作，更新 value_function_ 以及 policy
				action_star = np.argmax(action_values)
				value_function_[s] = action_values[action_star]
				policy[s] = action_star

		# stopping tolerance
		if np.abs(np.max(value_function_ - value_function)) < tol:
			break
		else:
			# 将最新的 value_function_ 赋值给 value_function
			value_function = value_function_.copy()

	return value_function, policy
```



## Task2 Test Environment

- Maximum sum of rewards: **6.1**

- Trajectort of maximum sum of rewards: 
  $$
  s_0=0,a_0=2,
  \\
  r_0=0.0,s_1=2,a_1=1,
  \\
  r_1=3.0,s_2=1,a_2=2,
  \\
  r_2=0.0,s_3=2,a_3=1,
  \\
  r_3=3.0,s_4=1,a_4=0,
  \\
  r_4=3.0,s_5=0,a_5=0,
  \\
  r_5=0.1
  $$
  



## Task3 Tabular Q-Learning

### 1.Introduction

$\epsilon - \mathrm{greddy}$ 是`Q-learning`中获取动作的一种策略。即以$1-\epsilon$的概率采取`Q`值最高的动作，以$\epsilon$的概率从动作集合中均匀抽样一个动作。这体现了强化学习中`exploration`和`exploitation`的权衡。

在迭代刚刚开始时，$\epsilon$应该比较大，即多`exploration`；随着迭代次数的增加，价值函数趋近于收敛，$\epsilon$应该随之减小，即`exploitation`的比重变大。



### 2.Method

#### 2.1 get_action

```python
def get_action(self, best_action):
	
  if np.random.random() < self.epsilon:
    return self.env.action_space.sample()

  return best_action
```

#### 2.2 update

```python
def update(self, t):

  if t > self.nsteps:
    self.epsilon = self.eps_end
  else:
    self.epsilon = self.eps_begin - (self.eps_begin - self.eps_end) * t / self.nsteps
```



## Task4 Maze Example

### 1. Q-learning

#### 1.1 Introduction

当动作空间和状态空间足够小时，可以使用`Tabular Q-learning`方法。`Tabular Q-learning`的核心是用表格记录每一个状态对应执行每个动作的最优分数，**即使用表格来近似动作价值函数 $Q_*$** 。将所有状态作为行，所有动作作为列，对于一个具有`m`个动作，`n`个状态的环境，那么`Tabular Q-learning`表格就应该是`m*n`的矩阵。

做决策时，根据当前的`Q`表格，使用此公式选择动作：
$$
a_t = \arg\max_{a \in \mathcal A}Q_*(s_t, a)
$$
根据状态价值函数表格做决策，不存在策略函数。



#### 1.2 Algorithm

此方法理论依据来源于另一种形式的`Bellman Optimality Equation`：
$$
Q_*(s_t,a_t)=\mathbb E_{S_{t+1}\thicksim p(\cdot|(s_t, a_t))}[R_t + \gamma \cdot \max_{A \in \mathcal A}Q_*(S_{t+1}, A)|S_t = s_t, A_t = a_t]
$$
在$s_t, a_t$已知的情况下，通过对方程右边的期望做蒙特卡洛近似：
$$
r_t + \gamma \cdot \max_{a \in \mathcal A}Q_*(s_{t+1}, a)
$$
再进一步近似为：
$$
y_t = r_t + \gamma \cdot \max_{a \in \mathcal A}Q(s_{t+1}, a)\approx Q_*(s_t,a_t)
$$
即可近似计算当前动作状态价值$Q_*(s_t,a_t)$。

更新$Q$：
$$
Q(s_t, a_t) \gets(1-\alpha)\cdot Q(s_t,a_t) + \alpha \cdot y_t
$$

#### 1.3 Method

##### 1.3.1 创建`Q table`

```python
self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
```

##### 1.3.2 choose_action ($\epsilon-greddy$)

```python
def choose_action(self, observation):

  self.check_state_exist(observation)

  if np.random.random() < self.epsilon:
    action = np.random.choice(self.actions)
  else:
    series = pd.Series(self.q_table.loc[observation])
    action = series.argmax()
    action = series.index[action]

  return action
```

##### 1.3.3 learn

```python
def learn(self, s, a, r, s_):

  self.check_state_exist(s_)
  q_s = self.q_table.loc[s, a]
  if s_ == 'terminal':
    q_s_ = 0
  else:
    q_s_ = self.q_table.loc[s_, :].max()
    td_error = r + self.gamma * q_s_ - q_s
    self.q_table.loc[s, a] = q_s + self.lr * td_error
```

##### 1.3.4 check_state_exist

```python
def check_state_exist(self, state):

  if state not in self.q_table.index:
    self.q_table.loc[state] = pd.Series(np.zeros(len(self.actions)), index=self.actions)
```

### 2. Sarsa

#### 2.1 Introduction

**`Sarsa`与`Q-learning`不同，后者用表格近似最优动作状态价值函数，而`Sarsa`用表格近似某一个策略 $\pi$ 的动作状态价值函数 $Q_{\pi}$。**

所以，在`Sarsa`中`Q`表格是与一个策略 $\pi$ 相关的。当策略改变时，`Q`表格也会发生变化。显然，**在`Sarsa`中不能使用过时的`Q`表格来指导当前的迭代更新，属于同策略(`on-policy`)；`Q-learning`则与之相反，属于异策略(`off-policy`)**。



#### 2.2 Algorithm

`Sarsa`算法由下面的贝尔曼方程推导得出：
$$
Q_\pi(s_t,a_t)=\mathbb E_{S_{t+1}, A_{t+1}}[R_t + \gamma \cdot Q_\pi(S_{t+1}, A_{t+1})|S_t = s_t, A_t = a_t]
$$
对方程右边做近似，得到`TD target`：
$$
y_t \triangleq r_t + \gamma \cdot q(s_{t+1}, a_{t+1})
$$
更新$Q_\pi$：
$$
q(s_t, a_t) \gets(1-\alpha)\cdot q(s_t,a_t) + \alpha \cdot y_t
$$

#### 2.3 Method

##### 1.3.1 创建`Q table`

```python
self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
```

##### 1.3.2 choose_action ($\epsilon-greddy$)

```python
def choose_action(self, observation):

  self.check_state_exist(observation)

  self.check_state_exist(observation)

  if np.random.random() < self.epsilon:
    action = np.random.choice(self.actions)
  else:
    series = pd.Series(self.q_table.loc[observation])
    action = series.argmax()
    action = series.index[action]

  return action
```

##### 1.3.3 learn

```python
def learn(self, s, a, r, s_):

  self.check_state_exist(s_)
  q_s = self.q_table.loc[s, a]
  if s_ == 'terminal':
    q_s_ = 0
  else:
    a_ = self.choose_action(s_)
    q_s_ = self.q_table.loc[s_, a_]
  td_error = r + self.gamma * q_s_ - q_s
  self.q_table.loc[s, a] = q_s + self.lr * td_error
```

##### 1.3.4 check_state_exist

```python
def check_state_exist(self, state):

  if state not in self.q_table.index:
    self.q_table.loc[state] = pd.Series(np.zeros(len(self.actions)), index=self.actions)
```

### 





