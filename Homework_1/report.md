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

### Q-learning



### Sarsa













