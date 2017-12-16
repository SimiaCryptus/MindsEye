# BinarySumLayer
## Float_Add
### Json Serialization
Code from [JsonTest.java:36](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/JsonTest.java#L36) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    if ((echo == null)) throw new AssertionError("Failed to deserialize");
    if ((layer == echo)) throw new AssertionError("Serialization did not copy");
    if ((!layer.equals(echo))) throw new AssertionError("Serialization not equal");
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer",
      "id": "1cd6dda2-8fb4-4675-b24d-3e978fc68f3e",
      "isFrozen": false,
      "name": "BinarySumLayer/1cd6dda2-8fb4-4675-b24d-3e978fc68f3e",
      "rightFactor": 1.0,
      "leftFactor": 1.0,
      "precision": "Float"
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n--------------------\nDerivative: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(),
      Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 1.636 ], [ -1.368 ] ],
    	[ [ -1.104 ], [ 1.924 ] ]
    ],
    [
    	[ [ 1.12 ], [ -1.212 ] ],
    	[ [ -0.716 ], [ 1.204 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ],
    [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.01 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.928 ], [ 1.364 ] ],
    	[ [ 0.416 ], [ -1.988 ] ]
    ],
    [
    	[ [ -1.66 ], [ 0.72 ] ],
    	[ [ -1.776 ], [ -0.892 ] ]
    ]
    Inputs Statistics: {meanExponent=0.08435777764382732, negative=2, min=-1.988, max=-1.988, mean=-0.534, count=4.0, positive=2, stdDev=1.4630666423645917, zeros=0},
    {meanExponent=0.06931210007250718, negative=3, min=-0.892, max=-0.892, mean=-0.902, count=4.0, positive=1, stdDev=0.9961706681086329, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=4.0, positive=0, stdDev=0.0, zeros=4}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.928 ], [ 1.364 ] ],
    	[ [ 0.416 ], [ -1.988 ] ]
    ]
    Value Statistics: {meanExponent=0.08435777764382732, negative=2, min=-1.988, max=-1.988, mean=-0.534, count=4.0, positive=2, stdDev=1.4630666423645917, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {me
```
...[skipping 739 bytes](etc/51.txt)...
```
    egative=3, min=-0.892, max=-0.892, mean=-0.902, count=4.0, positive=1, stdDev=0.9961706681086329, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=16.0, positive=0, stdDev=0.0, zeros=16}
    Measured Feedback: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=16.0, positive=0, stdDev=0.0, zeros=16}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=16.0, positive=0, stdDev=0.0, zeros=16}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (32#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (32#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.30 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[100, 100, 1]
    	[100, 100, 1]
    Performance:
    	Evaluation performance: 0.011173s +- 0.000373s [0.010587s - 0.011573s]
    	Learning performance: 0.033072s +- 0.004735s [0.027803s - 0.039912s]
    
```

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.144 ], [ 1.46 ], [ -1.18 ], [ 0.008 ], [ -0.32 ], [ -0.84 ], [ 0.648 ], [ 1.528 ], ... ],
    	[ [ -0.572 ], [ 1.768 ], [ 1.172 ], [ 1.232 ], [ 1.128 ], [ 1.136 ], [ 1.432 ], [ 0.548 ], ... ],
    	[ [ 0.964 ], [ -1.084 ], [ 0.96 ], [ -1.52 ], [ -1.912 ], [ 0.572 ], [ -1.688 ], [ -1.512 ], ... ],
    	[ [ 0.496 ], [ -0.156 ], [ 0.744 ], [ -1.948 ], [ -1.796 ], [ 1.276 ], [ -0.3 ], [ -1.904 ], ... ],
    	[ [ 0.42 ], [ -1.616 ], [ -1.744 ], [ -1.468 ], [ -0.244 ], [ 1.272 ], [ -0.556 ], [ 1.164 ], ... ],
    	[ [ -0.4 ], [ 1.4 ], [ 0.084 ], [ -0.456 ], [ 0.48 ], [ -1.172 ], [ 0.332 ], [ -0.784 ], ... ],
    	[ [ 0.5 ], [ 1.584 ], [ -0.924 ], [ -0.152 ], [ 0.652 ], [ -1.424 ], [ -1.892 ], [ 1.44 ], ... ],
    	[ [ -0.648 ], [ 0.892 ], [ 0.056 ], [ -1.472 ], [ -0.036 ], [ 0.048 ], [ 1.648 ], [ -1.308 ], ... ],
    	...
    ]
    [
    	[ [ -0.984 ], [ -1.876 ], [ -1.94 ], [ -1.028 ], [ -0.284 ], [ 1.5 ], [ -1.776 ], [ -0.98 ], ... ],
    	[ [ -0.328 ], [ -1.32 ], [ -0.396 ], [ -1.12 ], [ -0.928 ], [ -1.216 ], [ 0.808 ], [ -0.444 ], ... ],
    	[ [ 1.16 ], [ -0.436 ], [ 1.504 ], [ -0.16 ], [ 1.72 ], [ 1.76 ], [ -1.064 ], [ 1.988 ], ... ],
    	[ [ -0.432 ], [ 0.256 ], [ -1.504 ], [ -0.748 ], [ 1.216 ], [ 1.768 ], [ -0.2 ], [ 0.272 ], ... ],
    	[ [ -0.108 ], [ 0.832 ], [ -0.504 ], [ -0.588 ], [ -1.064 ], [ -1.808 ], [ -0.676 ], [ 0.056 ], ... ],
    	[ [ 1.112 ], [ -1.472 ], [ 0.612 ], [ 0.88 ], [ 1.56 ], [ 0.608 ], [ 0.912 ], [ -1.372 ], ... ],
    	[ [ 1.696 ], [ -0.3 ], [ -1.008 ], [ 0.528 ], [ -0.636 ], [ 1.94 ], [ -0.632 ], [ -0.708 ], ... ],
    	[ [ -0.044 ], [ -1.6 ], [ -0.96 ], [ -0.124 ], [ 1.764 ], [ 1.304 ], [ -1.972 ], [ 1.02 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:300](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L300) executed in 0.00 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new QuadraticSearch())
      .setOrientation(new GradientDescent())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```

Returns: 

```
    0.0
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.272 ], [ 1.692 ], [ -1.928 ], [ 1.828 ], [ -1.264 ], [ 0.932 ], [ -1.508 ], [ -1.836 ], ... ],
    	[ [ 0.692 ], [ 0.748 ], [ 1.964 ], [ -1.14 ], [ 1.456 ], [ 0.492 ], [ -1.828 ], [ -1.092 ], ... ],
    	[ [ -1.968 ], [ -1.948 ], [ 0.148 ], [ -1.952 ], [ 0.156 ], [ -0.136 ], [ 1.116 ], [ -0.576 ], ... ],
    	[ [ -1.848 ], [ 1.784 ], [ -0.1 ], [ -0.424 ], [ 1.096 ], [ -0.84 ], [ -0.036 ], [ 0.896 ], ... ],
    	[ [ 1.868 ], [ 1.124 ], [ 0.548 ], [ 1.92 ], [ 1.264 ], [ 1.62 ], [ 0.668 ], [ -1.708 ], ... ],
    	[ [ -0.564 ], [ -1.024 ], [ -0.5 ], [ 1.02 ], [ 0.68 ], [ 0.964 ], [ -1.088 ], [ -1.48 ], ... ],
    	[ [ -0.96 ], [ 0.6 ], [ -0.816 ], [ -1.292 ], [ 0.148 ], [ 1.44 ], [ 1.268 ], [ -0.552 ], ... ],
    	[ [ 1.184 ], [ 0.14 ], [ 1.252 ], [ -0.58 ], [ -0.332 ], [ -1.388 ], [ -1.256 ], [ 0.964 ], ... ],
    	...
    ]
    [
    	[ [ -0.096 ], [ 1.168 ], [ 0.444 ], [ 1.812 ], [ -0.108 ], [ -0.096 ], [ -1.616 ], [ -0.144 ], ... ],
    	[ [ 1.224 ], [ 1.376 ], [ 1.3 ], [ 0.864 ], [ -1.22 ], [ 0.92 ], [ -0.016 ], [ 0.2 ], ... ],
    	[ [ 0.164 ], [ -0.756 ], [ 1.728 ], [ -1.152 ], [ 1.808 ], [ -1.8 ], [ -1.528 ], [ 1.28 ], ... ],
    	[ [ -1.392 ], [ -1.068 ], [ -1.2 ], [ -0.124 ], [ 0.488 ], [ 0.036 ], [ -0.308 ], [ -0.572 ], ... ],
    	[ [ -1.164 ], [ 0.832 ], [ -0.744 ], [ 1.688 ], [ -1.124 ], [ 0.792 ], [ 1.852 ], [ 0.172 ], ... ],
    	[ [ -1.796 ], [ -0.236 ], [ 0.036 ], [ -1.744 ], [ 0.984 ], [ -0.836 ], [ 0.108 ], [ 1.64 ], ... ],
    	[ [ 1.828 ], [ -1.172 ], [ 0.848 ], [ -1.268 ], [ 0.384 ], [ -1.068 ], [ -0.748 ], [ 1.092 ], ... ],
    	[ [ -1.304 ], [ -0.944 ], [ -0.324 ], [ 0.24 ], [ 0.26 ], [ 1.556 ], [ 0.596 ], [ 0.164 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:324](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L324) executed in 0.00 seconds: 
```java
    return new IterativeTrainer(trainable)
      .setLineSearchFactory(label -> new ArmijoWolfeSearch())
      .setOrientation(new LBFGS())
      .setMonitor(monitor)
      .setTimeout(30, TimeUnit.SECONDS)
      .setMaxIterations(250)
      .setTerminateThreshold(0)
      .run();
```

Returns: 

```
    0.0
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.272 ], [ 1.692 ], [ -1.928 ], [ 1.828 ], [ -1.264 ], [ 0.932 ], [ -1.508 ], [ -1.836 ], ... ],
    	[ [ 0.692 ], [ 0.748 ], [ 1.964 ], [ -1.14 ], [ 1.456 ], [ 0.492 ], [ -1.828 ], [ -1.092 ], ... ],
    	[ [ -1.968 ], [ -1.948 ], [ 0.148 ], [ -1.952 ], [ 0.156 ], [ -0.136 ], [ 1.116 ], [ -0.576 ], ... ],
    	[ [ -1.848 ], [ 1.784 ], [ -0.1 ], [ -0.424 ], [ 1.096 ], [ -0.84 ], [ -0.036 ], [ 0.896 ], ... ],
    	[ [ 1.868 ], [ 1.124 ], [ 0.548 ], [ 1.92 ], [ 1.264 ], [ 1.62 ], [ 0.668 ], [ -1.708 ], ... ],
    	[ [ -0.564 ], [ -1.024 ], [ -0.5 ], [ 1.02 ], [ 0.68 ], [ 0.964 ], [ -1.088 ], [ -1.48 ], ... ],
    	[ [ -0.96 ], [ 0.6 ], [ -0.816 ], [ -1.292 ], [ 0.148 ], [ 1.44 ], [ 1.268 ], [ -0.552 ], ... ],
    	[ [ 1.184 ], [ 0.14 ], [ 1.252 ], [ -0.58 ], [ -0.332 ], [ -1.388 ], [ -1.256 ], [ 0.964 ], ... ],
    	...
    ]
    [
    	[ [ -0.096 ], [ 1.168 ], [ 0.444 ], [ 1.812 ], [ -0.108 ], [ -0.096 ], [ -1.616 ], [ -0.144 ], ... ],
    	[ [ 1.224 ], [ 1.376 ], [ 1.3 ], [ 0.864 ], [ -1.22 ], [ 0.92 ], [ -0.016 ], [ 0.2 ], ... ],
    	[ [ 0.164 ], [ -0.756 ], [ 1.728 ], [ -1.152 ], [ 1.808 ], [ -1.8 ], [ -1.528 ], [ 1.28 ], ... ],
    	[ [ -1.392 ], [ -1.068 ], [ -1.2 ], [ -0.124 ], [ 0.488 ], [ 0.036 ], [ -0.308 ], [ -0.572 ], ... ],
    	[ [ -1.164 ], [ 0.832 ], [ -0.744 ], [ 1.688 ], [ -1.124 ], [ 0.792 ], [ 1.852 ], [ 0.172 ], ... ],
    	[ [ -1.796 ], [ -0.236 ], [ 0.036 ], [ -1.744 ], [ 0.984 ], [ -0.836 ], [ 0.108 ], [ 1.64 ], ... ],
    	[ [ 1.828 ], [ -1.172 ], [ 0.848 ], [ -1.268 ], [ 0.384 ], [ -1.068 ], [ -0.748 ], [ 1.092 ], ... ],
    	[ [ -1.304 ], [ -0.944 ], [ -0.324 ], [ 0.24 ], [ 0.26 ], [ 1.556 ], [ 0.596 ], [ 0.164 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:96](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L96) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:99](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L99) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

