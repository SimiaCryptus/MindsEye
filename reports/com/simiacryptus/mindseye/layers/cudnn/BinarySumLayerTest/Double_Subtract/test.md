# BinarySumLayer
## Double_Subtract
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.01 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.01 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.292 ], [ 1.568 ] ],
    	[ [ 1.896 ], [ 0.096 ] ]
    ],
    [
    	[ [ 1.52 ], [ 1.252 ] ],
    	[ [ -1.664 ], [ 0.276 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.26979038104038655, negative=1, min=0.096, max=0.096, mean=0.817, count=4.0, positive=3, stdDev=0.9324650127484678, zeros=0},
    {meanExponent=-0.01462241979022344, negative=1, min=0.276, max=0.276, mean=0.34600000000000003, count=4.0, positive=3, stdDev=1.2494110612604643, zeros=0}
    Output: [
    	[ [ -1.812 ], [ 0.31600000000000006 ] ],
    	[ [ 3.5599999999999996 ], [ -0.18000000000000002 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.10885805524115516, negative=2, min=-0.18000000000000002, max=-0.18000000000000002, mean=0.4709999999999999, count=4.0, positive=2, stdDev=1.949476596422742, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.292 ], [ 1.568 ] ],
    	[ [ 1.896 ], [ 0.096 ] ]
    ]
    Value Statistics: {meanExponent=-0.26979038104038655, negative=1, min=0.096, max=0.096, mean=0.817, count=4.0, positive=3, stdDev=0.9324650127484678, zeros=0}
    Implemented F
```
...[skipping 1548 bytes](etc/34.txt)...
```
    ve=0, stdDev=0.4330127018922193, zeros=12}
    Measured Feedback: [ [ -0.9999999999998899, 0.0, 0.0, 0.0 ], [ 0.0, -0.9999999999976694, 0.0, 0.0 ], [ 0.0, 0.0, -0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.0, -0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-2.889125089796616E-13, negative=4, min=-0.9999999999998899, max=-0.9999999999998899, mean=-0.2499999999998337, count=16.0, positive=0, stdDev=0.43301270189193125, zeros=12}
    Feedback Error: [ [ 1.1013412404281553E-13, 0.0, 0.0, 0.0 ], [ 0.0, 2.3305801732931286E-12, 0.0, 0.0 ], [ 0.0, 0.0, 1.1013412404281553E-13, 0.0 ], [ 0.0, 0.0, 0.0, 1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.626692561651478, negative=0, min=1.1013412404281553E-13, max=1.1013412404281553E-13, mean=1.6631140908884845E-13, count=16.0, positive=4, stdDev=5.60437371796192E-13, zeros=12}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.6122e-13 +- 5.3450e-13 [0.0000e+00 - 2.3306e-12] (32#)
    relativeTol: 3.2244e-13 +- 4.5576e-13 [5.5067e-14 - 1.1653e-12] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.6122e-13 +- 5.3450e-13 [0.0000e+00 - 2.3306e-12] (32#), relativeTol=3.2244e-13 +- 4.5576e-13 [5.5067e-14 - 1.1653e-12] (8#)}
```



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
      "id": "9e3eff11-b108-44b2-b262-a7e5bfdc0737",
      "isFrozen": false,
      "name": "BinarySumLayer/9e3eff11-b108-44b2-b262-a7e5bfdc0737",
      "rightFactor": -1.0,
      "leftFactor": 1.0
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
    	[ [ -0.612 ], [ -1.16 ] ],
    	[ [ -0.3 ], [ -1.196 ] ]
    ],
    [
    	[ [ 0.192 ], [ 1.432 ] ],
    	[ [ 0.432 ], [ 1.956 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.804 ], [ -2.5919999999999996 ] ],
    	[ [ -0.732 ], [ -3.152 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ], [ 1.0 ] ],
    	[ [ 1.0 ], [ 1.0 ] ]
    ],
    [
    	[ [ -1.0 ], [ -1.0 ] ],
    	[ [ -1.0 ], [ -1.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.028 ], [ -1.008 ], [ -1.048 ], [ -1.988 ], [ 0.52 ], [ 0.812 ], [ 0.288 ], [ -1.688 ], ... ],
    	[ [ -1.084 ], [ 1.748 ], [ -0.848 ], [ -1.064 ], [ -1.044 ], [ -1.4 ], [ -0.224 ], [ 0.94 ], ... ],
    	[ [ -0.788 ], [ -1.588 ], [ 1.56 ], [ -0.888 ], [ -0.068 ], [ -1.272 ], [ 0.704 ], [ 0.728 ], ... ],
    	[ [ -0.76 ], [ -0.824 ], [ -0.308 ], [ 1.1 ], [ -1.48 ], [ 1.268 ], [ 0.14 ], [ -0.164 ], ... ],
    	[ [ 1.004 ], [ -1.964 ], [ 1.564 ], [ 0.944 ], [ 0.12 ], [ -0.444 ], [ -1.604 ], [ 1.48 ], ... ],
    	[ [ 0.104 ], [ -1.812 ], [ 1.312 ], [ -1.48 ], [ -0.264 ], [ -1.672 ], [ -1.864 ], [ 0.244 ], ... ],
    	[ [ -1.132 ], [ 1.596 ], [ 0.396 ], [ -0.988 ], [ -0.072 ], [ -0.82 ], [ 0.428 ], [ -0.916 ], ... ],
    	[ [ 0.232 ], [ 1.304 ], [ -1.976 ], [ 0.124 ], [ 1.388 ], [ -0.988 ], [ 1.592 ], [ 0.488 ], ... ],
    	...
    ]
    [
    	[ [ 0.464 ], [ -1.904 ], [ 0.104 ], [ 1.424 ], [ 0.592 ], [ -0.844 ], [ -1.404 ], [ -0.088 ], ... ],
    	[ [ -0.068 ], [ -0.368 ], [ 1.312 ], [ -1.176 ], [ 1.216 ], [ -1.336 ], [ 0.032 ], [ -1.064 ], ... ],
    	[ [ -1.656 ], [ -0.428 ], [ 0.652 ], [ -0.776 ], [ 0.728 ], [ 0.816 ], [ 0.14 ], [ 0.892 ], ... ],
    	[ [ -0.396 ], [ 0.964 ], [ 1.952 ], [ -0.424 ], [ 1.372 ], [ 0.324 ], [ 0.488 ], [ -0.812 ], ... ],
    	[ [ 0.364 ], [ 2.0 ], [ -0.852 ], [ 1.112 ], [ 0.876 ], [ -1.316 ], [ -1.808 ], [ -1.776 ], ... ],
    	[ [ 0.684 ], [ 1.276 ], [ -1.512 ], [ -1.768 ], [ -0.668 ], [ 1.208 ], [ 1.476 ], [ 1.528 ], ... ],
    	[ [ -1.86 ], [ -1.988 ], [ 0.22 ], [ 1.628 ], [ 0.832 ], [ -0.64 ], [ -0.712 ], [ 1.136 ], ... ],
    	[ [ -1.772 ], [ -0.028 ], [ -0.872 ], [ 1.4 ], [ 1.032 ], [ -0.116 ], [ 1.252 ], [ -0.096 ], ... ],
    	...
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.03 seconds: 
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
Logging: 
```
    Zero gradient: 0.0
    Constructing line search parameters: GD
    F(0.0) = LineSearchPoint{point=PointSample{avg=2.672687614399998}, derivative=0.0}
    Iteration 1 failed, aborting. Error: 2.672687614399998 Total: 239460481336116.5300; Orientation: 0.0004; Line Search: 0.0035
    
```

Returns: 

```
    2.672687614399998
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.98 ], [ 1.396 ], [ -0.788 ], [ 0.072 ], [ 0.352 ], [ -0.316 ], [ 0.916 ], [ 0.616 ], ... ],
    	[ [ -1.392 ], [ 0.056 ], [ -1.172 ], [ 0.34 ], [ -0.128 ], [ -0.12 ], [ -0.272 ], [ -0.912 ], ... ],
    	[ [ -0.772 ], [ 1.5 ], [ 0.656 ], [ -1.428 ], [ 1.588 ], [ -0.852 ], [ 1.276 ], [ 0.336 ], ... ],
    	[ [ -1.188 ], [ -1.472 ], [ -1.284 ], [ 0.644 ], [ -0.404 ], [ -1.852 ], [ -1.268 ], [ 0.104 ], ... ],
    	[ [ -1.172 ], [ -0.504 ], [ -0.864 ], [ -1.888 ], [ -1.404 ], [ 1.852 ], [ -1.792 ], [ 0.936 ], ... ],
    	[ [ 0.752 ], [ 1.108 ], [ -0.864 ], [ 1.148 ], [ -1.844 ], [ -1.904 ], [ 0.892 ], [ 1.072 ], ... ],
    	[ [ 1.484 ], [ 1.512 ], [ -0.248 ], [ 0.852 ], [ -1.876 ], [ 0.664 ], [ -0.4 ], [ -0.66 ], ... ],
    	[ [ 0.364 ], [ 1.812 ], [ -1.496 ], [ 1.356 ], [ 1.18 ], [ 0.344 ], [ -0.072 ], [ -1.16 ], ... ],
    	...
    ]
    [
    	[ [ -1.884 ], [ 1.6 ], [ 1.032 ], [ 0.328 ], [ 1.292 ], [ -1.984 ], [ 1.696 ], [ 1.072 ], ... ],
    	[ [ -0.8 ], [ 1.372 ], [ 0.356 ], [ -1.388 ], [ -1.976 ], [ -0.376 ], [ -0.148 ], [ -1.688 ], ... ],
    	[ [ 0.612 ], [ 0.576 ], [ 1.78 ], [ 0.036 ], [ 0.888 ], [ 1.608 ], [ 0.956 ], [ -1.9 ], ... ],
    	[ [ -1.336 ], [ 0.684 ], [ -1.928 ], [ 0.188 ], [ -0.232 ], [ -0.04 ], [ 0.2 ], [ 0.388 ], ... ],
    	[ [ -0.668 ], [ -1.48 ], [ -1.968 ], [ -1.952 ], [ -0.152 ], [ -0.92 ], [ 0.0 ], [ 0.308 ], ... ],
    	[ [ -1.752 ], [ 0.244 ], [ -0.056 ], [ 1.72 ], [ -0.796 ], [ 1.428 ], [ -0.312 ], [ 1.4 ], ... ],
    	[ [ -1.472 ], [ -0.448 ], [ 0.296 ], [ -0.3 ], [ -0.936 ], [ 1.984 ], [ -0.524 ], [ 0.38 ], ... ],
    	[ [ 0.288 ], [ -1.416 ], [ 1.42 ], [ -1.304 ], [ 1.892 ], [ -1.252 ], [ 0.628 ], [ -1.628 ], ... ],
    	...
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.01 seconds: 
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
Logging: 
```
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: GD
    th(0)=2.672687614399998;dx=0.0 (ERROR: Starting derivative negative)
    Iteration 1 failed, aborting. Error: 2.672687614399998 Total: 239460508717979.5000; Orientation: 0.0005; Line Search: 0.0085
    
```

Returns: 

```
    2.672687614399998
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.01 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ -0.98 ], [ 1.396 ], [ -0.788 ], [ 0.072 ], [ 0.352 ], [ -0.316 ], [ 0.916 ], [ 0.616 ], ... ],
    	[ [ -1.392 ], [ 0.056 ], [ -1.172 ], [ 0.34 ], [ -0.128 ], [ -0.12 ], [ -0.272 ], [ -0.912 ], ... ],
    	[ [ -0.772 ], [ 1.5 ], [ 0.656 ], [ -1.428 ], [ 1.588 ], [ -0.852 ], [ 1.276 ], [ 0.336 ], ... ],
    	[ [ -1.188 ], [ -1.472 ], [ -1.284 ], [ 0.644 ], [ -0.404 ], [ -1.852 ], [ -1.268 ], [ 0.104 ], ... ],
    	[ [ -1.172 ], [ -0.504 ], [ -0.864 ], [ -1.888 ], [ -1.404 ], [ 1.852 ], [ -1.792 ], [ 0.936 ], ... ],
    	[ [ 0.752 ], [ 1.108 ], [ -0.864 ], [ 1.148 ], [ -1.844 ], [ -1.904 ], [ 0.892 ], [ 1.072 ], ... ],
    	[ [ 1.484 ], [ 1.512 ], [ -0.248 ], [ 0.852 ], [ -1.876 ], [ 0.664 ], [ -0.4 ], [ -0.66 ], ... ],
    	[ [ 0.364 ], [ 1.812 ], [ -1.496 ], [ 1.356 ], [ 1.18 ], [ 0.344 ], [ -0.072 ], [ -1.16 ], ... ],
    	...
    ]
    [
    	[ [ -1.884 ], [ 1.6 ], [ 1.032 ], [ 0.328 ], [ 1.292 ], [ -1.984 ], [ 1.696 ], [ 1.072 ], ... ],
    	[ [ -0.8 ], [ 1.372 ], [ 0.356 ], [ -1.388 ], [ -1.976 ], [ -0.376 ], [ -0.148 ], [ -1.688 ], ... ],
    	[ [ 0.612 ], [ 0.576 ], [ 1.78 ], [ 0.036 ], [ 0.888 ], [ 1.608 ], [ 0.956 ], [ -1.9 ], ... ],
    	[ [ -1.336 ], [ 0.684 ], [ -1.928 ], [ 0.188 ], [ -0.232 ], [ -0.04 ], [ 0.2 ], [ 0.388 ], ... ],
    	[ [ -0.668 ], [ -1.48 ], [ -1.968 ], [ -1.952 ], [ -0.152 ], [ -0.92 ], [ 0.0 ], [ 0.308 ], ... ],
    	[ [ -1.752 ], [ 0.244 ], [ -0.056 ], [ 1.72 ], [ -0.796 ], [ 1.428 ], [ -0.312 ], [ 1.4 ], ... ],
    	[ [ -1.472 ], [ -0.448 ], [ 0.296 ], [ -0.3 ], [ -0.936 ], [ 1.984 ], [ -0.524 ], [ 0.38 ], ... ],
    	[ [ 0.288 ], [ -1.416 ], [ 1.42 ], [ -1.304 ], [ 1.892 ], [ -1.252 ], [ 0.628 ], [ -1.628 ], ... ],
    	...
    ]
```



Code from [LearningTester.java:95](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:98](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.27 seconds: 
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
    	Evaluation performance: 0.013080s +- 0.000867s [0.012056s - 0.014512s]
    	Learning performance: 0.027241s +- 0.001546s [0.025390s - 0.029805s]
    
```

