# LoggingWrapperLayer
## LoggingWrapperLayerTest
Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.02 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.408, 1.88, 1.276 ]
    Inputs Statistics: {meanExponent=0.1762037261516389, negative=1, min=1.276, max=1.276, mean=0.5826666666666667, count=3.0, positive=2, stdDev=1.4290484790781435, zeros=0}
    Output: [ -1.408, 1.88, 1.276 ]
    Outputs Statistics: {meanExponent=0.1762037261516389, negative=1, min=1.276, max=1.276, mean=0.5826666666666667, count=3.0, positive=2, stdDev=1.4290484790781435, zeros=0}
    Input 0 for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ -1.408, 1.88, 1.276 ]
    Output for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ -1.408, 1.88, 1.276 ]
    Feedback Input for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 1.0, 0.0, 0.0 ]
    Feedback Output 0 for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 1.0, 0.0, 0.0 ]
    Input 0 for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ -1.408, 1.88, 1.276 ]
    Output for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ -1.
```
...[skipping 2012 bytes](etc/274.txt)...
```
    =1, min=1.0, max=1.0, mean=0.7913333333333333, count=6.0, positive=5, stdDev=1.0318098446688495, zeros=0}
    Measured Gradient: [ [ -1.40800000000052, 1.88000000000077, 1.2759999999989446 ], [ 0.9999999999998899, 0.9999999999998899, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=0.08810186307579206, negative=1, min=0.9999999999998899, max=0.9999999999998899, mean=0.791333333333144, count=6.0, positive=5, stdDev=1.031809844669076, zeros=0}
    Gradient Error: [ [ -5.200284647344233E-13, 7.700506898800086E-13, -1.0553780072086738E-12 ], [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-12.54137997064271, negative=5, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-1.892930256985892E-13, count=6.0, positive=1, stdDev=5.475137571461476E-13, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.0042e-13 +- 3.1064e-13 [0.0000e+00 - 1.0554e-12] (15#)
    relativeTol: 1.2594e-13 +- 1.1668e-13 [5.5067e-14 - 4.1355e-13] (9#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.0042e-13 +- 3.1064e-13 [0.0000e+00 - 1.0554e-12] (15#), relativeTol=1.2594e-13 +- 1.1668e-13 [5.5067e-14 - 4.1355e-13] (9#)}
```



### Json Serialization
Code from [JsonTest.java:36](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/JsonTest.java#L36) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.LoggingWrapperLayer",
      "id": "aa285c1b-7aa7-485f-9d27-3a01cc271101",
      "isFrozen": false,
      "name": "LoggingWrapperLayer/aa285c1b-7aa7-485f-9d27-3a01cc271101",
      "inner": {
        "class": "com.simiacryptus.mindseye.layers.java.LinearActivationLayer",
        "id": "f3131647-3f91-4019-837f-7a6a2d0a9625",
        "isFrozen": false,
        "name": "LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625",
        "weights": [
          1.0,
          0.0
        ]
      }
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.00 seconds: 
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
    [[ 1.444, -1.248, 1.452 ]]
    --------------------
    Output: 
    [ 1.444, -1.248, 1.452 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ 1.156, -0.868, -0.416 ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.00 seconds: 
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
    Input 0 for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ -0.868, -0.416, 1.156 ]
    Output for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 1.0, 1.0, 1.0 ]
    Feedback Input for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 0.0, 0.0, 0.0 ]
    Feedback Output 0 for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 0.0, 0.0, 0.0 ]
    
```

Returns: 

```
    0.0
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:144](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -0.868, -0.416, 1.156 ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.00 seconds: 
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
    Input 0 for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ -0.868, -0.416, 1.156 ]
    Output for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 1.0, 1.0, 1.0 ]
    Feedback Input for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 0.0, 0.0, 0.0 ]
    Feedback Output 0 for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 0.0, 0.0, 0.0 ]
    
```

Returns: 

```
    0.0
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [ -0.868, -0.416, 1.156 ]
```



Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.00 seconds: 
```java
    return TestUtil.compare(runs);
```

Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

### Model Learning
In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:

Code from [LearningTester.java:176](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L176) executed in 0.00 seconds: 
```java
    return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [1.0, 0.0]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.00 seconds: 
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
    Input 0 for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ -0.416, 1.156, -0.868 ]
    Output for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ -0.416, 1.156, -0.868 ]
    Feedback Input for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 0.0, 0.0, 0.0 ]
    
```

Returns: 

```
    0.0
```



This training run resulted in the following configuration:

Code from [LearningTester.java:189](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L189) executed in 0.00 seconds: 
```java
    return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [-0.416, 1.156, -0.868]
    [1.0, 0.0]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:249](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L249) executed in 0.03 seconds: 
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
    Input 0 for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ -0.416, 1.156, -0.868 ]
    Output for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 1.0, 1.0, 1.0 ]
    Feedback Input for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 0.944, -0.10399999999999994, 1.2453333333333332 ]
    Input 0 for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ -0.416, 1.156, -0.868 ]
    Output for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 1.0, 1.0, 1.0 ]
    Feedback Input for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 0.944, -0.10399999999999994, 1.2453333333333332 ]
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: GD
    Input 0 for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ -0.416, 1.156, -0.868 ]
    Output for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 1.0, 1.0, 1.0 ]
    Feedback Input for layer LinearActivationLayer/f31
```
...[skipping 45800 bytes](etc/275.txt)...
```
    625: 
    	[ 3.700743415417188E-17, 0.0, 0.0 ]
    LBFGS Accumulation History: 1 points
    Input 0 for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ -0.416, 1.156, -0.868 ]
    Output for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ -0.4159999999999999, 1.156, -0.868 ]
    Feedback Input for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 3.700743415417188E-17, 0.0, 0.0 ]
    th(0)=1.0271626370065257E-33;dx=-1.606559059088436E-33
    Input 0 for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ -0.416, 1.156, -0.868 ]
    Output for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ -0.416, 1.156, -0.868 ]
    Feedback Input for layer LinearActivationLayer/f3131647-3f91-4019-837f-7a6a2d0a9625: 
    	[ 0.0, 0.0, 0.0 ]
    New Minimum: 1.0271626370065257E-33 > 0.0
    END: th(1.521947429820792)=0.0; dx=0.0 delta=1.0271626370065257E-33
    Iteration 22 complete. Error: 0.0 Total: 239671698187076.3000; Orientation: 0.0002; Line Search: 0.0008
    
```

Returns: 

```
    0.0
```



Training Converged

Code from [LearningTester.java:95](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L95) executed in 0.01 seconds: 
```java
    return TestUtil.compare(runs);
```

Returns: 

![Result](etc/test.182.png)



Code from [LearningTester.java:98](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L98) executed in 0.00 seconds: 
```java
    return TestUtil.compareTime(runs);
```

Returns: 

![Result](etc/test.183.png)



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.01 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[3]
    Performance:
    	Evaluation performance: 0.000755s +- 0.000126s [0.000617s - 0.000991s]
    	Learning performance: 0.000488s +- 0.000221s [0.000296s - 0.000895s]
    
```

