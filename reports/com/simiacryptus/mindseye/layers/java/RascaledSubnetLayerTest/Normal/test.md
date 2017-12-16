# RescaledSubnetLayer
## Normal
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 1.96 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (720#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.35 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.304 ], [ 1.128 ], [ -1.032 ], [ -1.736 ], [ -1.98 ], [ -1.004 ] ],
    	[ [ -1.172 ], [ -1.94 ], [ 1.076 ], [ -0.672 ], [ 1.452 ], [ 0.7 ] ],
    	[ [ 0.208 ], [ -0.12 ], [ -1.88 ], [ -0.068 ], [ 0.12 ], [ 0.628 ] ],
    	[ [ -0.156 ], [ -1.096 ], [ 0.704 ], [ 1.66 ], [ 1.22 ], [ -1.616 ] ],
    	[ [ 1.864 ], [ -1.644 ], [ -0.816 ], [ 0.14 ], [ -1.896 ], [ -0.532 ] ],
    	[ [ 0.28 ], [ -1.432 ], [ 0.552 ], [ -0.604 ], [ 1.436 ], [ 1.444 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.13119526538466675, negative=19, min=1.444, max=1.444, mean=-0.18000000000000005, count=36.0, positive=17, stdDev=1.1645073350277075, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent
```
...[skipping 1555 bytes](etc/320.txt)...
```
    , 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1296.0, positive=0, stdDev=0.0, zeros=1296}
    Feedback Error: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1296.0, positive=0, stdDev=0.0, zeros=1296}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1296#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1296#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
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
      "class": "com.simiacryptus.mindseye.layers.java.RescaledSubnetLayer",
      "id": "97774f27-ee1c-4ffb-8f46-1531da5d5d86",
      "isFrozen": false,
      "name": "RescaledSubnetLayer/97774f27-ee1c-4ffb-8f46-1531da5d5d86",
      "scale": 2,
      "subnetwork": {
        "class": "com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer",
        "id": "a618425e-b17f-4124-a056-008a3a3b817f",
        "isFrozen": false,
        "name": "ConvolutionLayer/a618425e-b17f-4124-a056-008a3a3b817f",
        "filter": [
          [
            [
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0
            ]
          ]
        ],
        "strideX": 1,
        "strideY": 1
      }
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.01 seconds: 
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
    	[ [ -0.944 ], [ 1.432 ], [ 1.744 ], [ 1.06 ], [ -1.048 ], [ -1.236 ] ],
    	[ [ -1.776 ], [ 1.1 ], [ 1.032 ], [ 1.656 ], [ -1.328 ], [ 0.692 ] ],
    	[ [ -1.48 ], [ 0.516 ], [ 0.152 ], [ 0.444 ], [ 0.108 ], [ -1.372 ] ],
    	[ [ 0.032 ], [ 0.352 ], [ -0.728 ], [ -1.236 ], [ -1.86 ], [ 0.796 ] ],
    	[ [ -1.524 ], [ -1.144 ], [ 0.124 ], [ 1.532 ], [ 0.064 ], [ -1.48 ] ],
    	[ [ 0.772 ], [ -1.828 ], [ -0.2 ], [ 1.008 ], [ -1.62 ], [ 0.4 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
```



### Input Learning
In this test, we use a network to learn this target input, given it's pre-evaluated output:

Code from [LearningTester.java:127](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L127) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 0.304 ], [ 1.524 ], [ -0.4 ], [ 0.348 ], [ 0.92 ], [ 0.656 ] ],
    	[ [ 0.876 ], [ -0.412 ], [ -0.46 ], [ -1.532 ], [ 1.344 ], [ 0.212 ] ],
    	[ [ -0.632 ], [ -1.044 ], [ 1.344 ], [ -0.916 ], [ 1.432 ], [ -1.9 ] ],
    	[ [ 1.328 ], [ 1.608 ], [ -1.908 ], [ -0.036 ], [ -1.148 ], [ 0.664 ] ],
    	[ [ -0.7 ], [ -0.82 ], [ -1.356 ], [ 0.12 ], [ 0.04 ], [ 0.516 ] ],
    	[ [ 0.2 ], [ -2.0 ], [ 0.792 ], [ -1.768 ], [ 1.816 ], [ -0.148 ] ]
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:225](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L225) executed in 0.01 seconds: 
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

Code from [LearningTester.java:144](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L144) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.432 ], [ -2.0 ], [ 0.348 ], [ 1.344 ], [ -0.46 ], [ -0.7 ] ],
    	[ [ 0.92 ], [ 0.792 ], [ 0.656 ], [ 1.608 ], [ -0.036 ], [ 1.816 ] ],
    	[ [ 0.304 ], [ 1.328 ], [ 1.344 ], [ 0.876 ], [ -0.4 ], [ 0.516 ] ],
    	[ [ -1.768 ], [ -1.9 ], [ 1.524 ], [ 0.12 ], [ -0.412 ], [ 0.212 ] ],
    	[ [ -0.632 ], [ -0.82 ], [ -1.532 ], [ 0.664 ], [ 0.04 ], [ -1.908 ] ],
    	[ [ 0.2 ], [ -1.148 ], [ -0.148 ], [ -0.916 ], [ -1.356 ], [ -1.044 ] ]
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

Returns: 

```
    0.0
```



This training run resulted in the following regressed input:

Code from [LearningTester.java:154](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L154) executed in 0.00 seconds: 
```java
    return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
```

Returns: 

```
    [
    	[ [ 1.432 ], [ -2.0 ], [ 0.348 ], [ 1.344 ], [ -0.46 ], [ -0.7 ] ],
    	[ [ 0.92 ], [ 0.792 ], [ 0.656 ], [ 1.608 ], [ -0.036 ], [ 1.816 ] ],
    	[ [ 0.304 ], [ 1.328 ], [ 1.344 ], [ 0.876 ], [ -0.4 ], [ 0.516 ] ],
    	[ [ -1.768 ], [ -1.9 ], [ 1.524 ], [ 0.12 ], [ -0.412 ], [ 0.212 ] ],
    	[ [ -0.632 ], [ -0.82 ], [ -1.532 ], [ 0.664 ], [ 0.04 ], [ -1.908 ] ],
    	[ [ 0.2 ], [ -1.148 ], [ -0.148 ], [ -0.916 ], [ -1.356 ], [ -1.044 ] ]
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

Code from [PerformanceTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.07 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[6, 6, 1]
    Performance:
    	Evaluation performance: 0.004914s +- 0.000981s [0.003874s - 0.006495s]
    	Learning performance: 0.003927s +- 0.000304s [0.003631s - 0.004500s]
    
```

