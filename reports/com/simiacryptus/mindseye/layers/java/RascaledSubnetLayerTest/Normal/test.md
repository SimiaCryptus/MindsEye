# RescaledSubnetLayer
## Normal
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
      "id": "6bfd1b59-df39-4e03-96ae-e6faf05cfd08",
      "isFrozen": false,
      "name": "RescaledSubnetLayer/6bfd1b59-df39-4e03-96ae-e6faf05cfd08",
      "scale": 2,
      "subnetwork": {
        "class": "com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer",
        "id": "e0364b26-6d2b-4f45-b3fe-84713ddbaadc",
        "isFrozen": false,
        "name": "ConvolutionLayer/e0364b26-6d2b-4f45-b3fe-84713ddbaadc",
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
        "strideY": 1,
        "precision": "Double"
      }
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.90 seconds: 
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
    	[ [ -0.216 ], [ -0.724 ], [ -1.28 ], [ -0.552 ], [ 1.992 ], [ 0.12 ] ],
    	[ [ 1.48 ], [ 1.276 ], [ 0.368 ], [ 0.864 ], [ -1.736 ], [ -0.444 ] ],
    	[ [ 0.444 ], [ 1.592 ], [ -1.072 ], [ -0.664 ], [ -0.912 ], [ -0.9 ] ],
    	[ [ 1.464 ], [ 0.084 ], [ -0.664 ], [ -1.388 ], [ 1.184 ], [ -1.736 ] ],
    	[ [ -0.56 ], [ 1.032 ], [ 0.568 ], [ 0.84 ], [ -0.9 ], [ 1.176 ] ],
    	[ [ 1.54 ], [ -0.016 ], [ 0.764 ], [ -1.612 ], [ 0.58 ], [ -0.168 ] ]
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



### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.99 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (720#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.33 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.932 ], [ -0.612 ], [ 0.508 ], [ -1.388 ], [ -1.244 ], [ -1.764 ] ],
    	[ [ -0.708 ], [ 0.144 ], [ -0.368 ], [ -0.972 ], [ 0.256 ], [ -1.62 ] ],
    	[ [ 1.08 ], [ 1.256 ], [ -1.068 ], [ 1.78 ], [ -0.532 ], [ 0.14 ] ],
    	[ [ -1.424 ], [ -1.172 ], [ -0.892 ], [ 1.716 ], [ -1.908 ], [ 0.252 ] ],
    	[ [ -1.328 ], [ 1.984 ], [ -0.888 ], [ 0.952 ], [ 0.664 ], [ -1.588 ] ],
    	[ [ -1.056 ], [ -1.828 ], [ -0.104 ], [ 0.472 ], [ 0.256 ], [ -1.508 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.090311015173659, negative=21, min=-1.508, max=-1.508, mean=-0.29388888888888887, count=36.0, positive=15, stdDev=1.1513553312359448, zeros=0}
    Output: [
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ],
    	[ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ] ]
    ]
    Outputs Statistics: {mea
```
...[skipping 1573 bytes](etc/378.txt)...
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
    	Evaluation performance: 0.004445s +- 0.000745s [0.003701s - 0.005504s]
    	Learning performance: 0.003844s +- 0.000236s [0.003589s - 0.004139s]
    
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
    	[ [ 0.676 ], [ -1.444 ], [ -1.744 ], [ -1.836 ], [ 1.132 ], [ -0.528 ] ],
    	[ [ -0.184 ], [ -0.9 ], [ 1.648 ], [ 0.416 ], [ 0.924 ], [ -1.62 ] ],
    	[ [ 0.676 ], [ -0.24 ], [ 0.728 ], [ -0.356 ], [ 0.868 ], [ -1.38 ] ],
    	[ [ -0.34 ], [ 1.188 ], [ -1.384 ], [ -1.472 ], [ -0.392 ], [ 1.012 ] ],
    	[ [ 1.616 ], [ -0.372 ], [ 1.86 ], [ 0.488 ], [ -0.504 ], [ -1.1 ] ],
    	[ [ 0.844 ], [ 0.076 ], [ -1.668 ], [ -0.916 ], [ -1.996 ], [ -1.96 ] ]
    ]
```



First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.

Code from [LearningTester.java:300](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L300) executed in 0.01 seconds: 
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
    	[ [ 0.076 ], [ -0.34 ], [ -0.504 ], [ 1.012 ], [ -1.472 ], [ 0.728 ] ],
    	[ [ 1.86 ], [ -1.668 ], [ -1.38 ], [ -0.372 ], [ -1.384 ], [ -0.916 ] ],
    	[ [ 0.676 ], [ -1.62 ], [ 1.132 ], [ -0.9 ], [ -0.24 ], [ -1.996 ] ],
    	[ [ 0.676 ], [ 0.844 ], [ -1.444 ], [ -1.744 ], [ 0.868 ], [ -1.836 ] ],
    	[ [ -1.96 ], [ 1.648 ], [ -0.356 ], [ 1.616 ], [ 1.188 ], [ 0.416 ] ],
    	[ [ 0.924 ], [ 0.488 ], [ -1.1 ], [ -0.392 ], [ -0.184 ], [ -0.528 ] ]
    ]
```



Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.

Code from [LearningTester.java:324](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/LearningTester.java#L324) executed in 0.01 seconds: 
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
    	[ [ 0.076 ], [ -0.34 ], [ -0.504 ], [ 1.012 ], [ -1.472 ], [ 0.728 ] ],
    	[ [ 1.86 ], [ -1.668 ], [ -1.38 ], [ -0.372 ], [ -1.384 ], [ -0.916 ] ],
    	[ [ 0.676 ], [ -1.62 ], [ 1.132 ], [ -0.9 ], [ -0.24 ], [ -1.996 ] ],
    	[ [ 0.676 ], [ 0.844 ], [ -1.444 ], [ -1.744 ], [ 0.868 ], [ -1.836 ] ],
    	[ [ -1.96 ], [ 1.648 ], [ -0.356 ], [ 1.616 ], [ 1.188 ], [ 0.416 ] ],
    	[ [ 0.924 ], [ 0.488 ], [ -1.1 ], [ -0.392 ], [ -0.184 ], [ -0.528 ] ]
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

