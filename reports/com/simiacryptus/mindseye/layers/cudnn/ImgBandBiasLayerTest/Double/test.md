# ImgBandBiasLayer
## Double
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer",
      "id": "1eef67d5-759a-4e98-8287-c5342070d033",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/1eef67d5-759a-4e98-8287-c5342070d033",
      "bias": [
        0.0,
        0.0
      ]
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
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
    	[ [ 1.496, -1.148 ], [ 1.14, 0.168 ], [ 0.9, -1.432 ] ],
    	[ [ 0.696, -1.256 ], [ 0.108, 1.52 ], [ 1.864, 1.3 ] ],
    	[ [ -1.788, -0.976 ], [ 0.536, 0.768 ], [ 0.716, 1.428 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.496, -1.148 ], [ 1.14, 0.168 ], [ 0.9, -1.432 ] ],
    	[ [ 0.696, -1.256 ], [ 0.108, 1.52 ], [ 1.864, 1.3 ] ],
    	[ [ -1.788, -0.976 ], [ 0.536, 0.768 ], [ 0.716, 1.428 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.04 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.404, -1.328 ], [ -0.108, 0.272 ], [ 1.828, -1.6 ] ],
    	[ [ -1.676, -0.264 ], [ 0.912, 0.112 ], [ -1.092, -0.868 ] ],
    	[ [ 1.892, -0.316 ], [ 1.236, 0.02 ], [ 0.424, -1.932 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2567482722667473, negative=9, min=-1.932, max=-1.932, mean=-0.11577777777777779, count=18.0, positive=9, stdDev=1.117079901427306, zeros=0}
    Output: [
    	[ [ 0.404, -1.328 ], [ -0.108, 0.272 ], [ 1.828, -1.6 ] ],
    	[ [ -1.676, -0.264 ], [ 0.912, 0.112 ], [ -1.092, -0.868 ] ],
    	[ [ 1.892, -0.316 ], [ 1.236, 0.02 ], [ 0.424, -1.932 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.2567482722667473, negative=9, min=-1.932, max=-1.932, mean=-0.11577777777777779, count=18.0, positive=9, stdDev=1.117079901427306, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.404, -1.328 ], [ -0.108, 0.272 ], [ 1.828, -1.6 ] ],
    	[ [ -1.676, -0.264 ], [ 0.912, 0.112 ], [ -1.092, -0.868 ] ],
    	[ [ 1.892, -0.316 ], [ 1.236, 0.02 ], [ 0.424, -1.932 ] ]
    ]
    Value Statistics: {meanExponent=-0.2567482722667473, ne
```
...[skipping 2669 bytes](etc/67.txt)...
```
    , 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, 0.9999999999998899, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=-3.862131503546135E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.4999999999999555, count=36.0, positive=18, stdDev=0.4999999999999555, zeros=18}
    Gradient Error: [ [ -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, 2.864375403532904E-14, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, -1.1013412404281553E-13, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-13.093294867187552, negative=16, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-4.446443213623752E-14, count=36.0, positive=2, stdDev=5.588794961510799E-14, zeros=18}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.5294e-15 +- 3.0484e-14 [0.0000e+00 - 1.1013e-13] (360#)
    relativeTol: 4.7647e-14 +- 1.6734e-14 [2.9976e-15 - 5.5067e-14] (36#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=9.5294e-15 +- 3.0484e-14 [0.0000e+00 - 1.1013e-13] (360#), relativeTol=4.7647e-14 +- 1.6734e-14 [2.9976e-15 - 5.5067e-14] (36#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    
```

Returns: 

```
    java.lang.RuntimeException: java.lang.RuntimeException: java.util.concurrent.ExecutionException: java.lang.AssertionError: 3 != 2
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:61)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:72)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:156)
    	at com.simiacryptus.mindseye.test.StandardLayerTests.test(StandardLayerTests.java:119)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:40)
    	at sun.reflect.GeneratedMethodAccessor16.invoke(Unknown Source)
    	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
    	at java.lang.reflect.Method.invoke(Method.java:498)
    	at org.junit.runners.model.FrameworkMethod$1.runReflectiveCall(FrameworkMethod.java:50)
```
...[skipping 2717 bytes](etc/68.txt)...
```
    downNotebookOutput.java:138)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	... 35 more
    Caused by: java.util.concurrent.ExecutionException: java.lang.AssertionError: 3 != 2
    	at java.util.concurrent.FutureTask.report(FutureTask.java:122)
    	at java.util.concurrent.FutureTask.get(FutureTask.java:192)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$call$1(GpuController.java:94)
    	... 44 more
    Caused by: java.lang.AssertionError: 3 != 2
    	at com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer.eval(ImgBandBiasLayer.java:91)
    	at com.simiacryptus.mindseye.test.SimpleEval.lambda$call$4(SimpleEval.java:95)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$null$0(GpuController.java:94)
    	at java.util.concurrent.FutureTask.run(FutureTask.java:266)
    	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
    	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
    	at java.lang.Thread.run(Thread.java:748)
    
```



