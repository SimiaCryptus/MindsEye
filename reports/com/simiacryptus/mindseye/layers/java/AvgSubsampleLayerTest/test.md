# AvgSubsampleLayer
## AvgSubsampleLayerTest
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.AvgSubsampleLayer",
      "id": "2f40a2f1-e772-47a8-961a-a5c024134a67",
      "isFrozen": false,
      "name": "AvgSubsampleLayer/2f40a2f1-e772-47a8-961a-a5c024134a67",
      "inner": [
        2,
        2,
        1
      ]
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
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
    	[ [ -1.384, -1.972, -0.928 ], [ -1.484, -1.492, 0.816 ] ],
    	[ [ -1.852, -0.8, 0.252 ], [ 0.648, 1.688, -1.528 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0, -2.0090000000000003 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.25, 0.25, 0.25 ], [ 0.25, 0.25, 0.25 ] ],
    	[ [ 0.25, 0.25, 0.25 ], [ 0.25, 0.25, 0.25 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.828, -0.22, 1.656 ], [ 1.624, -1.18, 0.736 ] ],
    	[ [ -1.896, 0.396, 1.012 ], [ -1.504, -0.484, 1.096 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.020381430441884393, negative=6, min=1.096, max=1.096, mean=-0.049333333333333396, count=12.0, positive=6, stdDev=1.2631920765355609, zeros=0}
    Output: [
    	[ [ 0.0, 0.0, -0.14800000000000002 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.8297382846050425, negative=1, min=-0.14800000000000002, max=-0.14800000000000002, mean=-0.04933333333333334, count=3.0, positive=0, stdDev=0.0697678690770727, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.828, -0.22, 1.656 ], [ 1.624, -1.18, 0.736 ] ],
    	[ [ -1.896, 0.396, 1.012 ], [ -1.504, -0.484, 1.096 ] ]
    ]
    Value Statistics: {meanExponent=-0.020381430441884393, negative=6, min=1.096, max=1.096, mean=-0.049333333333333396, count=12.0, positive=6, stdDev=1.2631920765355609, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0
```
...[skipping 462 bytes](etc/96.txt)...
```
    9997247 ], [ 0.0, 0.0, 0.24999999999997247 ], [ 0.0, 0.0, 0.24999999999997247 ], ... ]
    Measured Statistics: {meanExponent=-0.6020599913281711, negative=0, min=0.24999999999941735, max=0.24999999999941735, mean=0.08333333333329332, count=36.0, positive=12, stdDev=0.11785113019770133, zeros=24}
    Feedback Error: [ [ 0.0, 0.0, 5.275779813018744E-13 ], [ 0.0, 0.0, 5.275779813018744E-13 ], [ 0.0, 0.0, 5.275779813018744E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -2.7533531010703882E-14 ], [ 0.0, 0.0, -2.7533531010703882E-14 ], [ 0.0, 0.0, -2.7533531010703882E-14 ], [ 0.0, 0.0, -2.7533531010703882E-14 ], ... ]
    Error Statistics: {meanExponent=-12.687222674269144, negative=9, min=-5.826450433232822E-13, max=-5.826450433232822E-13, mean=-4.001737213204453E-14, count=36.0, positive=3, stdDev=2.6234911803022537E-13, zeros=24}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.2795e-13 +- 2.3250e-13 [0.0000e+00 - 5.8265e-13] (36#)
    relativeTol: 7.6768e-13 +- 5.0577e-13 [5.5067e-14 - 1.1653e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.2795e-13 +- 2.3250e-13 [0.0000e+00 - 5.8265e-13] (36#), relativeTol=7.6768e-13 +- 5.0577e-13 [5.5067e-14 - 1.1653e-12] (12#)}
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.00 seconds: 
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
    java.lang.RuntimeException: java.lang.RuntimeException: java.util.concurrent.ExecutionException: com.google.common.util.concurrent.UncheckedExecutionException: java.lang.IllegalStateException: Duplicate key [[I@1cdf3169, [I@420d1775, [I@37ca89dc, [I@2653bb94]
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:61)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:72)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:156)
    	at com.simiacryptus.mindseye.test.StandardLayerTests.test(StandardLayerTests.java:119)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:40)
    	at sun.reflect.GeneratedMethodAccessor16.invoke(Unknown Source)
    	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
    	at java.lang
```
...[skipping 5814 bytes](etc/97.txt)...
```
    va:116)
    	at java.util.Spliterators$IteratorSpliterator.forEachRemaining(Spliterators.java:1801)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.collect(ReferencePipeline.java:499)
    	at com.simiacryptus.mindseye.layers.java.AvgSubsampleLayer$1.load(AvgSubsampleLayer.java:52)
    	at com.simiacryptus.mindseye.layers.java.AvgSubsampleLayer$1.load(AvgSubsampleLayer.java:48)
    	at com.google.common.cache.LocalCache$LoadingValueReference.loadFuture(LocalCache.java:3599)
    	at com.google.common.cache.LocalCache$Segment.loadSync(LocalCache.java:2379)
    	at com.google.common.cache.LocalCache$Segment.lockedGetOrLoad(LocalCache.java:2342)
    	at com.google.common.cache.LocalCache$Segment.get(LocalCache.java:2257)
    	... 20 more
    
```



