# AvgSubsampleLayer
## AvgSubsampleLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.AvgSubsampleLayer",
      "id": "291de723-cca4-4a76-9bfc-647292277cdc",
      "isFrozen": false,
      "name": "AvgSubsampleLayer/291de723-cca4-4a76-9bfc-647292277cdc",
      "inner": [
        2,
        2,
        1
      ]
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.02 seconds: 
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
    	[ [ -0.988, 0.884, -1.46 ], [ -1.936, 1.956, -1.06 ] ],
    	[ [ -1.76, 0.684, 1.492 ], [ -1.724, -0.676, 2.0 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0, -0.647 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.25, 0.25, 0.25 ], [ 0.25, 0.25, 0.25 ] ],
    	[ [ 0.25, 0.25, 0.25 ], [ 0.25, 0.25, 0.25 ] ]
    ]
```



### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (150#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (130#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.684, -1.832, 0.368 ], [ 0.476, 1.12, 1.68 ] ],
    	[ [ -0.716, 1.156, 0.312 ], [ 1.724, -1.592, 1.04 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.010434271092879107, negative=3, min=1.04, max=1.04, mean=0.45166666666666666, count=12.0, positive=9, stdDev=1.180304433421969, zeros=0}
    Output: [
    	[ [ 0.0, 0.0, 1.355 ] ]
    ]
    Outputs Statistics: {meanExponent=0.13193929521042452, negative=0, min=1.355, max=1.355, mean=0.45166666666666666, count=3.0, positive=1, stdDev=0.6387531256718479, zeros=2}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.684, -1.832, 0.368 ], [ 0.476, 1.12, 1.68 ] ],
    	[ [ -0.716, 1.156, 0.312 ], [ 1.724, -1.592, 1.04 ] ]
    ]
    Value Statistics: {meanExponent=-0.010434271092879107, negative=3, min=1.04, max=1.04, mean=0.45166666666666666, count=12.0, positive=9, stdDev=1.180304433421969, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], [ 0.0, 0.0, 0.25 ], ... ]
```
...[skipping 395 bytes](etc/234.txt)...
```
    9999999941735 ], [ 0.0, 0.0, 0.24999999999941735 ], [ 0.0, 0.0, 0.24999999999941735 ], ... ]
    Measured Statistics: {meanExponent=-0.6020599913289747, negative=0, min=0.24999999999941735, max=0.24999999999941735, mean=0.08333333333313912, count=36.0, positive=12, stdDev=0.11785113019748325, zeros=24}
    Feedback Error: [ [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], [ 0.0, 0.0, -5.826450433232822E-13 ], ... ]
    Error Statistics: {meanExponent=-12.234595943823395, negative=12, min=-5.826450433232822E-13, max=-5.826450433232822E-13, mean=-1.9421501444109404E-13, count=36.0, positive=0, stdDev=2.7466150743908175E-13, zeros=24}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.9422e-13 +- 2.7466e-13 [0.0000e+00 - 5.8265e-13] (36#)
    relativeTol: 1.1653e-12 +- NaN [1.1653e-12 - 1.1653e-12] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.9422e-13 +- 2.7466e-13 [0.0000e+00 - 5.8265e-13] (36#), relativeTol=1.1653e-12 +- NaN [1.1653e-12 - 1.1653e-12] (12#)}
```



### Performance
Now we execute larger-scale runs to benchmark performance:

Code from [PerformanceTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/PerformanceTester.java#L66) executed in 0.00 seconds: 
```java
    test(component, inputPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    
```

Returns: 

```
    java.lang.RuntimeException: java.lang.RuntimeException: java.util.concurrent.ExecutionException: com.google.common.util.concurrent.UncheckedExecutionException: java.lang.IllegalStateException: Duplicate key [[I@1a266de3, [I@39e68736, [I@76760a84, [I@6d3a5d86]
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:61)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:72)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:156)
    	at com.simiacryptus.mindseye.test.unit.PerformanceTester.test(PerformanceTester.java:66)
    	at com.simiacryptus.mindseye.test.unit.PerformanceTester.test(PerformanceTester.java:39)
    	at com.simiacryptus.mindseye.test.unit.StandardLayerTests.lambda$test$4(StandardLayerTests.java:82)
    	at java.util.stream.ForEachOps$ForEachOp$OfRef.accept(
```
...[skipping 6828 bytes](etc/235.txt)...
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



