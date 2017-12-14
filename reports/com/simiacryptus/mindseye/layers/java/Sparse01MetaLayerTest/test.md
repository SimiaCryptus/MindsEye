# Sparse01MetaLayer
## Sparse01MetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.Sparse01MetaLayer",
      "id": "d8dc1fac-7c12-412e-99c0-bc97ca34b863",
      "isFrozen": false,
      "name": "Sparse01MetaLayer/d8dc1fac-7c12-412e-99c0-bc97ca34b863",
      "sparsity": 0.05
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
    [[ 1.616, 0.632, -1.116 ]]
    --------------------
    Output: 
    [ 0.0, 0.7741167746684372, 0.0 ]
    --------------------
    Derivative: 
    [ -1.5731483862671978, 2.5024078150798017, 0.4937631698409794 ]
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.508, -0.336, 1.272 ]
    Inputs Statistics: {meanExponent=-0.06359075658800185, negative=1, min=1.272, max=1.272, mean=0.8146666666666667, count=3.0, positive=2, stdDev=0.8193287225256757, zeros=0}
    Output: [ 0.0, 0.0, 0.0 ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=3.0, positive=0, stdDev=0.0, zeros=3}
    Feedback for input 0
    Inputs Values: [ 1.508, -0.336, 1.272 ]
    Value Statistics: {meanExponent=-0.06359075658800185, negative=1, min=1.272, max=1.272, mean=0.8146666666666667, count=3.0, positive=2, stdDev=0.8193287225256757, zeros=0}
    Implemented Feedback: [ [ -1.9032352388312201, 0.0, 0.0 ], [ 0.0, 0.859887368120901, 0.0 ], [ 0.0, 0.0, -3.531955234924158 ] ]
    Implemented Statistics: {meanExponent=0.25398307662220226, negative=2, min=-3.531955234924158, max=-3.531955234924158, mean=-0.5083670117371641, count=9.0, positive=1, stdDev=1.269754362688041, zeros=6}
    Measured: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=9.0, positive=0, stdDev=0.0, zeros=9}
    Feedback Error: [ [ 1.9032352388312201, 0.0, 0.0 ], [ 0.0, -0.859887368120901, 0.0 ], [ 0.0, 0.0, 3.531955234924158 ] ]
    Error Statistics: {meanExponent=0.25398307662220226, negative=1, min=3.531955234924158, max=3.531955234924158, mean=0.5083670117371641, count=9.0, positive=2, stdDev=1.269754362688041, zeros=6}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=6.9945e-01 +- 1.1754e+00 [0.0000e+00 - 3.5320e+00] (9#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (3#)}
    	at com.simiacryptus.mindseye.test.SingleDerivativeTester.lambda$test$6(SingleDerivativeTester.java:90)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.test.SingleDerivativeTester.test(SingleDerivativeTester.java:121)
    	at com.simiacryptus.minds
```
...[skipping 1899 bytes](etc/146.txt)...
```
    unner.java:268)
    	at org.junit.runners.ParentRunner.run(ParentRunner.java:363)
    	at org.junit.runners.Suite.runChild(Suite.java:128)
    	at org.junit.runners.Suite.runChild(Suite.java:27)
    	at org.junit.runners.ParentRunner$3.run(ParentRunner.java:290)
    	at org.junit.runners.ParentRunner$1.schedule(ParentRunner.java:71)
    	at org.junit.runners.ParentRunner.runChildren(ParentRunner.java:288)
    	at org.junit.runners.ParentRunner.access$000(ParentRunner.java:58)
    	at org.junit.runners.ParentRunner$2.evaluate(ParentRunner.java:268)
    	at org.junit.runners.ParentRunner.run(ParentRunner.java:363)
    	at org.junit.runner.JUnitCore.run(JUnitCore.java:137)
    	at com.intellij.junit4.JUnit4IdeaTestRunner.startRunnerWithArgs(JUnit4IdeaTestRunner.java:68)
    	at com.intellij.rt.execution.junit.IdeaTestRunner$Repeater.startRunnerWithArgs(IdeaTestRunner.java:47)
    	at com.intellij.rt.execution.junit.JUnitStarter.prepareStreamsAndStart(JUnitStarter.java:242)
    	at com.intellij.rt.execution.junit.JUnitStarter.main(JUnitStarter.java:70)
    
```



