# Sparse01MetaLayer
## Sparse01MetaLayerTest
### Json Serialization
Code from [LayerTestBase.java:121](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.Sparse01MetaLayer",
      "id": "e2a3bda5-e7e7-4c05-aeb3-4ede00003665",
      "isFrozen": false,
      "name": "Sparse01MetaLayer/e2a3bda5-e7e7-4c05-aeb3-4ede00003665",
      "sparsity": 0.05
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:159](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L159) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[ -1.968, -1.4, -1.216 ]]
    --------------------
    Output: 
    [ 0.0, 0.0, 0.0 ]
```



### Differential Validation
Code from [LayerTestBase.java:186](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L186) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.968, -1.4, -1.216 ]
    Inputs Statistics: {meanExponent=0.1750289015700923, negative=3, min=-1.216, max=-1.216, mean=-1.5279999999999998, count=3.0, positive=0, stdDev=0.3200666597236693, zeros=0}
    Output: [ 0.0, 0.0, 0.0 ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=3.0, positive=0, stdDev=0.0, zeros=3}
    Feedback for input 0
    Inputs Values: [ -1.968, -1.4, -1.216 ]
    Value Statistics: {meanExponent=0.1750289015700923, negative=3, min=-1.216, max=-1.216, mean=-1.5279999999999998, count=3.0, positive=0, stdDev=0.3200666597236693, zeros=0}
    Implemented Feedback: [ [ 0.34548736659873336, 0.0, 0.0 ], [ 0.0, 0.431547619047619, 0.0 ], [ 0.0, 0.0, 0.46981878206346184 ] ]
    Implemented Statistics: {meanExponent=-0.38486957642268727, negative=0, min=0.46981878206346184, max=0.46981878206346184, mean=0.13853930752331267, count=9.0, positive=3, stdDev=0.19821040963897155, zeros=6}
    Measured: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=9.0, positive=0, stdDev=0.0, zeros=9}
    Feedback Error: [ [ -0.34548736659873336, 0.0, 0.0 ], [ 0.0, -0.431547619047619, 0.0 ], [ 0.0, 0.0, -0.46981878206346184 ] ]
    Error Statistics: {meanExponent=-0.38486957642268727, negative=3, min=-0.46981878206346184, max=-0.46981878206346184, mean=-0.13853930752331267, count=9.0, positive=0, stdDev=0.19821040963897155, zeros=6}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=1.3854e-01 +- 1.9821e-01 [0.0000e+00 - 4.6982e-01] (9#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (3#)}
    	at com.simiacryptus.mindseye.test.DerivativeTester.lambda$test$6(DerivativeTester.java:90)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.test.DerivativeTester.test(DerivativeTester.java:121)
    	at com.simiacryptus.mindseye.layers.LayerTest
```
...[skipping 1863 bytes](etc/89.txt)...
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



