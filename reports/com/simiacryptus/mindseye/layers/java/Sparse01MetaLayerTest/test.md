# Sparse01MetaLayer
## Sparse01MetaLayerTest
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
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
      "id": "4b34286a-ebe1-4245-9734-0aaabc47d5f0",
      "isFrozen": false,
      "name": "Sparse01MetaLayer/4b34286a-ebe1-4245-9734-0aaabc47d5f0",
      "sparsity": 0.05
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
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
    [[ -1.476, -0.792, -0.58 ]]
    --------------------
    Output: 
    [ 0.0, 0.0, 0.0 ]
    --------------------
    Derivative: 
    [ 0.417558699011869, 0.5932652417027416, 0.6874727193365342 ]
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -0.692, 1.796, 1.784 ]
    Inputs Statistics: {meanExponent=0.1152690922760492, negative=1, min=1.784, max=1.784, mean=0.9626666666666667, count=3.0, positive=2, stdDev=1.17003627674055, zeros=0}
    Output: [ 0.0, 0.0, 0.0 ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=3.0, positive=0, stdDev=0.0, zeros=3}
    Feedback for input 0
    Inputs Values: [ -0.692, 1.796, 1.784 ]
    Value Statistics: {meanExponent=0.1152690922760492, negative=1, min=1.784, max=1.784, mean=0.9626666666666667, count=3.0, positive=2, stdDev=1.17003627674055, zeros=0}
    Implemented Feedback: [ [ 0.6337200563003047, 0.0, 0.0 ], [ 0.0, -1.2213069803359782, 0.0 ], [ 0.0, 0.0, -1.2397615997071472 ] ]
    Implemented Statistics: {meanExponent=-0.005979842721496843, negative=2, min=-1.2397615997071472, max=-1.2397615997071472, mean=-0.2030387248603134, count=9.0, positive=1, stdDev=0.5830168817783498, zeros=6}
    Measured: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=9.0, positive=0, stdDev=0.0, zeros=9}
    Feedback Error: [ [ -0.6337200563003047, 0.0, 0.0 ], [ 0.0, 1.2213069803359782, 0.0 ], [ 0.0, 0.0, 1.2397615997071472 ] ]
    Error Statistics: {meanExponent=-0.005979842721496843, negative=1, min=1.2397615997071472, max=1.2397615997071472, mean=0.2030387248603134, count=9.0, positive=2, stdDev=0.5830168817783498, zeros=6}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=3.4387e-01 +- 5.1273e-01 [0.0000e+00 - 1.2398e+00] (9#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (3#)}
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
...[skipping 1898 bytes](etc/105.txt)...
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



