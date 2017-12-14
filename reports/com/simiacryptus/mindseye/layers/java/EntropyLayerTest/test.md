# EntropyLayer
## EntropyLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.EntropyLayer",
      "id": "2f9ad0b2-00c6-4320-86bc-a3250fcabd6f",
      "isFrozen": true,
      "name": "EntropyLayer/2f9ad0b2-00c6-4320-86bc-a3250fcabd6f"
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
    	[ [ 0.08 ], [ -1.492 ], [ 0.632 ] ],
    	[ [ 0.948 ], [ 0.304 ], [ 0.504 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.20205829154466046 ], [ 0.5969753126581011 ], [ 0.2900032392158967 ] ],
    	[ [ 0.050623936337305296 ], [ 0.3619811835830783 ], [ 0.34533022149902726 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.5257286443082556 ], [ -1.400117501781569 ], [ -0.5411341151647204 ] ],
    	[ [ -0.9465992232728847 ], [ 0.1907275775759154 ], [ -0.3148209890892316 ] ]
    ]
```



### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.908 ], [ 0.752 ], [ 0.792 ] ],
    	[ [ -1.532 ], [ 1.552 ], [ -0.024 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.251768234227903, negative=3, min=-0.024, max=-0.024, mean=0.10533333333333335, count=6.0, positive=3, stdDev=1.057273642703513, zeros=0}
    Output: [
    	[ [ -0.08763189754580611 ], [ 0.21433425418428753 ], [ 0.18468955863682723 ] ],
    	[ [ 0.6535114772597882 ], [ -0.682172942573114 ], [ -0.08951283476722059 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.6431284727846306, negative=3, min=-0.08951283476722059, max=-0.08951283476722059, mean=0.0322029358657937, count=6.0, positive=3, stdDev=0.4045483701323853, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.908 ], [ 0.752 ], [ 0.792 ] ],
    	[ [ -1.532 ], [ 1.552 ], [ -0.024 ] ]
    ]
    Value Statistics: {meanExponent=-0.251768234227903, negative=3, min=-0.024, max=-0.024, mean=0.10533333333333335, count=6.0, positive=3, stdDev=1.057273642703513, zeros=0}
    Implemented Feedback: [ [ -0.9034890996191562, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.426574
```
...[skipping 471 bytes](etc/106.txt)...
```
    022, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.4265414335334903, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.7150475313827398, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.4395766375630004, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.7668692414886102, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 2.7317876815292905 ] ]
    Measured Statistics: {meanExponent=0.07398646650652106, negative=5, min=2.7317876815292905, max=2.7317876815292905, mean=-0.06999114427657366, count=36.0, positive=1, stdDev=0.6080315319620879, zeros=30}
    Feedback Error: [ [ 5.506810105404547E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 3.2637784909361756E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -6.6486415037037E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -3.2215801973212166E-5, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -6.312865632140952E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.002086232895099016 ] ]
    Error Statistics: {meanExponent=-4.049164670902041, negative=3, min=0.002086232895099016, max=0.002086232895099016, mean=5.589188632585457E-5, count=36.0, positive=3, stdDev=3.437315019437681E-4, zeros=30}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=6.4882e-05 +- 3.4215e-04 [0.0000e+00 - 2.0862e-03] (36#), relativeTol=8.7125e-05 +- 1.3255e-04 [1.1189e-05 - 3.8199e-04] (6#)}
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
    	at com.simiacryptus.mind
```
...[skipping 2006 bytes](etc/107.txt)...
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



