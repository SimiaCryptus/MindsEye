# AvgMetaLayer
## AvgMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
      "id": "42c3a793-2de1-4418-b1c0-ccafca417fad",
      "isFrozen": false,
      "name": "AvgMetaLayer/42c3a793-2de1-4418-b1c0-ccafca417fad",
      "minBatchCount": 0
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
    [[ 0.304, 0.076, -1.448 ]]
    --------------------
    Output: 
    [ 0.304, 0.076, -1.448 ]
    --------------------
    Derivative: 
    [ 1.0, 1.0, 1.0 ]
```



### Differential Validation
Code from [BatchDerivativeTester.java:76](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchDerivativeTester.java#L76) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.108, -0.9, 0.812 ],
    [ 1.108, -0.9, 0.812 ],
    [ 1.108, -0.9, 0.812 ],
    [ 1.108, -0.9, 0.812 ],
    [ 1.108, -0.9, 0.812 ],
    [ 1.108, -0.9, 0.812 ],
    [ 1.108, -0.9, 0.812 ],
    [ 1.108, -0.9, 0.812 ],
    [ 1.108, -0.9, 0.812 ],
    [ 1.108, -0.9, 0.812 ]
    Inputs Statistics: {meanExponent=-0.030553900309029602, negative=1, min=0.812, max=0.812, mean=0.34, count=3.0, positive=2, stdDev=0.885100370956123, zeros=0},
    {meanExponent=-0.030553900309029602, negative=1, min=0.812, max=0.812, mean=0.34, count=3.0, positive=2, stdDev=0.885100370956123, zeros=0},
    {meanExponent=-0.030553900309029602, negative=1, min=0.812, max=0.812, mean=0.34, count=3.0, positive=2, stdDev=0.885100370956123, zeros=0},
    {meanExponent=-0.030553900309029602, negative=1, min=0.812, max=0.812, mean=0.34, count=3.0, positive=2, stdDev=0.885100370956123, zeros=0},
    {meanExponent=-0.030553900309029602, negative=1, min=0.812, max=0.812, mean=0.34, count=3.0, positive=2, stdDev=0.885100370956123, zeros=0},
    {meanExponent=-0.030553900309029602, negative=1, min
```
...[skipping 954 bytes](etc/230.txt)...
```
    029602, negative=1, min=0.812, max=0.812, mean=0.34, count=3.0, positive=2, stdDev=0.885100370956123, zeros=0}
    Implemented Feedback: [ [ 0.1, 0.1, 0.1 ], [ 0.1, 0.1, 0.1 ], [ 0.1, 0.1, 0.1 ] ]
    Implemented Statistics: {meanExponent=-1.0, negative=0, min=0.1, max=0.1, mean=0.09999999999999999, count=9.0, positive=9, stdDev=1.862645149230957E-9, zeros=0}
    Measured: [ [ 0.10000000000065512, 0.0, 0.0 ], [ 0.0, 0.0999999999995449, 0.0 ], [ 0.0, 0.0, 0.10000000000065512 ] ]
    Measured Statistics: {meanExponent=-0.999999999998762, negative=0, min=0.10000000000065512, max=0.10000000000065512, mean=0.03333333333342835, count=9.0, positive=3, stdDev=0.047140452079237546, zeros=6}
    Feedback Error: [ [ 6.551148512556892E-13, -0.1, -0.1 ], [ -0.1, -4.551081733694673E-13, -0.1 ], [ -0.1, -0.1, 6.551148512556892E-13 ] ]
    Error Statistics: {meanExponent=-4.745472275012728, negative=7, min=6.551148512556892E-13, max=6.551148512556892E-13, mean=-0.06666666666657164, count=9.0, positive=2, stdDev=0.04714045207923756, zeros=0}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=6.6667e-02 +- 4.7140e-02 [4.5511e-13 - 1.0000e-01] (9#), relativeTol=6.6667e-01 +- 4.7140e-01 [2.2755e-12 - 1.0000e+00] (9#)}
    	at com.simiacryptus.mindseye.test.unit.BatchDerivativeTester.lambda$test$9(BatchDerivativeTester.java:107)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.test.unit.BatchDerivativeTester.test(BatchDerivativeTester.java:138)
    	at com.simiacryptu
```
...[skipping 3029 bytes](etc/231.txt)...
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



