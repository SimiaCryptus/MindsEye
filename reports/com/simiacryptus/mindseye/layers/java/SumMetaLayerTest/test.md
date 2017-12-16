# SumMetaLayer
## SumMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.SumMetaLayer",
      "id": "6c511366-c9f5-4cf0-964f-2f7ca1ca59b9",
      "isFrozen": false,
      "name": "SumMetaLayer/6c511366-c9f5-4cf0-964f-2f7ca1ca59b9",
      "minBatches": 0
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
    [[ -1.32, 1.86, 0.092 ]]
    --------------------
    Output: 
    [ -1.32, 1.86, 0.092 ]
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
    Inputs: [ 1.868, 1.192, -0.872 ],
    [ 1.868, 1.192, -0.872 ],
    [ 1.868, 1.192, -0.872 ],
    [ 1.868, 1.192, -0.872 ],
    [ 1.868, 1.192, -0.872 ],
    [ 1.868, 1.192, -0.872 ],
    [ 1.868, 1.192, -0.872 ],
    [ 1.868, 1.192, -0.872 ],
    [ 1.868, 1.192, -0.872 ],
    [ 1.868, 1.192, -0.872 ]
    Inputs Statistics: {meanExponent=0.09605653741028647, negative=1, min=-0.872, max=-0.872, mean=0.7293333333333334, count=3.0, positive=2, stdDev=1.1654599473550726, zeros=0},
    {meanExponent=0.09605653741028647, negative=1, min=-0.872, max=-0.872, mean=0.7293333333333334, count=3.0, positive=2, stdDev=1.1654599473550726, zeros=0},
    {meanExponent=0.09605653741028647, negative=1, min=-0.872, max=-0.872, mean=0.7293333333333334, count=3.0, positive=2, stdDev=1.1654599473550726, zeros=0},
    {meanExponent=0.09605653741028647, negative=1, min=-0.872, max=-0.872, mean=0.7293333333333334, count=3.0, positive=2, stdDev=1.1654599473550726, zeros=0},
    {meanExponent=0.09605653741028647, negative=1, min=-0.872, max=-0.872, mean=0.7293333333333334, count=3.0, positi
```
...[skipping 1124 bytes](etc/415.txt)...
```
    ent=0.09605653741028647, negative=1, min=-0.872, max=-0.872, mean=0.7293333333333334, count=3.0, positive=2, stdDev=1.1654599473550726, zeros=0}
    Implemented Feedback: [ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=9.0, positive=9, stdDev=0.0, zeros=0}
    Measured: [ [ 0.9999999999976694, 0.0, 0.0 ], [ 0.0, 0.9999999999976694, 0.0 ], [ 0.0, 0.0, 1.000000000015433 ] ]
    Measured Statistics: {meanExponent=1.5593818018964093E-12, negative=0, min=1.000000000015433, max=1.000000000015433, mean=0.3333333333345302, count=9.0, positive=3, stdDev=0.4714045207927243, zeros=6}
    Feedback Error: [ [ -2.3305801732931286E-12, -1.0, -1.0 ], [ -1.0, -2.3305801732931286E-12, -1.0 ], [ -1.0, -1.0, 1.5432988220709376E-11 ] ]
    Error Statistics: {meanExponent=-3.7862913200312422, negative=8, min=1.5432988220709376E-11, max=1.5432988220709376E-11, mean=-0.6666666666654698, count=9.0, positive=1, stdDev=0.47140452079272427, zeros=0}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=6.6667e-01 +- 4.7140e-01 [2.3306e-12 - 1.0000e+00] (9#), relativeTol=6.6667e-01 +- 4.7140e-01 [1.1653e-12 - 1.0000e+00] (9#)}
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
...[skipping 3029 bytes](etc/416.txt)...
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



