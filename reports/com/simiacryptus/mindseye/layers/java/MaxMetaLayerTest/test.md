# MaxMetaLayer
## MaxMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MaxMetaLayer",
      "id": "05c72a16-11a3-4203-bd58-95d71af1f840",
      "isFrozen": false,
      "name": "MaxMetaLayer/05c72a16-11a3-4203-bd58-95d71af1f840"
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
    [[ -0.716, -1.084, 0.892 ]]
    --------------------
    Output: 
    [ -0.716, -1.084, 0.892 ]
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
    Inputs: [ 1.696, 1.188, 0.82 ],
    [ 1.696, 1.188, 0.82 ],
    [ 1.696, 1.188, 0.82 ],
    [ 1.696, 1.188, 0.82 ],
    [ 1.696, 1.188, 0.82 ],
    [ 1.696, 1.188, 0.82 ],
    [ 1.696, 1.188, 0.82 ],
    [ 1.696, 1.188, 0.82 ],
    [ 1.696, 1.188, 0.82 ],
    [ 1.696, 1.188, 0.82 ]
    Inputs Statistics: {meanExponent=0.07268538031652878, negative=0, min=0.82, max=0.82, mean=1.2346666666666666, count=3.0, positive=3, stdDev=0.3591446628990729, zeros=0},
    {meanExponent=0.07268538031652878, negative=0, min=0.82, max=0.82, mean=1.2346666666666666, count=3.0, positive=3, stdDev=0.3591446628990729, zeros=0},
    {meanExponent=0.07268538031652878, negative=0, min=0.82, max=0.82, mean=1.2346666666666666, count=3.0, positive=3, stdDev=0.3591446628990729, zeros=0},
    {meanExponent=0.07268538031652878, negative=0, min=0.82, max=0.82, mean=1.2346666666666666, count=3.0, positive=3, stdDev=0.3591446628990729, zeros=0},
    {meanExponent=0.07268538031652878, negative=0, min=0.82, max=0.82, mean=1.2346666666666666, count=3.0, positive=3, stdDev=0.3591446628990729, zeros=0
```
...[skipping 1057 bytes](etc/344.txt)...
```
    nt=0.07268538031652878, negative=0, min=0.82, max=0.82, mean=1.2346666666666666, count=3.0, positive=3, stdDev=0.3591446628990729, zeros=0}
    Implemented Feedback: [ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ]
    Implemented Statistics: {meanExponent=0.0, negative=0, min=1.0, max=1.0, mean=1.0, count=9.0, positive=9, stdDev=0.0, zeros=0}
    Measured: [ [ 0.9999999999998899, 0.0, 0.0 ], [ 0.0, 0.9999999999998899, 0.0 ], [ 0.0, 0.0, 0.9999999999998899 ] ]
    Measured Statistics: {meanExponent=-4.7830642341045674E-14, negative=0, min=0.9999999999998899, max=0.9999999999998899, mean=0.3333333333332966, count=9.0, positive=3, stdDev=0.4714045207909798, zeros=6}
    Feedback Error: [ [ -1.1013412404281553E-13, -1.0, -1.0 ], [ -1.0, -1.1013412404281553E-13, -1.0 ], [ -1.0, -1.0, -1.1013412404281553E-13 ] ]
    Error Statistics: {meanExponent=-4.319359366012275, negative=9, min=-1.1013412404281553E-13, max=-1.1013412404281553E-13, mean=-0.6666666666667034, count=9.0, positive=0, stdDev=0.4714045207909797, zeros=0}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=6.6667e-01 +- 4.7140e-01 [1.1013e-13 - 1.0000e+00] (9#), relativeTol=6.6667e-01 +- 4.7140e-01 [5.5067e-14 - 1.0000e+00] (9#)}
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
...[skipping 3029 bytes](etc/345.txt)...
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



