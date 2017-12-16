# EntropyLayer
## EntropyLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.EntropyLayer",
      "id": "84cabb54-6f3a-4a01-b2dd-4f13164b2a21",
      "isFrozen": true,
      "name": "EntropyLayer/84cabb54-6f3a-4a01-b2dd-4f13164b2a21"
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
    [[
    	[ [ -1.092 ], [ 1.028 ], [ 1.808 ] ],
    	[ [ -1.796 ], [ -1.276 ], [ 1.48 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.09610787803640301 ], [ -0.028388391709896647 ], [ -1.0707360416417326 ] ],
    	[ [ 1.0516692979044941 ], [ 0.31099971596123527 ], [ -0.580222289908515 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ -1.0880108773227133 ], [ -1.0276151670329734 ], [ -1.5922212619699847 ] ],
    	[ [ -1.5855619698800079 ], [ -1.2437301849225981 ], [ -1.3920420877760238 ] ]
    ]
```



### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.804 ], [ -1.272 ], [ 0.632 ] ],
    	[ [ 0.04 ], [ 0.732 ], [ -0.024 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.4986294938502227, negative=3, min=-0.024, max=-0.024, mean=-0.2826666666666667, count=6.0, positive=3, stdDev=0.9425516195708091, zeros=0}
    Output: [
    	[ [ 1.0643715846393393 ], [ 0.30603107137560753 ], [ 0.2900032392158967 ] ],
    	[ [ 0.128755032994728 ], [ 0.22836552799524426 ], [ -0.08951283476722059 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.6007430473300595, negative=1, min=-0.08951283476722059, max=-0.08951283476722059, mean=0.32133560357559926, count=6.0, positive=5, stdDev=0.3577349030881641, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.804 ], [ -1.272 ], [ 0.632 ] ],
    	[ [ 0.04 ], [ 0.732 ], [ -0.024 ] ]
    ]
    Value Statistics: {meanExponent=-0.4986294938502227, negative=3, min=-0.024, max=-0.024, mean=-0.2826666666666667, count=6.0, positive=3, stdDev=0.9425516195708091, zeros=0}
    Implemented Feedback: [ [ -1.590006421640432, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 2.21887582
```
...[skipping 469 bytes](etc/265.txt)...
```
    33733, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 2.217626865234912, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.2405511557117554, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.6880935378797415, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.5412132249160706, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 2.7317876815292905 ] ]
    Measured Statistics: {meanExponent=0.1080595469165826, negative=4, min=2.7317876815292905, max=2.7317876815292905, mean=0.024710497869812817, count=36.0, positive=2, stdDev=0.6910526362971127, zeros=30}
    Feedback Error: [ [ 2.771669705858848E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.0012489596332887487, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 3.930920617500888E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -6.830290056702992E-5, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -7.91097513501704E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.002086232895099016 ] ]
    Error Statistics: {meanExponent=-3.802363982317851, negative=3, min=0.002086232895099016, max=0.002086232895099016, mean=2.1024625364629568E-5, count=36.0, positive=3, stdDev=4.051608621127629E-4, zeros=30}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=9.8601e-05 +- 3.9354e-04 [0.0000e+00 - 2.0862e-03] (36#), relativeTol=1.3513e-04 +- 1.4361e-04 [8.7160e-06 - 3.8199e-04] (6#)}
    	at com.simiacryptus.mindseye.test.unit.SingleDerivativeTester.lambda$test$7(SingleDerivativeTester.java:107)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.test.unit.SingleDerivativeTester.test(SingleDerivativeTester.java:138)
    	at com.simiac
```
...[skipping 3056 bytes](etc/266.txt)...
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



