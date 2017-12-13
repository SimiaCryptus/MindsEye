# ImgBandBiasLayer
## Float
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer",
      "id": "b245221b-9b1d-4413-a6c0-2d546fd613dd",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/b245221b-9b1d-4413-a6c0-2d546fd613dd",
      "bias": [
        0.0,
        0.0
      ]
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
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
    	[ [ 1.24, -0.852 ], [ -0.5, 1.396 ], [ -1.304, 0.66 ] ],
    	[ [ 1.164, -0.308 ], [ 0.352, -0.32 ], [ 1.92, -1.528 ] ],
    	[ [ -1.356, -0.76 ], [ 0.204, 1.82 ], [ -0.564, -1.444 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ],
    	[ [ 1.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 1.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.01 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (360#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (180#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.88, -0.72 ], [ 0.016, -1.708 ], [ 0.476, -1.292 ] ],
    	[ [ 1.164, 0.052 ], [ -0.228, 1.852 ], [ -1.932, 0.384 ] ],
    	[ [ 1.208, 1.516 ], [ 0.14, -1.28 ], [ -1.04, 0.468 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.23120991699413226, negative=8, min=0.468, max=0.468, mean=-0.15577777777777777, count=18.0, positive=10, stdDev=1.1514500787922233, zeros=0}
    Output: [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=18.0, positive=0, stdDev=0.0, zeros=18}
    Feedback for input 0
    Inputs Values: [
    	[ [ -1.88, -0.72 ], [ 0.016, -1.708 ], [ 0.476, -1.292 ] ],
    	[ [ 1.164, 0.052 ], [ -0.228, 1.852 ], [ -1.932, 0.384 ] ],
    	[ [ 1.208, 1.516 ], [ 0.14, -1.28 ], [ -1.04, 0.468 ] ]
    ]
    Value Statistics: {meanExponent=-0.23120991699413226, negative=8, min=0.468, max=0.468, mean=-0.15577777777777777, count=18.0, positive=10, stdDev=1.15145
```
...[skipping 712 bytes](etc/32.txt)...
```
     ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], ... ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=324.0, positive=0, stdDev=0.0, zeros=324}
    Feedback Error: [ [ -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, ... ], ... ]
    Error Statistics: {meanExponent=0.0, negative=18, min=-1.0, max=-1.0, mean=-0.05555555555555555, count=324.0, positive=0, stdDev=0.2290614236454256, zeros=306}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=5.5556e-02 +- 2.2906e-01 [0.0000e+00 - 1.0000e+00] (324#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (18#)}
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
    	at com.simiacryptus.mi
```
...[skipping 1901 bytes](etc/33.txt)...
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



