# ConvolutionLayer
## Float
### Json Serialization
Code from [JsonTest.java:36](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/JsonTest.java#L36) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer",
      "id": "01bc1745-5dd3-449d-aba7-e9b9ae0fcab1",
      "isFrozen": false,
      "name": "ConvolutionLayer/01bc1745-5dd3-449d-aba7-e9b9ae0fcab1",
      "filter": [
        [
          [
            -0.752
          ]
        ],
        [
          [
            1.2
          ]
        ],
        [
          [
            -1.328
          ]
        ],
        [
          [
            -1.424
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1,
      "precision": "Float"
    }
```



### Example Input/Output Pair
Code from [ReferenceIO.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/ReferenceIO.java#L68) executed in 0.00 seconds: 
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
    	[ [ 1.248, -0.108 ], [ 1.336, -0.556 ], [ 0.356, 0.676 ], [ -0.18, -0.748 ], [ 0.536, 1.096 ] ],
    	[ [ -0.284, 1.58 ], [ 1.368, -0.048 ], [ 1.1, 0.604 ], [ -0.684, 0.34 ], [ 1.72, 0.952 ] ],
    	[ [ -1.972, 0.516 ], [ 1.976, 0.808 ], [ -0.44, -0.464 ], [ -0.596, 1.9 ], [ -1.992, 1.632 ] ],
    	[ [ 1.672, -0.648 ], [ 0.64, 1.204 ], [ 1.852, -0.344 ], [ 1.256, 1.648 ], [ -1.012, -0.392 ] ],
    	[ [ 1.636, -1.024 ], [ 1.432, -1.008 ], [ -1.096, -1.404 ], [ -0.416, -1.704 ], [ -0.688, -0.94 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.02 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1000#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -1.672, -1.38 ], [ -1.048, -0.16 ], [ 1.212, 0.868 ], [ -1.552, 0.124 ], [ 1.296, 1.188 ] ],
    	[ [ -0.592, 1.02 ], [ 1.656, -0.308 ], [ 1.844, -1.764 ], [ -1.472, 1.912 ], [ 0.304, 0.168 ] ],
    	[ [ 1.992, 1.84 ], [ -0.88, 0.528 ], [ -0.816, -1.348 ], [ 0.732, -1.464 ], [ 1.464, 0.9 ] ],
    	[ [ 0.408, 0.536 ], [ -1.084, -0.864 ], [ -0.008, 1.948 ], [ -0.212, -0.004 ], [ 1.796, 1.772 ] ],
    	[ [ 0.928, -0.784 ], [ -1.18, 0.92 ], [ -0.916, -1.012 ], [ -1.32, -0.332 ], [ 0.924, 0.776 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.1408153452953678, negative=24, min=0.776, max=0.776, mean=0.13768, count=50.0, positive=26, stdDev=1.1600521615858488, zeros=0}
    Output: [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0
```
...[skipping 3213 bytes](etc/82.txt)...
```
    00}
    Measured Gradient: [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=200.0, positive=0, stdDev=0.0, zeros=200}
    Gradient Error: [ [ 1.6720000505447388, 0.5920000076293945, -1.9919999837875366, -0.40799999237060547, -0.9279999732971191, 1.0479999780654907, -1.656000018119812, 0.8799999952316284, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ], [ 1.3799999952316284, -1.0199999809265137, -1.840000033378601, -0.5360000133514404, 0.7839999794960022, 0.1599999964237213, 0.30799999833106995, -0.527999997138977, ... ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ... ] ]
    Error Statistics: {meanExponent=-0.1408153464291493, negative=52, min=-0.7760000228881836, max=-0.7760000228881836, mean=-0.06883999684359879, count=200.0, positive=48, stdDev=0.8231642927727829, zeros=100}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=5.1228e-01 +- 6.4800e-01 [0.0000e+00 - 1.9920e+00] (200#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (100#)}
    	at com.simiacryptus.mindseye.test.unit.SingleDerivativeTester.lambda$test$11(SingleDerivativeTester.java:150)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.test.unit.SingleDerivativeTester.test(SingleDerivativeTester.java:183)
    	at com.si
```
...[skipping 3036 bytes](etc/83.txt)...
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



