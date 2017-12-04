# ConvolutionLayer
## IrregularTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer",
      "id": "a864e734-2f23-44db-97c1-504000000351",
      "isFrozen": false,
      "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-504000000351",
      "filter": {
        "dimensions": [
          1,
          1,
          6
        ],
        "data": [
          1.968,
          -1.544,
          -1.708,
          -1.232,
          -1.232,
          1.912
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 0.06, -0.744 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.0346879959106445, 0.8239679932594299, -1.525007963180542 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:131](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L131) executed in 0.02 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "a864e734-2f23-44db-97c1-50400000035a",
      "isFrozen": false,
      "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-50400000035a",
      "filter": {
        "dimensions": [
          1,
          1,
          6
        ],
        "data": [
          1.968,
          -1.544,
          -1.708,
          -1.232,
          -1.232,
          1.912
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
    Inputs: [
    	[ [ 0.06, -0.744 ] ]
    ]
    Error: [
    	[ [ -4.089355520875415E-9, -6.740570102081733E-9, 3.681945792699537E-8 ] ]
    ]
    Accuracy:
    absoluteTol: 1.5883e-08 +- 1.4844e-08 [4.0894e-09 - 3.6819e-08] (3#)
    relativeTol: 6.0461e-09 +- 4.3474e-09 [1.9761e-09 - 1.2072e-08] (3#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.04 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.06, -0.744 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.6751379070352388, negative=1, min=-0.744, max=-0.744, mean=-0.34199999999999997, count=2.0, positive=1, stdDev=0.4020000000000001, zeros=0}
    Output: [
    	[ [ 1.0346879959106445, 0.8239679932594299, -1.525007963180542 ] ]
    ]
    Outputs Statistics: {meanExponent=0.037997288017426065, negative=1, min=-1.525007963180542, max=-1.525007963180542, mean=0.11121600866317749, count=3.0, positive=2, stdDev=1.1601788351819904, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.06, -0.744 ] ]
    ]
    Value Statistics: {meanExponent=-0.6751379070352388, negative=1, min=-0.744, max=-0.744, mean=-0.34199999999999997, count=2.0, positive=1, stdDev=0.4020000000000001, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, -1.7079999446868896 ], [ 0.0, 0.0, 1.9119999408721924 ] ]
    Implemented Statistics: {meanExponent=0.2569878633990914, negative=1, min=1.9119999408721924, max=1.9119999408721924, mean=0.033999999364217125, count=6.0, positive=1, stdDev=1.0461095851777897, zeros=4}
    Measured: [ [ 1.9681453704833984, -1.5437602996826172, -1.7082691192626953 ], [ -1.2314319610595703, -1.2320280075073242, 1.9121170043945312 ] ]
    Measured Statistics: {meanExponent=0.196289799479846, negative=4, min=1.9121170043945312, max=1.9121170043945312, mean=-0.3058711687723796, count=6.0, positive=2, stdDev=1.5970880553493316, zeros=0}
    Feedback Error: [ [ 1.9681453704833984, -1.5437602996826172, -2.6917457580566406E-4 ], [ -1.2314319610595703, -1.2320280075073242, 1.1706352233886719E-4 ] ]
    Error Statistics: {meanExponent=-1.1396460553484191, negative=4, min=1.1706352233886719E-4, max=1.1706352233886719E-4, mean=-0.3398711681365967, count=6.0, positive=2, stdDev=1.1970822422880627, zeros=0}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=9.9596e-01 +- 7.4605e-01 [1.1706e-04 - 1.9681e+00] (6#), relativeTol=6.6668e-01 +- 4.7138e-01 [3.0612e-05 - 1.0000e+00] (6#)}
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$test$6(DerivativeTester.java:90)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.test(DerivativeTester.java:121)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$16(LayerTestBase.java:145)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:83)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:133)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:144)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:68)
    	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    	at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
    	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
    	at java.lang.reflect.Method.invoke(Method.java:498)
    	at org.junit.runners.model.FrameworkMethod$1.runReflectiveCall(FrameworkMethod.java:50)
    	at org.junit.internal.runners.model.ReflectiveCallable.run(ReflectiveCallable.java:12)
    	at org.junit.runners.model.FrameworkMethod.invokeExplosively(FrameworkMethod.java:47)
    	at org.junit.internal.runners.statements.InvokeMethod.evaluate(InvokeMethod.java:17)
    	at org.junit.runners.ParentRunner.runLeaf(ParentRunner.java:325)
    	at org.junit.runners.BlockJUnit4ClassRunner.runChild(BlockJUnit4ClassRunner.java:78)
    	at org.junit.runners.BlockJUnit4ClassRunner.runChild(BlockJUnit4ClassRunner.java:57)
    	at org.junit.runners.ParentRunner$3.run(ParentRunner.java:290)
    	at org.junit.runners.ParentRunner$1.schedule(ParentRunner.java:71)
    	at org.junit.runners.ParentRunner.runChildren(ParentRunner.java:288)
    	at org.junit.runners.ParentRunner.access$000(ParentRunner.java:58)
    	at org.junit.runners.ParentRunner$2.evaluate(ParentRunner.java:268)
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



