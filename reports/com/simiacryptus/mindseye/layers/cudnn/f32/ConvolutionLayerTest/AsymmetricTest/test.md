# ConvolutionLayer
## AsymmetricTest
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
      "id": "a864e734-2f23-44db-97c1-5040000002f3",
      "isFrozen": false,
      "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-5040000002f3",
      "filter": {
        "dimensions": [
          1,
          1,
          8
        ],
        "data": [
          1.392,
          -0.564,
          -1.348,
          1.0,
          1.196,
          -1.148,
          -1.808,
          -0.7
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
    	[ [ 1.692, 1.632 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 4.307136058807373, -2.8278238773345947, -5.231472015380859, 0.5496000647544861 ] ]
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
      "id": "a864e734-2f23-44db-97c1-5040000002fc",
      "isFrozen": false,
      "name": "ConvolutionLayer/a864e734-2f23-44db-97c1-5040000002fc",
      "filter": {
        "dimensions": [
          1,
          1,
          8
        ],
        "data": [
          1.392,
          -0.564,
          -1.348,
          1.0,
          1.196,
          -1.148,
          -1.808,
          -0.7
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
    	[ [ 1.692, 1.632 ] ]
    ]
    Error: [
    	[ [ 5.8807374081482067E-8, 1.2266540494465517E-7, -1.5380859252900336E-8, 6.47544859955218E-8 ] ]
    ]
    Accuracy:
    absoluteTol: 6.5402e-08 +- 3.8161e-08 [1.5381e-08 - 1.2267e-07] (4#)
    relativeTol: 2.2224e-08 +- 2.2439e-08 [1.4700e-09 - 5.8911e-08] (4#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.05 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (60#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.692, 1.632 ] ]
    ]
    Inputs Statistics: {meanExponent=0.2205602565604235, negative=0, min=1.632, max=1.632, mean=1.662, count=2.0, positive=2, stdDev=0.029999999999994646, zeros=0}
    Output: [
    	[ [ 4.307136058807373, -2.8278238773345947, -5.231472015380859, 0.5496000647544861 ] ]
    ]
    Outputs Statistics: {meanExponent=0.38607790747807447, negative=2, min=0.5496000647544861, max=0.5496000647544861, mean=-0.8006399422883987, count=4.0, positive=2, stdDev=3.5935453009682266, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.692, 1.632 ] ]
    ]
    Value Statistics: {meanExponent=0.2205602565604235, negative=0, min=1.632, max=1.632, mean=1.662, count=2.0, positive=2, stdDev=0.029999999999994646, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, -1.3480000495910645, 1.0 ], [ 0.0, 0.0, -1.8079999685287476, -0.699999988079071 ] ]
    Implemented Statistics: {meanExponent=0.05799658984359703, negative=3, min=-0.699999988079071, max=-0.699999988079071, mean=-0.3570000007748604, count=8.0, positive=1, stdDev=0.8333948643036926, zeros=4}
    Measured: [ [ 1.392364501953125, -0.5650520324707031, -1.3494491577148438, 1.0001659393310547 ], [ 1.1920928955078125, -1.1491775512695312, -1.8072128295898438, -0.6997585296630859 ] ]
    Measured Statistics: {meanExponent=0.033090588300669826, negative=5, min=-0.6997585296630859, max=-0.6997585296630859, mean=-0.24825334548950195, count=8.0, positive=3, stdDev=1.177053767883624, zeros=0}
    Feedback Error: [ [ 1.392364501953125, -0.5650520324707031, -0.0014491081237792969, 1.659393310546875E-4 ], [ 1.1920928955078125, -1.1491775512695312, 7.871389389038086E-4, 2.4145841598510742E-4 ] ]
    Error Statistics: {meanExponent=-1.663439689043022, negative=3, min=2.4145841598510742E-4, max=2.4145841598510742E-4, mean=0.10874665528535843, count=8.0, positive=5, stdDev=0.7830272462225176, zeros=0}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=5.3767e-01 +- 5.7955e-01 [1.6594e-04 - 1.3924e+00] (8#), relativeTol=5.0013e-01 +- 4.9987e-01 [8.2963e-05 - 1.0000e+00] (8#)}
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



