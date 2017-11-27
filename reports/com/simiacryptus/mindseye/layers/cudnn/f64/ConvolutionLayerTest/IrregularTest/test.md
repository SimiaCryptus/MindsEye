# ConvolutionLayer
## IrregularTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.05 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer",
      "id": "161debb2-61c2-40b5-a2dd-73da00000002",
      "isFrozen": false,
      "name": "ConvolutionLayer/161debb2-61c2-40b5-a2dd-73da00000002",
      "filter": {
        "dimensions": [
          1,
          1,
          6
        ],
        "data": [
          -0.256,
          1.584,
          -1.112,
          1.944,
          1.224,
          1.924
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.02 seconds: 
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
    	[ [ -1.236, -1.668 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -2.926176, -3.999456, -1.8347999999999995 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.76 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "161debb2-61c2-40b5-a2dd-73da0000000c",
      "isFrozen": false,
      "name": "ConvolutionLayer/161debb2-61c2-40b5-a2dd-73da0000000c",
      "filter": {
        "dimensions": [
          1,
          1,
          6
        ],
        "data": [
          -0.256,
          1.584,
          -1.112,
          1.944,
          1.224,
          1.924
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
    	[ [ -1.236, -1.668 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (3#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (3#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: ConvolutionLayer/161debb2-61c2-40b5-a2dd-73da00000002
    Inputs: [
    	[ [ -1.236, -1.668 ] ]
    ]
    Outputs: [
    	[ [ -2.926176, -3.999456, -1.8347999999999995 ] ]
    ]
    Measured Gradient: [ [ -1.2359999999933533, 0.0, 0.0 ], [ 0.0, -1.2359999999933533, 0.0 ], [ 0.0, 0.0, -1.2360000000155578 ], [ -1.6679999999880122, 0.0, 0.0 ], [ 0.0, -1.6679999999880122, 0.0 ], [ 0.0, 0.0, -1.668000000032421 ] ]
    Implemented Gradient: [ [ -1.236, 0.0, 0.0 ], [ 0.0, -1.236, 0.0 ], [ 0.0, 0.0, -1.236 ], [ 0.0, 0.0, 0.0 ], [ 0.0, -1.668, 0.0 ], [ 0.0, 0.0, -1.668 ] ]
    [ [ 6.646683203825887E-12, 0.0, 0.0 ], [ 0.0, 6.646683203825887E-12, 0.0 ], [ 0.0, 0.0, -1.5557777288677244E-11 ], [ -1.6679999999880122, 0.0, 0.0 ], [ 0.0, 1.198774413069259E-11, 0.0 ], [ 0.0, 0.0, -3.242117685431367E-11 ] ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=9.2667e-02 +- 3.8207e-01 [0.0000e+00 - 1.6680e+00] (18#), relativeTol=1.6667e-01 +- 3.7268e-01 [2.6888e-12 - 1.0000e+00] (6#)}
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testLearning(DerivativeTester.java:254)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$test$2(DerivativeTester.java:77)
    	at java.util.stream.IntPipeline$4$1.accept(IntPipeline.java:250)
    	at java.util.stream.Streams$RangeIntSpliterator.forEachRemaining(Streams.java:110)
    	at java.util.Spliterator$OfInt.forEachRemaining(Spliterator.java:693)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.test(DerivativeTester.java:78)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$15(LayerTestBase.java:140)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:142)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:77)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:141)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:139)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:69)
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
    	at org.junit.runner.JUnitCore.run(JUnitCore.java:137)
    	at com.intellij.junit4.JUnit4IdeaTestRunner.startRunnerWithArgs(JUnit4IdeaTestRunner.java:68)
    	at com.intellij.rt.execution.junit.IdeaTestRunner$Repeater.startRunnerWithArgs(IdeaTestRunner.java:47)
    	at com.intellij.rt.execution.junit.JUnitStarter.prepareStreamsAndStart(JUnitStarter.java:242)
    	at com.intellij.rt.execution.junit.JUnitStarter.main(JUnitStarter.java:70)
    
```



