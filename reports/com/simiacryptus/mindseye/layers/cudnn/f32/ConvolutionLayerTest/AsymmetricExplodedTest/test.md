# PipelineNetwork
## AsymmetricExplodedTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.network.PipelineNetwork",
      "id": "370a9587-74a1-4959-b406-fa45000002ec",
      "isFrozen": false,
      "name": "PipelineNetwork/370a9587-74a1-4959-b406-fa45000002ec",
      "inputs": [
        "90e0b710-a341-4000-a285-0d8d51362b93"
      ],
      "nodes": {
        "e8753db3-e98c-4d4c-b793-f1b31a8ce3ee": "370a9587-74a1-4959-b406-fa45000002ee",
        "bc693cfc-3524-4e7b-9904-3dfbaa3e725b": "370a9587-74a1-4959-b406-fa45000002ef",
        "73517307-bb98-48ed-9a6b-63e9f4086c9b": "370a9587-74a1-4959-b406-fa45000002ed"
      },
      "layers": {
        "370a9587-74a1-4959-b406-fa45000002ee": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SimpleConvolutionLayer",
          "id": "370a9587-74a1-4959-b406-fa45000002ee",
          "isFrozen": false,
          "name": "SimpleConvolutionLayer/370a9587-74a1-4959-b406-fa45000002ee",
          "filter": {
            "dimensions": [
              1,
              1,
              4
            ],
            "data": [
              0.452,
              1.74,
              1.716,
              0.14
            ]
          },
          "strideX": 1,
          "strideY": 1,
          "simple": false
        },
        "370a9587-74a1-4959-b406-fa45000002ef": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SimpleConvolutionLayer",
          "id": "370a9587-74a1-4959-b406-fa45000002ef",
          "isFrozen": false,
          "name": "SimpleConvolutionLayer/370a9587-74a1-4959-b406-fa45000002ef",
          "filter": {
            "dimensions": [
              1,
              1,
              4
            ],
            "data": [
              -0.52,
              -0.704,
              -0.552,
              1.78
            ]
          },
          "strideX": 1,
          "strideY": 1,
          "simple": false
        },
        "370a9587-74a1-4959-b406-fa45000002ed": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgConcatLayer",
          "id": "370a9587-74a1-4959-b406-fa45000002ed",
          "isFrozen": false,
          "name": "ImgConcatLayer/370a9587-74a1-4959-b406-fa45000002ed",
          "maxBands": 3
        }
      },
      "links": {
        "e8753db3-e98c-4d4c-b793-f1b31a8ce3ee": [
          "90e0b710-a341-4000-a285-0d8d51362b93"
        ],
        "bc693cfc-3524-4e7b-9904-3dfbaa3e725b": [
          "90e0b710-a341-4000-a285-0d8d51362b93"
        ],
        "73517307-bb98-48ed-9a6b-63e9f4086c9b": [
          "e8753db3-e98c-4d4c-b793-f1b31a8ce3ee",
          "bc693cfc-3524-4e7b-9904-3dfbaa3e725b"
        ]
      },
      "labels": {},
      "head": "73517307-bb98-48ed-9a6b-63e9f4086c9b"
    }
```



### Network Diagram
Code from [LayerTestBase.java:94](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L94) executed in 2.17 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.1.png)



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.01 seconds: 
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
    	[ [ -2.0, -1.024 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -2.685760021209717, -3.57535982131958, 1.7608959674835205 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.04 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (50#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -2.0, -1.024 ] ]
    ]
    Inputs Statistics: {meanExponent=0.1556649761518966, negative=2, min=-1.024, max=-1.024, mean=-1.512, count=2.0, positive=0, stdDev=0.48799999999999966, zeros=0}
    Output: [
    	[ [ -2.685760021209717, -3.57535982131958, 1.7608959674835205 ] ]
    ]
    Outputs Statistics: {meanExponent=0.40937355305063866, negative=2, min=1.7608959674835205, max=1.7608959674835205, mean=-1.5000746250152588, count=3.0, positive=1, stdDev=2.334279882525552, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ -2.0, -1.024 ] ]
    ]
    Value Statistics: {meanExponent=0.1556649761518966, negative=2, min=-1.024, max=-1.024, mean=-1.512, count=2.0, positive=0, stdDev=0.48799999999999966, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, -0.5199999809265137 ], [ 0.0, 0.0, -0.7039999961853027 ] ]
    Implemented Statistics: {meanExponent=-0.2182120077530927, negative=2, min=-0.7039999961853027, max=-0.7039999961853027, mean=-0.20399999618530273, count=6.0, positive=0, stdDev=0.2933484799491496, zeros=4}
    Measured: [ [ 0.4506111145019531, 1.71661376953125, -0.5197525024414062 ], [ 1.7404556274414062, 0.13828277587890625, -0.7033348083496094 ] ]
    Measured Statistics: {meanExponent=-0.19452262729580735, negative=2, min=-0.7033348083496094, max=-0.7033348083496094, mean=0.4704793294270833, count=6.0, positive=4, stdDev=0.9690922597440996, zeros=0}
    Feedback Error: [ [ 0.4506111145019531, 1.71661376953125, 2.474784851074219E-4 ], [ 1.7404556274414062, 0.13828277587890625, 6.651878356933594E-4 ] ]
    Error Statistics: {meanExponent=-1.2522687860952626, negative=0, min=6.651878356933594E-4, max=6.651878356933594E-4, mean=0.6744793256123861, count=6.0, positive=6, stdDev=0.7603275025828713, zeros=0}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=6.7448e-01 +- 7.6033e-01 [2.4748e-04 - 1.7405e+00] (6#), relativeTol=6.6679e-01 +- 4.7124e-01 [2.3802e-04 - 1.0000e+00] (6#)}
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



