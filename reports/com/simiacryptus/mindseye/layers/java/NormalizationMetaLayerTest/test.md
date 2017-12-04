# NormalizationMetaLayer
## NormalizationMetaLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.NormalizationMetaLayer",
      "id": "a864e734-2f23-44db-97c1-504000002c3b",
      "isFrozen": false,
      "name": "NormalizationMetaLayer/a864e734-2f23-44db-97c1-504000002c3b",
      "inputs": [
        "c0b87c46-32f2-4572-9de0-8fccc54c3236"
      ],
      "nodes": {
        "3d8e14d0-dc4d-47bd-b77d-ca8a1f4e1f9b": "a864e734-2f23-44db-97c1-504000002c40",
        "63ffb810-21e3-4649-b372-505d0718fa08": "a864e734-2f23-44db-97c1-504000002c3f",
        "12e1f640-c389-43d4-a28c-3782c41507fa": "a864e734-2f23-44db-97c1-504000002c3e",
        "8a8310f3-d022-4d8b-a25c-e8b66c902e80": "a864e734-2f23-44db-97c1-504000002c3d",
        "429763a9-ebcd-4dd9-a3ed-a3a7887206a2": "a864e734-2f23-44db-97c1-504000002c3c"
      },
      "layers": {
        "a864e734-2f23-44db-97c1-504000002c40": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "a864e734-2f23-44db-97c1-504000002c40",
          "isFrozen": true,
          "name": "SqActivationLayer/a864e734-2f23-44db-97c1-504000002c40"
        },
        "a864e734-2f23-44db-97c1-504000002c3f": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgReducerLayer",
          "id": "a864e734-2f23-44db-97c1-504000002c3f",
          "isFrozen": false,
          "name": "AvgReducerLayer/a864e734-2f23-44db-97c1-504000002c3f"
        },
        "a864e734-2f23-44db-97c1-504000002c3e": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "a864e734-2f23-44db-97c1-504000002c3e",
          "isFrozen": false,
          "name": "AvgMetaLayer/a864e734-2f23-44db-97c1-504000002c3e"
        },
        "a864e734-2f23-44db-97c1-504000002c3d": {
          "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
          "id": "a864e734-2f23-44db-97c1-504000002c3d",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/a864e734-2f23-44db-97c1-504000002c3d",
          "power": -0.5
        },
        "a864e734-2f23-44db-97c1-504000002c3c": {
          "class": "com.simiacryptus.mindseye.layers.java.ProductInputsLayer",
          "id": "a864e734-2f23-44db-97c1-504000002c3c",
          "isFrozen": false,
          "name": "ProductInputsLayer/a864e734-2f23-44db-97c1-504000002c3c"
        }
      },
      "links": {
        "3d8e14d0-dc4d-47bd-b77d-ca8a1f4e1f9b": [
          "c0b87c46-32f2-4572-9de0-8fccc54c3236"
        ],
        "63ffb810-21e3-4649-b372-505d0718fa08": [
          "3d8e14d0-dc4d-47bd-b77d-ca8a1f4e1f9b"
        ],
        "12e1f640-c389-43d4-a28c-3782c41507fa": [
          "63ffb810-21e3-4649-b372-505d0718fa08"
        ],
        "8a8310f3-d022-4d8b-a25c-e8b66c902e80": [
          "12e1f640-c389-43d4-a28c-3782c41507fa"
        ],
        "429763a9-ebcd-4dd9-a3ed-a3a7887206a2": [
          "c0b87c46-32f2-4572-9de0-8fccc54c3236",
          "8a8310f3-d022-4d8b-a25c-e8b66c902e80"
        ]
      },
      "labels": {},
      "head": "429763a9-ebcd-4dd9-a3ed-a3a7887206a2"
    }
```



### Network Diagram
Code from [LayerTestBase.java:94](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L94) executed in 0.20 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.1.png)



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    [[ -1.488, 1.792, -1.084 ]]
    --------------------
    Output: 
    [ -1.0031741328743267, 1.208123686902415, -0.7308069623896305 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ -1.488, 1.792, -1.084 ]
    Inputs Statistics: {meanExponent=0.15365673957944484, negative=2, min=-1.084, max=-1.084, mean=-0.26, count=3.0, positive=1, stdDev=1.4603269040412379, zeros=0}
    Output: [ -1.0031741328743267, 1.208123686902415, -0.7308069623896305 ]
    Outputs Statistics: {meanExponent=-0.017569866404117413, negative=2, min=-0.7308069623896305, max=-0.7308069623896305, mean=-0.17528580278718073, count=3.0, positive=1, stdDev=0.9845175911791794, zeros=0}
    Feedback for input 0
    Inputs Values: [ -1.488, 1.792, -1.084 ]
    Value Statistics: {meanExponent=0.15365673957944484, negative=2, min=-1.084, max=-1.084, mean=-0.26, count=3.0, positive=1, stdDev=1.4603269040412379, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=9.0, positive=0, stdDev=0.0, zeros=9}
    Measured: [ [ 0.6741761645656652, 0.0, 0.0 ], [ 0.0, 0.6741761645656652, 0.0 ], [ 0.0, 0.0, 0.6741761645656652 ] ]
    Measured Statistics: {meanExponent=-0.1712266059838293, negative=0, min=0.6741761645656652, max=0.6741761645656652, mean=0.22472538818855506, count=9.0, positive=3, stdDev=0.3178096917858131, zeros=6}
    Feedback Error: [ [ 0.6741761645656652, 0.0, 0.0 ], [ 0.0, 0.6741761645656652, 0.0 ], [ 0.0, 0.0, 0.6741761645656652 ] ]
    Error Statistics: {meanExponent=-0.1712266059838293, negative=0, min=0.6741761645656652, max=0.6741761645656652, mean=0.22472538818855506, count=9.0, positive=3, stdDev=0.3178096917858131, zeros=6}
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=2.2473e-01 +- 3.1781e-01 [0.0000e+00 - 6.7418e-01] (9#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (3#)}
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
    	at sun.reflect.GeneratedMethodAccessor1.invoke(Unknown Source)
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



