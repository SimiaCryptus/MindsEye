# StdDevMetaLayer
## StdDevMetaLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.StdDevMetaLayer",
      "id": "a864e734-2f23-44db-97c1-504000002ca2",
      "isFrozen": false,
      "name": "StdDevMetaLayer/a864e734-2f23-44db-97c1-504000002ca2",
      "inputs": [
        "68572b83-5a9f-485e-83b2-ce0c91973a42"
      ],
      "nodes": {
        "489fb1cd-0cc8-47a5-9e1a-330076e856e4": "a864e734-2f23-44db-97c1-504000002ca6",
        "8cda9da7-47d6-4b94-85ed-277af2740db2": "a864e734-2f23-44db-97c1-504000002ca5",
        "6ac486d3-d902-49e2-a8fc-dd352c7565f0": "a864e734-2f23-44db-97c1-504000002ca9",
        "0b81fdfe-b3ee-4ece-9d87-922be33421a3": "a864e734-2f23-44db-97c1-504000002ca8",
        "3459f8b6-2627-4588-a2e8-aa7c351dea68": "a864e734-2f23-44db-97c1-504000002ca7",
        "36f15924-8ead-4222-b713-e6610e48877d": "a864e734-2f23-44db-97c1-504000002ca4",
        "0c75411e-2cde-4ddb-80fb-aee91c27504a": "a864e734-2f23-44db-97c1-504000002ca3"
      },
      "layers": {
        "a864e734-2f23-44db-97c1-504000002ca6": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "a864e734-2f23-44db-97c1-504000002ca6",
          "isFrozen": true,
          "name": "SqActivationLayer/a864e734-2f23-44db-97c1-504000002ca6"
        },
        "a864e734-2f23-44db-97c1-504000002ca5": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "a864e734-2f23-44db-97c1-504000002ca5",
          "isFrozen": false,
          "name": "AvgMetaLayer/a864e734-2f23-44db-97c1-504000002ca5"
        },
        "a864e734-2f23-44db-97c1-504000002ca9": {
          "class": "com.simiacryptus.mindseye.layers.java.AvgMetaLayer",
          "id": "a864e734-2f23-44db-97c1-504000002ca9",
          "isFrozen": false,
          "name": "AvgMetaLayer/a864e734-2f23-44db-97c1-504000002ca9"
        },
        "a864e734-2f23-44db-97c1-504000002ca8": {
          "class": "com.simiacryptus.mindseye.layers.java.SqActivationLayer",
          "id": "a864e734-2f23-44db-97c1-504000002ca8",
          "isFrozen": true,
          "name": "SqActivationLayer/a864e734-2f23-44db-97c1-504000002ca8"
        },
        "a864e734-2f23-44db-97c1-504000002ca7": {
          "class": "com.simiacryptus.mindseye.layers.java.LinearActivationLayer",
          "id": "a864e734-2f23-44db-97c1-504000002ca7",
          "isFrozen": false,
          "name": "LinearActivationLayer/a864e734-2f23-44db-97c1-504000002ca7",
          "weights": {
            "dimensions": [
              2
            ],
            "data": [
              -1.0,
              0.0
            ]
          }
        },
        "a864e734-2f23-44db-97c1-504000002ca4": {
          "class": "com.simiacryptus.mindseye.layers.java.SumInputsLayer",
          "id": "a864e734-2f23-44db-97c1-504000002ca4",
          "isFrozen": false,
          "name": "SumInputsLayer/a864e734-2f23-44db-97c1-504000002ca4"
        },
        "a864e734-2f23-44db-97c1-504000002ca3": {
          "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
          "id": "a864e734-2f23-44db-97c1-504000002ca3",
          "isFrozen": false,
          "name": "NthPowerActivationLayer/a864e734-2f23-44db-97c1-504000002ca3",
          "power": 0.5
        }
      },
      "links": {
        "489fb1cd-0cc8-47a5-9e1a-330076e856e4": [
          "68572b83-5a9f-485e-83b2-ce0c91973a42"
        ],
        "8cda9da7-47d6-4b94-85ed-277af2740db2": [
          "489fb1cd-0cc8-47a5-9e1a-330076e856e4"
        ],
        "6ac486d3-d902-49e2-a8fc-dd352c7565f0": [
          "68572b83-5a9f-485e-83b2-ce0c91973a42"
        ],
        "0b81fdfe-b3ee-4ece-9d87-922be33421a3": [
          "6ac486d3-d902-49e2-a8fc-dd352c7565f0"
        ],
        "3459f8b6-2627-4588-a2e8-aa7c351dea68": [
          "0b81fdfe-b3ee-4ece-9d87-922be33421a3"
        ],
        "36f15924-8ead-4222-b713-e6610e48877d": [
          "8cda9da7-47d6-4b94-85ed-277af2740db2",
          "3459f8b6-2627-4588-a2e8-aa7c351dea68"
        ],
        "0c75411e-2cde-4ddb-80fb-aee91c27504a": [
          "36f15924-8ead-4222-b713-e6610e48877d"
        ]
      },
      "labels": {},
      "head": "0c75411e-2cde-4ddb-80fb-aee91c27504a"
    }
```



### Network Diagram
Code from [LayerTestBase.java:94](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L94) executed in 0.23 seconds: 
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
    [[ 1.072, 1.884, 0.62 ]]
    --------------------
    Output: 
    [ 0.0, 0.0, 0.0 ]
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
    Inputs: [ 1.072, 1.884, 0.62 ]
    Inputs Statistics: {meanExponent=0.03255579110395454, negative=0, min=0.62, max=0.62, mean=1.192, count=3.0, positive=3, stdDev=0.5229557023942535, zeros=0}
    Output: [ 0.0, 0.0, 0.0 ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=3.0, positive=0, stdDev=0.0, zeros=3}
    Feedback for input 0
    Inputs Values: [ 1.072, 1.884, 0.62 ]
    Value Statistics: {meanExponent=0.03255579110395454, negative=0, min=0.62, max=0.62, mean=1.192, count=3.0, positive=3, stdDev=0.5229557023942535, zeros=0}
    Implemented Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=9.0, positive=0, stdDev=0.0, zeros=9}
    Measured Feedback: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=9.0, positive=0, stdDev=0.0, zeros=9}
    Feedback Error: [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=9.0, positive=0, stdDev=0.0, zeros=9}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (9#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    java.lang.RuntimeException: java.lang.RuntimeException: java.util.concurrent.ExecutionException: java.lang.RuntimeException: Frozen component did not pass input backwards
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:61)
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
    	at org.junit.runners.Sui
```
...[skipping 693 bytes](etc/1.txt)...
```
    at com.intellij.rt.execution.junit.JUnitStarter.prepareStreamsAndStart(JUnitStarter.java:242)
    	at com.intellij.rt.execution.junit.JUnitStarter.main(JUnitStarter.java:70)
    Caused by: java.lang.RuntimeException: java.util.concurrent.ExecutionException: java.lang.RuntimeException: Frozen component did not pass input backwards
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$run$9(GpuController.java:219)
    	at com.simiacryptus.util.lang.StaticResourcePool.apply(StaticResourcePool.java:88)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.run(GpuController.java:215)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testFrozen(DerivativeTester.java:181)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.test(DerivativeTester.java:172)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$16(LayerTestBase.java:145)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	... 35 more
    Caused by: java.util.concurrent.ExecutionException: java.lang.RuntimeException: Frozen component did not pass input backwards
    	at java.util.concurrent.FutureTask.report(FutureTask.java:122)
    	at java.util.concurrent.FutureTask.get(FutureTask.java:192)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$run$9(GpuController.java:217)
    	... 42 more
    Caused by: java.lang.RuntimeException: Frozen component did not pass input backwards
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$testFrozen$19(DerivativeTester.java:199)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$null$8(GpuController.java:217)
    	at java.util.concurrent.Executors$RunnableAdapter.call(Executors.java:511)
    	at java.util.concurrent.FutureTask.run(FutureTask.java:266)
    	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
    	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
    	at java.lang.Thread.run(Thread.java:748)
    
```



