# ImgBandBiasLayer
## ImgBandBiasLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ImgBandBiasLayer",
      "id": "f4569375-56fe-4e46-925c-95f4000000e4",
      "isFrozen": false,
      "name": "ImgBandBiasLayer/f4569375-56fe-4e46-925c-95f4000000e4",
      "bias": [
        0.0,
        0.0
      ]
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    	[ [ 1.024, 1.84 ], [ 1.984, 0.312 ], [ -1.396, -1.108 ] ],
    	[ [ -1.536, -1.752 ], [ 0.66, 1.372 ], [ 1.912, -1.804 ] ],
    	[ [ -0.584, 1.18 ], [ -1.488, 1.676 ], [ 0.696, 0.064 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.0240000486373901, 1.840000033378601 ], [ 1.9839999675750732, 0.31200000643730164 ], [ -1.3960000276565552, -1.1080000400543213 ] ],
    	[ [ -1.5360000133514404, -1.7519999742507935 ], [ 0.6600000262260437, 1.371999979019165 ], [ 1.9119999408721924, -1.8040000200271606 ] ],
    	[ [ -0.5839999914169312, 1.1799999475479126 ], [ -1.4880000352859497, 1.6759999990463257 ], [ 0.6959999799728394, 0.06400000303983688 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [
    	[ [ 1.024, 1.84 ], [ 1.984, 0.312 ], [ -1.396, -1.108 ] ],
    	[ [ -1.536, -1.752 ], [ 0.66, 1.372 ], [ 1.912, -1.804 ] ],
    	[ [ -0.584, 1.18 ], [ -1.488, 1.676 ], [ 0.696, 0.064 ] ]
    ]
    Output: [
    	[ [ 1.0240000486373901, 1.840000033378601 ], [ 1.9839999675750732, 0.31200000643730164 ], [ -1.3960000276565552, -1.1080000400543213 ] ],
    	[ [ -1.5360000133514404, -1.7519999742507935 ], [ 0.6600000262260437, 1.371999979019165 ], [ 1.9119999408721924, -1.8040000200271606 ] ],
    	[ [ -0.5839999914169312, 1.1799999475479126 ], [ -1.4880000352859497, 1.6759999990463257 ], [ 0.6959999799728394, 0.06400000303983688 ] ]
    ]
    Measured: [ [ 0.9989738464355469, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.9995698928833008, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0001659393310547, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0001659393310547, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0001659393310547, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0001659393310547, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0001659393310547 ] ]
    Implemented: [ [ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    Error: [ [ -0.001026153564453125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
```
...[skipping 81 bytes](etc/1.txt)...
```
    [ 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -4.3010711669921875E-4, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4 ] ]
    Learning Gradient for weight set 0
    Inputs: [
    	[ [ 1.024, 1.84 ], [ 1.984, 0.312 ], [ -1.396, -1.108 ] ],
    	[ [ -1.536, -1.752 ], [ 0.66, 1.372 ], [ 1.912, -1.804 ] ],
    	[ [ -0.584, 1.18 ], [ -1.488, 1.676 ], [ 0.696, 0.064 ] ]
    ]
    Outputs: [
    	[ [ 1.0240000486373901, 1.840000033378601 ], [ 1.9839999675750732, 0.31200000643730164 ], [ -1.3960000276565552, -1.1080000400543213 ] ],
    	[ [ -1.5360000133514404, -1.7519999742507935 ], [ 0.6600000262260437, 1.371999979019165 ], [ 1.9119999408721924, -1.8040000200271606 ] ],
    	[ [ -0.5839999914169312, 1.1799999475479126 ], [ -1.4880000352859497, 1.6759999990463257 ], [ 0.6959999799728394, 0.06400000303983688 ] ]
    ]
    Measured Gradient: [ [ 1.0001659393310547, 1.0001659393310547, 1.0001659393310547, 1.0001659393310547, 1.0001659393310547, 1.0001659393310547, 1.0001659393310547, 1.0001659393310547, 1.0001659393310547, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0001659393310547 ] ]
    Implemented Gradient: [ [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ] ]
    Error: [ [ 1.659393310546875E-4, 1.659393310546875E-4, 1.659393310546875E-4, 1.659393310546875E-4, 1.659393310546875E-4, 1.659393310546875E-4, 1.659393310546875E-4, 1.659393310546875E-4, 1.659393310546875E-4, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.659393310546875E-4 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.8814e-05 +- 7.4071e-05 [0.0000e+00 - 1.0262e-03] (360#)
    relativeTol: 9.4074e-05 +- 7.5883e-05 [8.4638e-06 - 5.1334e-04] (36#)
    
```

Returns: 

```
    java.lang.RuntimeException: java.lang.RuntimeException: java.util.concurrent.ExecutionException: java.lang.AssertionError: Nonfrozen component not listed in delta. Deltas: []
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:61)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:82)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:156)
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
    	at org.junit.runners.ParentRunner.run(ParentRunner.jav
```
...[skipping 879 bytes](etc/2.txt)...
```
    com.intellij.rt.execution.junit.JUnitStarter.main(JUnitStarter.java:70)
    Caused by: java.lang.RuntimeException: java.util.concurrent.ExecutionException: java.lang.AssertionError: Nonfrozen component not listed in delta. Deltas: []
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$run$8(GpuController.java:215)
    	at com.simiacryptus.util.lang.StaticResourcePool.apply(StaticResourcePool.java:88)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.run(GpuController.java:211)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testUnFrozen(DerivativeTester.java:125)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.test(DerivativeTester.java:92)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$15(LayerTestBase.java:140)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:157)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	... 36 more
    Caused by: java.util.concurrent.ExecutionException: java.lang.AssertionError: Nonfrozen component not listed in delta. Deltas: []
    	at java.util.concurrent.FutureTask.report(FutureTask.java:122)
    	at java.util.concurrent.FutureTask.get(FutureTask.java:192)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$run$8(GpuController.java:213)
    	... 44 more
    Caused by: java.lang.AssertionError: Nonfrozen component not listed in delta. Deltas: []
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$testUnFrozen$17(DerivativeTester.java:142)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$null$7(GpuController.java:213)
    	at java.util.concurrent.Executors$RunnableAdapter.call(Executors.java:511)
    	at java.util.concurrent.FutureTask.run(FutureTask.java:266)
    	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
    	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
    	at java.lang.Thread.run(Thread.java:748)
    
```



